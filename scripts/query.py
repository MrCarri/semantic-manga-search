""" """

import argparse
from string import Template

import chromadb
import requests
from sentence_transformers import SentenceTransformer

# --- Templates ---

BASE_PROMPT = Template(
    """
        # Role: Expert librarian, specialized in manga research. Based on the following context:

        ## User context
        $user_query

        ## Font context
        $result

        # Instructions
        Your task is to **analyze all manga descriptions and tags** provided in the Font context. From that analysis,
        select the 5 manga that best match the User context and elaborate a short description on them.

        # Response Rules
        - You will have to provide an accurate recommendation based on font context and user context *only*
        - **Output Format:** Present each of the 5 results clearly, detailing their Title and Author, and include a brief summary (1-2 sentences)
        explaining why they match the user's query.
        - Tags and manga descriptions must be taken into account.
        - Tags must have more priority than description in evaluations.
        - If result is not conclusive  you must say so.
        - **CRITICAL RULE: DO NOT INVENT FACTS.** If a user requirement is not explicitly mentioned, you MUST NOT include it in your justification.
        - Direct responses only, without preambles, goodbyes and extra info.
        - Tone must be neutral and objective
        - Speak in english
        """
)

# --- Validation functions ---


def _validate_non_empty_string(value):
    """Checks that the str is not empty and removes starting/ending spaces."""
    if not value.strip():
        raise argparse.ArgumentTypeError("User query cannot be empty")
    return value.strip()


def _validate_positive_integer(value: str) -> int:
    """Checks that the value is a positive integer (> 0)."""
    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(
                f"Limit must be a positive integer (> 0), received {ivalue}."
            )
        return ivalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer.")


def _validate_temperature(value: str) -> float:
    """Checks that the temperature value is between 0.0 and 1.0."""
    try:
        fvalue = float(value)
        if not (0.0 <= fvalue <= 1.0):
            raise argparse.ArgumentTypeError(
                f"Temperature must be between 0.0 and 1.0, received {fvalue}."
            )
        return fvalue
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid float.")


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manga RAG searcher. Finds manga based on a user query using Ollama and ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 1. User query
    parser.add_argument(
        "query",
        type=_validate_non_empty_string,
        help="The natural language prompt describing the manga you are looking for.",
    )
    # 2. Database path
    parser.add_argument(
        "--db-path",
        type=str,
        default="./manga_rag_db",
        help="Path to the ChromaDB directory (e.g., './my_data/db').",
    )

    # 3. Collection name (on chroma DB)
    parser.add_argument(
        "--collection",
        type=str,
        default="mystery_manga",
        help="Name of the ChromaDB collection to query.",
    )

    # 4. Limit results (how many mangas are being considered)
    parser.add_argument(
        "--limit",
        type=_validate_positive_integer,
        default=15,
        help="Number of documents to retrieve from ChromaDB.",
    )

    # 5. LLM model for Ollama
    parser.add_argument(
        "--model",
        type=str,
        default="llama3:8b",
        help="The LLM model name to use from Ollama (e.g., 'llama3:70b').",
    )

    # 6. LLM temperature
    parser.add_argument(
        "--temperature",
        type=_validate_temperature,
        default=0.2,
        help="LLM generation temperature (0.0 to 1.0). Controls randomness.",
    )

    # 7. Ollama API URL
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Full URL for the Ollama API generation endpoint.",
    )
    # 8. Embedding model
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Embedding model from sentence-transformes to be used.",
    )
    # 9. Output tokens
    parser.add_argument(
        "--num-predict",
        type=_validate_positive_integer,
        default=800,
        help="Number of output tokens allowed for response",
    )

    return parser.parse_args()


def _generate_str_result(result: dict) -> str:
    """Build prompt str

    Construct concatenated str from query results.

    Args:
        result(dict): query result dictionary to process

    Returns:
        str: result string of concat results
    """

    try:
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Error. No manga descriptions or metadata were retrieved from vector database. Error was {str(exc)}"
        ) from exc

    result_string = ""
    for i in range(len(documents)):
        # ChromaDB query guarantees that documents and metadata are the same size, so it's access here.
        doc_text = documents[i]
        meta = metadatas[i]
        manga_block = f"""
        [RESULT {i + 1}] {meta.get("title", "N/A")} ({meta.get("author", "N/A")}). Tags: {meta.get("tags", "N/A")}.
        {doc_text}
        """
        result_string += manga_block
    return result_string


def _generate_payload(
    model: str,
    query: str,
    result_str: str,
    temperature: float,
    num_predict: int,
) -> dict:
    """Build payload dictionary

    Build payload filling base prompt template with result str and user query

    Args:
        model(str): Reasoning model to be used
        query(str): user query string added to prompt
        result_str(str): concatenated result string added to prompt
        temperature(float): temperature for the query. More temperature, more freedom.
        num_predict(int): Number for output tokens.
    Returns:
        dict: created dict to be used on request.
    """
    templatized_prompt = BASE_PROMPT.substitute(
        {"user_query": query, "result": result_str}
    )
    output_dict = {
        "model": model,
        "prompt": templatized_prompt,
        "stream": False,
        "temperature": temperature,
        "options": {"num_predict": num_predict},
    }
    return output_dict


def _run_rag(args: argparse.Namespace, embedding_model: SentenceTransformer):
    """Run rag operation

    Run a rag operation.

    Args:
        args(argparse.Namespace): Args to run the operation
        embedding_model(SentenceTransformer): Embedding model to be used with user query
    """
    user_query = args.query
    print(f"Searching for: '{user_query}'\n")

    query_embedding = embedding_model.encode([user_query]).tolist()
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=args.limit,
        include=["documents", "metadatas"],
    )

    payload = _generate_payload(
        model=args.model,
        query=user_query,
        result_str=_generate_str_result(result=dict(results)),
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    try:
        response = requests.post(args.ollama_url, json=payload)
        response.raise_for_status()
        json_response = response.json()
        final_response = json_response.get("response", "Error processing response.")
        print(f"{args.model} Response: {final_response} \n")

    except (
        requests.exceptions.RequestException,
        requests.exceptions.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"Ollama connection error or non-200 HTTP status. Detail: {str(exc)}"
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            f"Unexpected Error during retrieval of results. Original exception message was: {str(exc)}"
        ) from exc


if __name__ == "__main__":
    args = _parse_args()
    print("--- Manga recomendation expert initializing ---")
    print(f"Loading embedding model: {args.embedding_model}...")
    try:
        EMBEDDING_MODEL = SentenceTransformer(args.embedding_model)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model: {args.embedding_model}. Detail: {exc}"
        ) from exc
    _run_rag(args=args, embedding_model=EMBEDDING_MODEL)
