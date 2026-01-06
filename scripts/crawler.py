import argparse
import json
import random
import time

import requests

# --- Constants ---
BASE_URL = "https://api.mangadex.org"


def _get_tag_list() -> dict:
    """Recover taglist from mangadex.

    Raises:
        TypeError: When returned json is not evaluated as a python dictionary.
        RuntimeError: When returned json is invalid.

    Returns:
        dict: Contains the tag information from mangadex, cleaned up
    """
    try:
        response = requests.get(f"{BASE_URL}/manga/tag")
        response.raise_for_status()
        data_response = response.json()
        if isinstance(data_response, dict):
            return _clean_tag_list(data_response)

        else:
            raise TypeError("Unexpected Json format on the response.")

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Error Invalid response, original error message was {str(exc)}"
        ) from exc


def _clean_tag_list(data: dict) -> dict:
    """Cleans mangadex taglist

    Args:
        data(dict): taglist dict to be cleaned
    Returns:
        dict: Key pair dictionary with tag name:id
    """
    output_dict = {}
    for tag in data["data"]:
        output_dict[tag["attributes"]["name"]["en"]] = tag["id"]
    return output_dict


def _get_manga(included_tags: list, excluded_tags: list, offset: int) -> dict:
    """Recovers manga from api

    Args:
        included_tags (list): list of allowed tags
        excluded_tags (list): list of excluded tags
        offset (int): offset to handle pagination

    Raises:
        TypeError: When returned json is not evaluated as a python dictionary.
        RuntimeError: When returned json is invalid.

    Returns:
        dict: Dictionary containing the returned collection.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/manga",
            params={
                "includedTags[]": included_tags,
                "excludedTags[]": excluded_tags,
                "limit": 100,
                "offset": offset,
                "includes[]": ["author", "artist"],
                "originalLanguage[]": ["ja"],
            },
        )
        response.raise_for_status()
        response_data = response.json()
        if isinstance(response_data, dict):
            return response_data
        else:
            raise TypeError("Unexpected Json format on the response.")

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Error retrieving manga from API. Original error: {str(exc)}"
        ) from exc


def _args_cleanup(taglist: str) -> list:
    """Converts from comma separated tags into a cleaned list

    Removes front and tail spaces, and each tag individually.

    Args:
        taglist(str): str with comma separated tags
    Returns:
        list: clean list
    """

    return [name.strip() for name in taglist.strip().split(",") if name.strip()]


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mangadex manga crawler. Finds manga based on a tag list.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 1. Included tags query
    parser.add_argument(
        "--included-tags",
        type=str,
        default="Mystery",
        help="Tags to include on mangadex API query",
    )
    # 2. Excluded tags
    parser.add_argument(
        "--excluded-tags",
        type=str,
        default="",
        help="Tags to exclude on mangadex API query",
    )
    return parser.parse_args()


def _process(args: argparse.Namespace):
    included_tag_names = _args_cleanup(taglist=args.included_tags)
    excluded_tag_names = _args_cleanup(taglist=args.excluded_tags)
    print("--- MangaDex Crawler Initializing ---")
    print(f"Requested Included Tags: {included_tag_names}")
    print(f"Requested Excluded Tags: {excluded_tag_names}")
    tags = _get_tag_list()
    try:
        included_tag_ids = [tags[tag] for tag in included_tag_names]
        excluded_tag_ids = [tags[tag] for tag in excluded_tag_names]
    except KeyError as exc:
        raise RuntimeError(
            f"Failed loading tag id's. Included or Excluded tags not found on mangadex. Error message was: {str(exc)}"
        ) from exc

    print(f"Successfully mapped {len(included_tag_ids)} included tags to IDs.")

    finished = False
    page_count = 0
    total_manga_processed = 0
    offset = 0

    while not finished:
        print(f"\n[PAGE {page_count + 1}] Fetching manga data with OFFSET: {offset}")
        response = _get_manga(
            included_tags=included_tag_ids,
            excluded_tags=excluded_tag_ids,
            offset=offset,
        )
        json_lines = ""
        for manga in response["data"]:
            to_write = {
                "id": manga["id"],
                "status": manga["attributes"]["status"],
                "year": manga["attributes"]["year"],
                "tags": [
                    tag["attributes"]["name"]["en"]
                    for tag in manga["attributes"]["tags"]
                ],
            }

            title = manga["attributes"]["title"].get("en")
            if not title:
                title = manga["attributes"]["title"].get("ja-ro")
            if not title:
                continue
            to_write["title"] = title

            description = manga["attributes"]["description"].get("en")

            if not description:
                continue

            to_write["description"] = description

            for relation in manga["relationships"]:
                if relation["type"] == "author" and "attributes" in relation:
                    to_write["author"] = relation["attributes"].get("name", "Unknown")

                if relation["type"] == "artist" and "attributes" in relation:
                    to_write["artist"] = relation["attributes"].get("name", "Unknown")

            json_lines += f"{json.dumps(to_write)}\n"
            total_manga_processed += 1

        with open("mangas_index.jsonl", "a", encoding="utf-8") as f:
            f.write(json_lines)

        offset += 100
        page_count += 1
        wait = random.uniform(2, 5)
        time.sleep(wait)

        if response.get("total", 0) <= offset:
            finished = True

    print("\n[PROCESS COMPLETE]")
    print(f"Total Manga entries written: {total_manga_processed}")


if __name__ == "__main__":
    args = _parse_args()
    _process(args)
