from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from src.csv_pages_to_names import (
    book_to_directory,
    file_paths_to_names,
    flatten_page_names,
)
from src.similarity_matrix import create_sim_m, create_slices


def book_id_to_path(id, book_id_to_book, csv_data_directory):
    book_info = book_id_to_book[id]
    start_year = book_info["Start_year"]
    end_year = book_info["End_year"]
    source = book_info["source"].lower()

    return Path(csv_data_directory).glob(
        f"*{book_info["parish_normalized"]}/*{start_year}-{end_year}_{source}"
    )


def check_years(start_year_a, end_year_a, start_year_b, end_year_b):
    if start_year_a == start_year_b and end_year_a == end_year_b:
        return "full"

    overlap_start = max(start_year_a, start_year_b)
    overlap_end = min(end_year_a, end_year_b)

    if overlap_start < overlap_end:

        if (start_year_a < start_year_b and end_year_a > end_year_b) or (
            start_year_b < start_year_a and end_year_b > end_year_a
        ):
            return "partial_center"
        else:
            return "partial_left_or_right"
    else:
        return "no_overlap"


def parish_check_overlaps_by_year(parish_books: list[dict]):
    checked_idxs = []
    overlapping_books = {
        "full": [],
        "partial_center": [],
        "partial_left_or_right": [],
    }

    for i, book_a in enumerate(parish_books):
        for j, book_b in enumerate(parish_books):
            if j in checked_idxs:
                continue
            if (
                book_a["in/out"] == book_b["in/out"]
                and "mixed" not in book_a["in/out"]
            ):
                continue

            overlap_type = check_years(
                book_a["Start_year"],
                book_a["End_year"],
                book_b["Start_year"],
                book_b["End_year"],
            )

            if overlap_type != "no_overlap":
                overlapping_books[overlap_type].append((book_a, book_b))

        checked_idxs.append(i)

    return overlapping_books


def check_churchbook_overlaps_by_year(
    parish_ids: list[int], churchbook_metadata: dict[list[dict]]
):
    overlapping_books = {
        "full": [],
        "partial_center": [],
        "partial_left_or_right": [],
    }

    for p_id in parish_ids:
        parish_overlaps = parish_check_overlaps_by_year(
            churchbook_metadata[p_id]
        )

        for overlap_type in overlapping_books:
            overlapping_books[overlap_type] += parish_overlaps[overlap_type]

    return overlapping_books
