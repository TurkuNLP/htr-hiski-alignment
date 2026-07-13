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
        checked_idxs.append(i)
        for j, book_b in enumerate(parish_books):
            if j in checked_idxs:
                continue
            book_a_in_out = book_a["in/out"]
            book_b_in_out = book_b["in/out"]
            if (
                book_a_in_out["in"] is not book_b_in_out["in"]
                and book_a_in_out["out"] is not book_b_in_out["out"]
                and not book_a_in_out["misc"]
                and not book_a_in_out["misc"]
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


def books_sim_m(candidate_pair, base_dir, block_size=1, step=1, use_whole_pages=False):
    book_a = book_to_directory(candidate_pair[0])
    book_b = book_to_directory(candidate_pair[1])

    book_a_dir = base_dir / book_a
    book_b_dir = base_dir / book_b

    csv_files_a = [str(path) for path in book_a_dir.glob("*.csv")]
    csv_files_b = [str(path) for path in book_b_dir.glob("*.csv")]

    if len(csv_files_a) == 0 or len(csv_files_b) == 0:
        return np.array([[]]), [], [], [], []

    pages_a, names_a = file_paths_to_names(csv_files_a)
    pages_b, names_b = file_paths_to_names(csv_files_b)

    names_flat_b = flatten_page_names(names_b)
    names_flat_a = flatten_page_names(names_a)

    page_lens_a = [len(p) for p in names_a]
    page_lens_b = [len(p) for p in names_b]

    if use_whole_pages:
        slices_a = names_a
        slices_b = names_b
    else:
        block_size = min(block_size, len(names_flat_a), len(names_flat_b))
        slices_a, slices_b, ranges_a, ranges_b = create_slices(
            names_flat_a,
            names_flat_b,
            a_block_size=block_size,
            b_block_size=block_size,
            a_step=step,
            b_step=step,
        )

    sim_m, individual_scores = create_sim_m(
        slices_a,
        slices_b,
        scorer=fuzz.ratio,
        join_char="\n",
        progressbar=False,
    )

    return sim_m, slices_a, slices_b, page_lens_a, page_lens_b


def check_book_overlap(
    sim_m, threshold, n_overlaps_required, block_size=1, step=1
):
    if (
        len(overlaps := np.column_stack(np.where(sim_m >= threshold)))
        >= n_overlaps_required
    ):
        return True, overlaps
    else:
        return False, []
