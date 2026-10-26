from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain, product
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from rapidfuzz import distance, process


# Config
PageKey = tuple[str, int, str]

@dataclass(frozen=True)
class DuplicateConfig:
    """
    threshold scale must match the scorer:
      normalized scorers (e.g. distance.Levenshtein.normalized_similarity) -> 0.0-1.0
      fuzz.* scorers (e.g. fuzz.ratio, fuzz.token_sort_ratio)              -> 0-100

    dtype: score matrix dtype passed to rapidfuzz.process.cdist.
    """

    max_year_diff: int = 10
    correct_year_prediction_bounds: tuple = (1500, 1980)
    threshold: float = 0.6
    scorer: Callable = distance.Levenshtein.normalized_similarity
    max_duplicate_of_entries: int = 3
    all_directions: tuple = ("in", "out", "both", "?")
    join_char: str = ""
    n_workers: int = -1
    output_columns: tuple = (
        "row_data", "row_columns", "row_columns_categorized",
        "possible_name", "possibly_related_toponyms",
        "has_information",
        "year_prediction",
        "book_year_start", "book_year_end",
        "duplicate", "duplicate_of",
        "page_n", "file_row", "in/out", "side",
        "parish_id", "book", "print_type", "book_source",
        "image_range_start", "image_range_end",
        "file_path",
    )
    dtype: type = np.float32


# Data loading
def load_used_data(
    directions_csv: str | Path, book_ids_json: str | Path
) -> tuple[pd.DataFrame, dict]:
    book_migration_directions = pd.read_csv(directions_csv)
    with open(book_ids_json, "r") as fp:
        book_id_to_info = json.load(fp)
    return book_migration_directions, book_id_to_info


def load_parish_df(parish_jsonl: str | Path) -> pd.DataFrame:
    parish_df = pd.read_json(parish_jsonl, lines=True)
    parish_df["duplicate"] = False
    parish_df["duplicate_of"] = None
    parish_df["is_ap"] = parish_df["book_source"].str.contains("ap")
    return parish_df


# Duplicate discovery preparations
def directions_eligible(direction_1: str, direction_2: str) -> bool:
    """False only for a strict "in" vs "out" pair; "both" and "?" match anything."""
    strict = {"in", "out"}
    return not (
        direction_1 in strict
        and direction_2 in strict
        and direction_1 != direction_2
    )

def build_valid_direction_pairs(all_directions) -> set[tuple[str, str]]:
    return {
        (direction_1, direction_2)
        for direction_1, direction_2 in product(all_directions, repeat=2)
        if directions_eligible(direction_1, direction_2)
    }

def build_page_records(parish_df: pd.DataFrame, join_char: str = "") -> list[dict]:
    """
    One record per (book, page_n, side). Each record carries the joined page
    text and its character Counter, computed once and reused for every
    comparison involving the page.
    """
    aggregated = (
        parish_df.groupby(["book", "page_n", "side"], as_index=False)
        .agg(
            book_source=("book_source", "first"),
            is_ap=("is_ap", "first"),
            in_out=("in/out", "first"),
            year_prediction=("year_prediction", "first"),
            row_data_list=(
                "row_data",
                lambda x: list(chain.from_iterable(x)),
            ),
            row_indices=("row_data", lambda x: x.index.tolist()),
        )
    )
    pages = aggregated.to_dict(orient="records")
    for p in pages:
        p["text"] = join_char.join(p["row_data_list"])
        p["char_counter"] = Counter(p["text"])
    return pages

def group_pages_by_book(pages: list[dict]) -> dict[str, list[dict]]:
    book_to_pages: dict[str, list] = defaultdict(list)
    for p in pages:
        book_to_pages[p["book"]].append(p)
    return book_to_pages

def group_by_direction_and_year(
    ref_pages: list[dict],
) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list] = defaultdict(list)
    for ref_p in ref_pages:
        grouped[(ref_p["in_out"], ref_p["year_prediction"])].append(ref_p)
    return grouped


def group_by_direction(
    cand_pages: list[dict],
) -> dict[str, list[dict]]:
    by_dir: dict[str, list] = defaultdict(list)
    for page in cand_pages:
        by_dir[page["in_out"]].append(page)

    return by_dir

def compute_book_order(pages: list[dict]) -> list[str]:
    """AP books first (reference), then non-AP books by descending page count."""
    ap_books = {p["book"] for p in pages if p["is_ap"]}
    non_ap_counts: dict[str, int] = defaultdict(int)
    for p in pages:
        if not p["is_ap"]:
            non_ap_counts[p["book"]] += 1
    non_ap_order = sorted(
        non_ap_counts,
        key=lambda b: -non_ap_counts[b]
    )
    return list(ap_books) + non_ap_order


# Similarity
def page_similarity_matrix(
    ref_pages: list[dict], cand_pages: list[dict], config: DuplicateConfig
) -> np.ndarray:
    """
    (len(ref_pages), len(cand_pages)) similarity matrix; entries below
    config.threshold are 0.

    rapidfuzz.process.cdist scores the remainder with score_cutoff so
    unreachable pairs are rejected early.
    """
    sim = np.zeros((len(ref_pages), len(cand_pages)), dtype=config.dtype)

    ref_idx = np.arange(len(ref_pages))
    cand_idx = np.arange(len(cand_pages))

    sub = process.cdist(
        [ref_pages[i]["text"] for i in ref_idx],
        [cand_pages[j]["text"] for j in cand_idx],
        scorer=config.scorer,
        score_cutoff=config.threshold,
        dtype=config.dtype,
        workers=config.n_workers,
    )
    sim[np.ix_(ref_idx, cand_idx)] = sub
    return sim

# Duplicate discovery
def select_year_eligible(
    cand_dir_pages: list[dict],
    ref_year: int,
    max_year_diff: int,
    correct_year_bounds: tuple,
) -> list[dict]:
    return [
        page
        for page in cand_dir_pages
        if (
            min(ref_year, page["year_prediction"]) <= correct_year_bounds[0]
            or min(ref_year, page["year_prediction"]) >= correct_year_bounds[1]
        )
        or abs(page["year_prediction"] - ref_year) <= max_year_diff
    ]


def match_ref_group_to_candidates(
    ref_group: list[dict],
    year_eligible_cands: list[dict],
    config: DuplicateConfig,
    duplicate_page_matches: dict[PageKey, list],
) -> None:
    """Score one reference group against candidates and record matches in place."""
    sim_m = page_similarity_matrix(ref_group, year_eligible_cands, config)
    threshold = config.dtype(config.threshold)

    for cand_i, cand_page in enumerate(year_eligible_cands):
        scores = sim_m[:, cand_i]
        above = np.where(scores >= threshold)[0]
        if len(above) == 0:
            continue
        above = above[np.argsort(-scores[above])]

        page_key = (cand_page["book"], cand_page["page_n"], cand_page["side"])
        for ref_i in above:
            ref_page = ref_group[ref_i]
            duplicate_page_matches[page_key].append(
                {
                    "book": ref_page["book"],
                    "page_n": ref_page["page_n"],
                    "side": ref_page["side"],
                    "similarity_score": float(scores[ref_i]),
                }
            )


def match_book_pair(
    ref_by_dir_year: dict[tuple[str, int], list[dict]],
    cand_pages: list[dict],
    config: DuplicateConfig,
    duplicate_page_matches: dict[PageKey, list],
    valid_direction_pairs: set
) -> None:
    cand_by_dir = group_by_direction(cand_pages)

    for (ref_dir, ref_year), ref_group in ref_by_dir_year.items():
        for cand_dir, cand_dir_pages in cand_by_dir.items():
            if (ref_dir, cand_dir) not in valid_direction_pairs:
                continue
            year_eligible = select_year_eligible(
                cand_dir_pages,
                ref_year,
                config.max_year_diff,
                config.correct_year_prediction_bounds
            )
            if not year_eligible:
                continue
            match_ref_group_to_candidates(
                ref_group, year_eligible, config, duplicate_page_matches
            )


def find_duplicate_pages(
    pages: list[dict], config: DuplicateConfig
) -> dict[PageKey, list[dict]]:
    """page_key -> matches accumulated across all ordered book pairs."""
    book_order = compute_book_order(pages)
    book_to_pages = group_pages_by_book(pages)
    duplicate_page_matches: dict[PageKey, list] = defaultdict(list)
    valid_direction_pairs = build_valid_direction_pairs(config.all_directions)

    for i, ref_book in enumerate(book_order[:-1]):
        ref_pages = book_to_pages[ref_book]
        if not ref_pages:
            continue
        ref_by_dir_year = group_by_direction_and_year(ref_pages)

        for cand_book in book_order[i + 1 :]:
            cand_pages = book_to_pages[cand_book]
            if not cand_pages:
                continue
            match_book_pair(
                ref_by_dir_year, cand_pages, config, duplicate_page_matches,
                valid_direction_pairs
            )

    return duplicate_page_matches


# Handling results
def apply_duplicate_flags(
    parish_df: pd.DataFrame,
    pages: list[dict],
    duplicate_page_matches: dict[PageKey, list[dict]],
    max_entries: int,
) -> pd.DataFrame:
    page_key_to_row_indices = {
        (p["book"], p["page_n"], p["side"]): p["row_indices"] for p in pages
    }
    for page_key, matches in duplicate_page_matches.items():
        top_matches = sorted(matches, key=lambda x: -x["similarity_score"])[
            :max_entries
        ]
        row_indices = page_key_to_row_indices[page_key]
        parish_df.loc[row_indices, "duplicate"] = True
        for row_idx in row_indices:
            parish_df.at[row_idx, "duplicate_of"] = top_matches
    return parish_df


def write_parish_jsonl(
    parish_df: pd.DataFrame, output_path: str | Path, config: DuplicateConfig
) -> None:
    parish_df[config.output_columns].to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False
    )


# Orchestrating
def process_parish(
    parish_jsonl: str | Path,
    output_dir: str | Path,
    config: DuplicateConfig,
) -> None:
    parish_df = load_parish_df(parish_jsonl)
    p_id = parish_df.loc[0, "parish_id"]
    pages = build_page_records(parish_df, join_char=config.join_char)

    matches = find_duplicate_pages(pages, config)
    parish_df = apply_duplicate_flags(
        parish_df, pages, matches, config.max_duplicate_of_entries
    )
    write_parish_jsonl(
        parish_df, Path(output_dir) / f"{p_id}_migration_events.jsonl"
    )