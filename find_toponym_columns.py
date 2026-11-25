#!/usr/bin/env python

import argparse
import orjson as json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rapidfuzz
from tqdm.cli import tqdm

from src.toponym_column_detection import (
    compute_column_features, fill_values, find_toponym_columns,
    should_score_cell, transpose_page_table,
)


def is_place_category(category: str) -> bool:
    return (
        category == "place"
        or category.startswith("source")
        or category.startswith("destination")
    )


def build_possibly_related_toponyms(
    group_toponym_cols: list[tuple],   # [(col_idx, score), …] — all found columns
    row_data_filled: list[str],
    row_columns_categorized: list[str],
) -> list[list]:
    """
    Map found toponym columns onto the ordered list of place-categories from
    row_columns_categorized.

    Rules:
    - place_categories  = ordered subset of row_columns_categorized values that
                          are either "place" or starts with
                          "source"/"destination"
    - If place_categories is empty, no filtering is done and
      row_columns_categorized[col_idx] used for each found column as before.
    - If len(found) > len(place_categories), the top-n by score (n =
      len(place_categories)) are kept and original left-to-right order
      is restored.
    - Categories are assigned positionally: 1st kept column becomes
      place_categories[0], etc.
    """
    def value_at(col_idx: int) -> str:
        return row_data_filled[col_idx] if col_idx < len(row_data_filled) else ""

    all_entries = [
        (col_idx, value_at(col_idx), score)
        for col_idx, score in group_toponym_cols
    ]

    place_categories = [
        category for category in row_columns_categorized
        if is_place_category(category)
    ]

    if not place_categories:
        return [
            [
                col_idx,
                value,
                row_columns_categorized[col_idx]
                    if col_idx < len(row_columns_categorized) else "unknown",
                score,
            ]
            for col_idx, value, score in all_entries
        ]

    n = len(place_categories)

    if len(all_entries) > n:
        top_n_positions = sorted(
            range(len(all_entries)),
            key=lambda i: -all_entries[i][2],
        )[:n]
        selected = sorted(top_n_positions)
        filtered = [all_entries[i] for i in selected]
    else:
        filtered = all_entries

    return [
        [col_idx, value, place_categories[i], score]
        for i, (col_idx, value, score) in enumerate(filtered)
        if i < n
    ]


argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input-dir")
argparser.add_argument("-o", "--output-dir")
argparser.add_argument("-p", "--print-type-columns")
argparser.add_argument("-pc", "--print-type-columns-categorized")
argparser.add_argument("-t", "--valid-toponyms")
args = argparser.parse_args()

# "outputs/json/all_migration_events_parish_splits/"
INPUT_DIR = Path(args.input_dir)
# "outputs/toponym_cols/parishes"
OUTPUT_DIR = args.output_dir
# "data/json/placenames.jsonl"
VALID_TOPONYMS = args.valid_toponyms

with open(VALID_TOPONYMS, "rb") as fp:
    valid_toponyms = set(json.loads(line)["spelling"] for line in fp)

with open("data/etunimi_split.txt", "r") as fp:
    valid_first_names = set(name.strip() for name in fp)
with open("data/sukunimi_split.txt", "r") as fp:
    valid_last_names = set(name.strip() for name in fp)
with open("data/patron_split.txt", "r") as fp:
    valid_patronyms = set(name.strip() for name in fp)

valid_names = valid_first_names | valid_last_names | valid_patronyms

valid_months = {
    "Tammikuu", "tammikuu", "Tammik.", "tammik.", "Tammi", "tammi",
    "Helmikuu", "helmikuu", "Helmik.", "helmik.", "Helmi", "helmi",
    "Maaliskuu", "maaliskuu", "Maalisk.", "maalisk.", "Maalis", "maalis",
    "Huhtikuu", "huhtikuu", "Huhtik.", "huhtik.", "Huhti", "huhti",
    "Toukokuu", "toukokuu", "Toukok.", "toukok.", "Touko", "touko",
    "Kesäkuu", "kesäkuu", "Kesäk.", "kesäk.", "Kesä", "kesä",
    "Heinäkuu", "heinäkuu", "Heinäk.", "heinäk.", "Heinä", "heinä",
    "Elokuu", "elokuu", "Elok.", "elok.", "Elo", "elo",
    "Syyskuu", "syyskuu", "Syysk.", "syysk.", "Syys", "syys",
    "Lokakuu", "lokakuu", "Lokak.", "lokak.", "Loka", "loka",
    "Marraskuu", "marraskuu", "Marrask.", "marrask.", "Marras", "marras",
    "Joulukuu", "joulukuu", "Jouluk.", "jouluk.", "Joulu", "joulu",
    "Januari", "januari", "Jan.", "jan.",
    "Februari", "februari", "Febr.", "febr.",
    "Mars", "mars",
    "April", "april", "Apr.", "apr.",
    "Maj", "maj",
    "Juni", "juni", "Jun.", "jun.",
    "Juli", "juli", "Jul.", "jul.",
    "Augusti", "augusti", "Aug.", "aug.",
    "September", "september", "Sept.", "sept.",
    "Oktober", "oktober", "Okt.", "okt.",
    "November", "november", "Nov.", "nov.",
    "December", "december", "Dec.", "dec.",
}
valid_marital_statuses = {
    "Nainut", "nainut", "Naimaton", "naimaton", "Leski", "leski", "Lapsi", "lapsi",
}

SCORER = rapidfuzz.fuzz.ratio
SCORE_CUTOFF = 70
MIN_SCORE = 70
GAP_THRESHOLD = 10
MAX_MEAN_N_WORDS = 3.0
MIN_MEAN_VALUE_LEN = 2.5
VALUES_TO_FILL = {
    '"',
    "''",
    '-"-',
    '-"',
    '"-',
    "1o",
    "2o",
    "2i",
    "\"Do",
    "do",
    "Do",
    "Do.",
    'Dito',
    "jo",
    "J",
}
MARITAL_STATUS_SCORE_THRESHOLD = 75
MONTH_SCORE_THRESHOLD = 75
OUTPUT_COLUMNS = [
    "row_data",
    "row_data_filled",
    "possibly_related_toponyms",
    "row_columns",
    "row_columns_categorized",
    "year_prediction",
    "duplicate",
    "duplicate_of",
    "has_information",
    "parish_id",
    "book",
    "doctype",
    "book_source",
    "book_year_start",
    "book_year_end",
    "print_type",
    "image_range_start",
    "image_range_end",
    "page_n",
    "in/out",
    "side",
    "file_row",
    "file_path"
]

combined_valid = [*valid_toponyms, *valid_months, *valid_marital_statuses]
n_top = len(valid_toponyms)
n_month = len(valid_months)

MAX_CACHED_VALUES = 2_000_000

score_index: dict[str, int] = {}
score_table = np.empty((0, 3), dtype=np.float32)


def score_values(values, combined_valid, n_top, n_month):
    global score_table

    new = [v for v in values if v not in score_index]

    if len(score_index) + len(new) > MAX_CACHED_VALUES:
        score_index.clear()
        score_table = np.empty((0, 3), dtype=np.float32)
        new = list(values)

    if new:
        sim = rapidfuzz.process.cdist(
            new,
            combined_valid,
            scorer=SCORER,
            score_cutoff=SCORE_CUTOFF,
            workers=-1,
            dtype=np.float32,
        )
        block = np.column_stack([
            sim[:, :n_top].max(axis=1),
            sim[:, n_top:n_top + n_month].max(axis=1),
            sim[:, n_top + n_month:].max(axis=1),
        ])
        start = score_table.shape[0]
        score_table = np.vstack([score_table, block])
        score_index.update((v, start + i) for i, v in enumerate(new))

    return np.fromiter(
        (score_index[v] for v in values), dtype=np.intp, count=len(values)
    )


def load_events(path):
    with open(path, "rb") as fp:
        return [json.loads(line) for line in fp if line.strip()]


output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

jsons = sorted(
    INPUT_DIR.glob("*.jsonl"),
    key=lambda x: x.stat().st_size,
    reverse=True
)

for events_json in tqdm(jsons):
    events = load_events(events_json)
    info_records = [e for e in events if e.get("has_information") == True]

    all_values = tuple({
        value
        for e in info_records
        for value in e["row_data"] if should_score_cell(value)
    })

    value_idx = score_values(all_values, combined_valid, n_top, n_month)
    rows = score_table[value_idx]

    toponym_best = np.append(rows[:, 0], np.float32(0.0))
    month_best = np.append(rows[:, 1], np.float32(0.0))
    marital_best = np.append(rows[:, 2], np.float32(0.0))

    value_pos = {v: i for i, v in enumerate(all_values)}

    for e in events:
        e["row_data_filled"] = None
        e["possibly_related_toponyms"] = None

    groups = defaultdict(list)
    for e in info_records:
        key = (e.get("book"), e.get("page_n"), e.get("side"), e.get("in/out"))
        if any(k is None for k in key):
            continue
        groups[key].append(e)

    for members in tqdm(groups.values(), leave=False):
        group_row_data = [e["row_data"] for e in members]
        group_cols = transpose_page_table(group_row_data)
        group_cols_filled = [
            fill_values(col, VALUES_TO_FILL) for col in group_cols
        ]

        column_features = compute_column_features(
            filled_columns=group_cols_filled,
            value_pos=value_pos,
            toponym_best=toponym_best,
            month_best=month_best,
            marital_best=marital_best,
        )
        group_toponym_cols = find_toponym_columns(
            column_features,
            min_score=MIN_SCORE,
            gap_threshold=GAP_THRESHOLD,
            max_mean_n_words=MAX_MEAN_N_WORDS,
            min_mean_alphabetic_value_len=MIN_MEAN_VALUE_LEN,
            marital_status_score_threshold=MARITAL_STATUS_SCORE_THRESHOLD,
            month_score_threshold=MONTH_SCORE_THRESHOLD,
        )

        row_columns_categorized = members[0]["row_columns_categorized"]
        place_categories = [
            category for category in row_columns_categorized
            if is_place_category(category)
        ]
        n_place = len(place_categories)

        if place_categories and len(group_toponym_cols) > n_place:
            top_positions = sorted(
                range(len(group_toponym_cols)),
                key=lambda i: -group_toponym_cols[i][1],
            )[:n_place]
            selected_entries = [
                group_toponym_cols[i] for i in sorted(top_positions)
            ]
        else:
            selected_entries = group_toponym_cols

        if place_categories:
            entry_categories = place_categories[:len(selected_entries)]
        else:
            entry_categories = [
                row_columns_categorized[col_idx]
                if col_idx < len(row_columns_categorized) else "unknown"
                for col_idx, _ in selected_entries
            ]

        rows_filled = [
            [col[row_i] if row_i < len(col) else "" for col in group_cols_filled]
            for row_i in range(len(members))
        ]

        for e, row_data_filled in zip(members, rows_filled):
            e["row_data_filled"] = row_data_filled
            e["possibly_related_toponyms"] = [
                [
                    col_idx,
                    row_data_filled[col_idx] if col_idx < len(row_data_filled) else "",
                    category,
                    score,
                ]
                for (col_idx, score), category in zip(
                    selected_entries, entry_categories
                )
            ]

    output_filename = output_dir / f"{events_json.stem}.jsonl"
    with open(output_filename, "wb") as fp:
        for e in tqdm(events, leave=False):
            record = {k: e.get(k) for k in OUTPUT_COLUMNS}
            fp.write(
                json.dumps(record) + b"\n"
            )
