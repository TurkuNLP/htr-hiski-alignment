#!/usr/bin/env python

import argparse
import pandas as pd
import json
import rapidfuzz
from pathlib import Path
from tqdm.cli import tqdm

from src.toponym_column_detection import (
    compute_column_features, fill_values, find_toponym_columns,
    get_matches, transpose_page_table,
)


def is_place_category(category: str) -> bool:
    return (
        category== "place"
        or category.startswith("source")
        or category.startswith("destination")
    )


def build_possibly_related_toponyms(
    group_toponym_cols: list[tuple],   # [(col_idx, score), …]  — all found columns
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
      len(place_categories)) are kept and original original left-to-right order
      is restored.
    - Categories are assigned positionally: 1st kept column becomes
      place_categories[0], etc.
    """
    def value_at(col_idx: int) -> str:
        return row_data_filled[col_idx] if col_idx < len(row_data_filled) else ""

    # All found entries in their original order
    all_entries = [
        (col_idx, value_at(col_idx), score)
        for col_idx, score in group_toponym_cols
    ]

    place_categories = [
        category for category in row_columns_categorized
        if is_place_category(category)
    ]

    if not place_categories:
        # Fall back to the col_idx lookup if no categorized columns
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
        # Select top-n by score
        top_n_positions = sorted(
            range(len(all_entries)),
            key=lambda i: -all_entries[i][2],
        )[:n]
        # restore original left-to-right order
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
# "outputs/csv/print_type_columns.csv"
PRINT_TYPE_COLUMNS = args.print_type_columns
# "outputs/json/print_type_column_names_categorized.json"
PRINT_TYPE_COLUMNS_CATEGORIZED = args.print_type_columns_categorized
# "data/json/placenames.jsonl"
VALID_TOPONYMS = args.valid_toponyms

print_columns_df = pd.read_csv(PRINT_TYPE_COLUMNS)
print_columns_df = print_columns_df.rename(
    {"print type": "print_type", "sarakkeet": "column"}, axis="columns"
)
print_columns_df["column"] = print_columns_df["column"].apply(
    lambda cols: json.loads(cols.replace("'", '"'))
)
print_columns_df = print_columns_df.explode("column").reset_index(drop=True)
print_columns_df["column_idx"] = print_columns_df.groupby("print_type").cumcount()

with open(PRINT_TYPE_COLUMNS_CATEGORIZED, "r") as fp:
    print_column_name_to_categorized = json.load(fp)

print_columns_df["column_categorized"] = print_columns_df["column"].apply(
    print_column_name_to_categorized.get
)

with open(VALID_TOPONYMS, "r") as fp:
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
SCORE_CUTOFF = 60
MIN_SCORE = 60
GAP_THRESHOLD = 10
MAX_MEAN_N_WORDS = 3.0
MIN_MEAN_VALUE_LEN = 2.5
VALUES_TO_FILL = {'"', '-"-' '-"', '"-', "1o", "2o", "2i", "do", "Do", "jo", "J"}
MARITAL_STATUS_SCORE_THRESHOLD = 75
MONTH_SCORE_THRESHOLD = 75
OUTPUT_COLUMNS = [
    "row_data",
    "row_data_filled",
    "possibly_related_toponyms",
    "row_columns",
    "row_columns_categorized",
    "found_toponym_columns",
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
]

valid_toponyms = tuple(valid_toponyms)
valid_months = tuple(valid_months)
valid_marital_statuses = tuple(valid_marital_statuses)

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

jsons = tuple(INPUT_DIR.glob("*.jsonl"))


for events_json in tqdm(jsons):
    events = pd.read_json(events_json, lines=True)

    all_values = tuple({
        value
        for row in events["row_data"].to_list()
        for value in row
    })

    toponym_scores = get_matches(
        all_values, valid_toponyms, scorer=SCORER, score_cutoff=SCORE_CUTOFF, workers=-1
    )
    month_scores = get_matches(
        all_values, valid_months, scorer=SCORER, score_cutoff=SCORE_CUTOFF, workers=-1
    )
    marital_status_scores = get_matches(
        all_values, valid_marital_statuses, scorer=SCORER, score_cutoff=SCORE_CUTOFF, workers=-1
    )

    n_rows = len(events)
    found_toponym_columns_col    = [None] * n_rows
    row_data_filled_col          = [None] * n_rows
    possibly_related_toponyms_col = [None] * n_rows

    index_to_pos = {idx: pos for pos, idx in enumerate(events.index)}

    for (book, page_n, side, direction), group_df in events.groupby(
        ["book", "page_n", "side", "in/out"]
    ):
        group_row_data = group_df["row_data"].to_list()
        group_cols = transpose_page_table(group_row_data)
        group_cols_filled = [
            fill_values(col, VALUES_TO_FILL) for col in group_cols
        ]

        page_features = compute_column_features(
            filled_columns=group_cols_filled,
            toponym_scores=toponym_scores,
            month_scores=month_scores,
            marital_status_scores=marital_status_scores,
        )
        group_toponym_cols = find_toponym_columns(
            page_features,
            min_score=MIN_SCORE,
            gap_threshold=GAP_THRESHOLD,
            max_mean_n_words=MAX_MEAN_N_WORDS,
            min_mean_alphabetic_value_len=MIN_MEAN_VALUE_LEN,
            marital_status_score_threshold=MARITAL_STATUS_SCORE_THRESHOLD,
            month_score_threshold=MONTH_SCORE_THRESHOLD,
        )

        found_toponym_columns = [list(tc) for tc in group_toponym_cols]

        row_columns_categorized = group_df.iloc[0]["row_columns_categorized"]
        place_categories = [
            category for category in row_columns_categorized
            if is_place_category(category)
        ]
        n_place = len(place_categories)

        # Precompute which found columns survive the top-n filter (order preserved)
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

        # Transpose filled columns back to row-oriented format in one go
        n_cols = len(group_cols_filled)
        n_rows_group = len(group_df)
        all_rows_filled = [
            [
                group_cols_filled[col_j][row_i] if row_i < len(group_cols_filled[col_j]) else ""
                for col_j in range(n_cols)
            ]
            for row_i in range(n_rows_group)
        ]

        if place_categories:
            entry_categories = place_categories[:len(selected_entries)]
        else:
            entry_categories = [
                row_columns_categorized[col_idx]
                if col_idx < len(row_columns_categorized) else "unknown"
                for col_idx, _ in selected_entries
            ]

        for row_i, orig_idx in enumerate(group_df.index):
            row_data_filled = all_rows_filled[row_i]
            possibly_related_toponyms = [
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

            pos = index_to_pos[orig_idx]
            found_toponym_columns_col[pos] = found_toponym_columns
            row_data_filled_col[pos] = row_data_filled
            possibly_related_toponyms_col[pos] = possibly_related_toponyms

    events["found_toponym_columns"]    = found_toponym_columns_col
    events["row_data_filled"]          = row_data_filled_col
    events["possibly_related_toponyms"] = possibly_related_toponyms_col
    events = events.drop(columns=["possible_name"])

    output_filename = output_dir / f"{events_json.stem}.jsonl"
    events[OUTPUT_COLUMNS].to_json(
        output_filename, orient="records", lines=True, force_ascii=False
    )
