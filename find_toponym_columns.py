#!/usr/bin/env python

import argparse
import orjson as json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rapidfuzz
from tqdm.cli import tqdm

from src.toponym_column_detection import (
    build_possibly_related_toponyms, compute_column_features,
    find_toponym_columns, load_events, should_score_cell, transpose_page_table,
)


argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input-dir")
argparser.add_argument("-o", "--output-dir")
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
MARITAL_STATUS_SCORE_THRESHOLD = 75
MONTH_SCORE_THRESHOLD = 75
OUTPUT_COLUMNS = [
    "row_data",
    "row_data_filled",
    "possibly_related_toponyms",
    "col_names",
    "col_names_categorized",
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


output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

jsons = sorted(
    INPUT_DIR.glob("*.jsonl"),
    key=lambda x: x.stat().st_size,
    reverse=True
)

for events_json in tqdm(jsons):
    events = load_events(events_json)
    info_records = [e for e in events if e["has_information"] is True]

    all_values = tuple({
        value for e in info_records
        for value in e["row_data"] if should_score_cell(value)
    })

    value_idx = score_values(all_values, combined_valid, n_top, n_month)
    rows = score_table[value_idx]

    toponym_best = np.append(rows[:, 0], np.float32(0.0))
    month_best = np.append(rows[:, 1], np.float32(0.0))
    marital_best = np.append(rows[:, 2], np.float32(0.0))

    value_pos = {v: i for i, v in enumerate(all_values)}

    for e in events:
        e["possibly_related_toponyms"] = None

    groups = defaultdict(list)
    for e in info_records:
        key = (e.get("book"), e.get("page_n"), e.get("side"), e.get("in/out"))
        if any(k is None for k in key):
            continue
        groups[key].append(e)

    for members in tqdm(groups.values(), leave=False):
        group_cols_filled = transpose_page_table(
            [e["row_data_filled"] for e in members]
        )

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

        col_names_categorized = members[0]["col_names_categorized"]
        for e in members:
            e["possibly_related_toponyms"] = build_possibly_related_toponyms(
                group_toponym_cols, e["row_data_filled"], col_names_categorized
            )

    output_filename = output_dir / f"{events_json.stem}.jsonl"
    with open(output_filename, "wb") as fp:
        for e in tqdm(events, leave=False):
            record = {k: e.get(k) for k in OUTPUT_COLUMNS}
            fp.write(
                json.dumps(record) + b"\n"
            )