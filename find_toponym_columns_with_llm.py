#!/usr/bin/env python

import argparse
from collections import defaultdict
from pathlib import Path

import orjson as json
from dotenv import load_dotenv
from tqdm import tqdm

from src.toponym_column_detection import (
    load_events,
    categorize_toponym_columns,
    transpose_page_table
)
from src.toponym_column_detection_llm import (
    ToponymColumnDetectionConfig,
    create_client,
    find_toponym_columns_batch,
)

MAX_MEAN_N_WORDS = 3.0
MIN_MEAN_VALUE_LEN = 2.5
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
    "file_path",
]


def load_events(path):
    with open(path, "rb") as fp:
        return [json.loads(line) for line in fp if line.strip()]


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "-i", "--input-dir", required=True, type=Path,
    help="e.g. outputs/json/all_migration_events_parish_splits/",
)
argparser.add_argument(
    "-o", "--output-dir", required=True, type=Path,
    help="e.g. outputs/toponym_cols/parishes",
)
args = argparser.parse_args()

load_dotenv()
config = ToponymColumnDetectionConfig(
    model="Qwen/Qwen3.8-27B",
    max_mean_n_words=MAX_MEAN_N_WORDS,
    min_mean_alphabetic_value_len=MIN_MEAN_VALUE_LEN,
)
client = create_client(config)

args.output_dir.mkdir(parents=True, exist_ok=True)
jsonl_files = sorted(
    args.input_dir.glob("*.jsonl"),
    key=lambda x: x.stat().st_size,
    reverse=True
)

for events_path in tqdm(jsonl_files):
    events = load_events(events_path)

    groups = defaultdict(list)
    for e in events:
        if e.get("has_information") is not True:
            continue
        key = (e.get("book"), e.get("page_n"), e.get("side"), e.get("in/out"))
        if None in key:
            continue
        groups[key].append(e)

    pages_filled = [
        transpose_page_table([e["row_data_filled"] for e in members])
        for members in groups.values()
    ]
    toponym_cols_per_page = find_toponym_columns_batch(
        pages_filled, config, client
    )

    for members, toponym_cols in zip(groups.values(), toponym_cols_per_page):
        toponyms = categorize_toponym_columns(
            toponym_cols, members[0]["col_names_categorized"]
        )
        for e in members:
            e["possibly_related_toponyms"] = [
                [col_idx, e["row_data_filled"][col_idx], category]
                for col_idx, category in toponyms
            ]

    with open(args.output_dir / f"{events_path.stem}.jsonl", "wb") as fp:
        for e in events:
            fp.write(json.dumps({k: e.get(k) for k in OUTPUT_COLUMNS}) + b"\n")