#!/usr/bin/env python

import argparse
from collections import defaultdict
from pathlib import Path

import orjson as json
from tqdm import tqdm

from src.toponym_column_detection import fill_values, transpose_page_table

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
    help="directory for the jsonl files with row_data_filled added",
)
args = argparser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)

for events_path in tqdm(sorted(args.input_dir.glob("*.jsonl"))):
    events = load_events(events_path)

    groups = defaultdict(list)
    for e in events:
        e["row_data_filled"] = None
        if e.get("has_information") is not True:
            continue
        key = (e.get("book"), e.get("page_n"), e.get("side"), e.get("in/out"))
        if None in key:
            continue
        groups[key].append(e)

    for members in groups.values():
        columns = transpose_page_table([e["row_data"] for e in members])
        columns_filled = [fill_values(col, VALUES_TO_FILL) for col in columns]
        for e, row_data_filled in zip(members, transpose_page_table(columns_filled)):
            e["row_data_filled"] = row_data_filled

    with open(args.output_dir / events_path.name, "wb") as fp:
        for e in events:
            fp.write(json.dumps(e) + b"\n")