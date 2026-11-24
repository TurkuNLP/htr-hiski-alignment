#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm.cli import tqdm
from collections import defaultdict

from src.year_predictions import fill_years, pick_year

argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--year-predictions")
argparser.add_argument("-b", "--books")
argparser.add_argument("-ids", "--book-ids")
argparser.add_argument("-o", "--output")
argparser.add_argument("--fill", action="store_true")
argparser.add_argument("--fill-limit", type=int, default=2)
args = argparser.parse_args()

# "data/year-predictions"
YEAR_PREDICTIONS = Path(args.year_predictions)
# "outputs/json/book_ids.json"
BOOK_ID_JSON = args.book_ids
# "data/migration-data-csv-release-v1"
BOOK_CSV_DIR = Path(args.books)
# "outputs/csv/year_predictions_extended.csv"
OUTPUT = args.output
FILL_YEARS = args.fill
FILL_LIMIT = args.fill_limit

DOC_TYPES = ("muuttaneet", "muuttokirjoja", "sekalaiset")

# Load book IDs data
with open(BOOK_ID_JSON, "r") as f:
    book_id_to_book = json.load(f)

book_id_to_book = {
    int(b_id): values for b_id, values in book_id_to_book.items()
}

# Create a mapping from book name to book ID
book_to_id = {}

for book_id, metadata in tqdm(book_id_to_book.items()):
    parish = metadata["parish_normalized"]
    start_year = metadata["year_start"]
    end_year = metadata["year_end"]
    source = metadata["source"].lower()
    doctype = metadata["doc"]

    expected_filename = f"{parish}/{doctype}_{start_year}-{end_year}_{source}"

    file_path = BOOK_CSV_DIR / expected_filename
    key = parish + "_" + file_path.stem
    book_to_id[key] = book_id

# Load and process event year data
files = list(YEAR_PREDICTIONS.glob("*.jsonl"))

if len(files) == 0:
    raise ValueError("No event files found")

all_year_objects = []
for file_path in tqdm(files):
    with open(file_path, "rb") as fp:
        all_year_objects.extend(json.loads(line) for line in fp)

all_year_data = pd.DataFrame.from_dict(all_year_objects)
all_year_data = all_year_data.rename(
    {
        "year_range": "year_range",
        "page": "page",
        "side": "side",
        "original": "year_original",
        "corrected": "year_corrected",
    },
    axis="columns",
)

all_year_data = all_year_data[
    ["book", "year_range", "page", "side", "year_original", "year_corrected"]
]
all_year_data["page"] = all_year_data["page"].astype(int)
all_year_data["year"] = pick_year(
    all_year_data, "year_corrected", "year_original"
)

if FILL_YEARS:
    all_year_data["year_fill_groups"] = (
        all_year_data["book"] + all_year_data["side"]
    )
    reading_order = all_year_data.sort_values(
        ["book", "page"], kind="stable"
    ).index
    all_year_data["year_filled"] = fill_years(
        all_year_data.loc[reading_order],
        column_to_fill_from="year",
        column_to_group_by="year_fill_groups",
        column_to_read_range_from="year_range",
        limit=FILL_LIMIT,
    )
    all_year_data = all_year_data.drop("year_fill_groups", axis="columns")
    all_year_data["year"] = all_year_data.pop("year_filled")

event_year_df = all_year_data[all_year_data["year"] != 0].copy()
event_year_df["year"] = event_year_df["year"].astype("int64")
event_year_df["book_id"] = event_year_df["book"].map(
    lambda x: book_to_id.get(x, 0)
)

# Count the number of rows (events) for each page
files = tuple(BOOK_CSV_DIR.glob("*/*/**.csv"))
file_to_n_rows = {}
for file in tqdm(files):
    file_to_n_rows[file.stem] = file.read_text().count("\n")

page_to_n_rows = defaultdict(int)
for f, rows in tqdm(file_to_n_rows.items()):
    page_to_n_rows["_".join(f.split("_")[1:-1])] += rows

# Add event count and parish information to the event year DataFrame
key = (
    event_year_df["book"]
    + "_"
    + event_year_df["page"].astype(str)
    + "_"
    + event_year_df["side"]
)
event_year_df["events"] = key.map(page_to_n_rows)
event_year_df["parish_normalized"] = event_year_df["book"].map(
    lambda x: x.split(
        "_" + next((sep for sep in DOC_TYPES if sep in x), None)
    )[0]
)

event_year_df.to_csv(OUTPUT, index=False)