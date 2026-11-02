#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm.cli import tqdm
from collections import defaultdict

argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--year-predictions")
argparser.add_argument("-b", "--books")
argparser.add_argument("-ids", "--book-ids")
argparser.add_argument("-o", "--output")
args = argparser.parse_args()

# "data/year-predictions"
YEAR_PREDICTIONS = Path(args.year_predictions)
# "outputs/json/book_ids.json"
BOOK_ID_JSON = args.book_ids
# "data/migration-data-csv-release-v1"
BOOK_CSV_DIR = Path(args.books)
# "outputs/csv/year_predictions_extended.csv"
OUTPUT = args.output

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

events = []
for file_path in tqdm(files):
    file_df = pd.read_json(file_path, lines=True)
    events.append(file_df)
event_year_df = pd.concat(events)
event_year_df = event_year_df[
    ~event_year_df["corrected"].isna()
    & event_year_df["corrected"].astype(str).str.isnumeric()
]
event_year_df["year"] = event_year_df["corrected"].astype("int64")
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
