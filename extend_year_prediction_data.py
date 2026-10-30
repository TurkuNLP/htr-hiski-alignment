#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
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

# Load book IDs data
book_ids_df = pd.read_json(BOOK_ID_JSON, orient="index")
book_ids_df.columns = ["parish_normalized", "year_start", "year_end", "source"]
book_ids_df = book_ids_df.rename_axis("book_id")

# Create a mapping from book name to book ID
book_to_id = {
    book: book_id for book_id, book in (
        book_ids_df["parish_normalized"]
        + "_" + book_ids_df.index.map(
            lambda x: BOOK_CSV_DIR.glob(
                f"*{book_ids_df.loc[x, 'parish_normalized']}/"
                f"*{book_ids_df.loc[x, 'year_start']}-"
                f"{book_ids_df.loc[x, 'year_end']}_"
                f"{book_ids_df.loc[x, 'source'].lower()}"
            )
        ).apply(lambda x: x[0].stem if len(x) > 0 else None)
    )
    .to_dict()
    .items()
}

# Load and process event year data
files = list(YEAR_PREDICTIONS.glob("*.jsonl"))

if len(files) == 0:
    raise ValueError("No event files found")

event_year_df = pd.concat(
    tuple(pd.read_json(file_path, lines=True) for file_path in files)
)
event_year_df = event_year_df[
    ~event_year_df["corrected"].isna()
    & event_year_df["corrected"].astype(str).str.isnumeric()
]
event_year_df["year"] = event_year_df["corrected"].astype("int64")
event_year_df["book_id"] = event_year_df["book"].map(
    lambda x: book_to_id.get(x, 0)
).astype("int64")

# Count the number of rows (events) for each page
files = tuple(BOOK_CSV_DIR.glob("*/*/**.csv"))
file_to_n_rows = {}
for file in tqdm(files):
    file_to_n_rows[file.stem] = file.read_text().count("\n")

page_to_n_rows = defaultdict(int)
for f, rows in tqdm(file_to_n_rows.items()):
    page_to_n_rows["_".join(f.split("_")[1:-1])] += rows

# Add event count and parish information to the event year DataFrame
key = event_year_df["book"] + "_" + event_year_df["page"].astype(str) + "_" + event_year_df["side"]
event_year_df["events"] = key.map(page_to_n_rows)
event_year_df["parish_normalized"] = event_year_df["book_id"].map(book_ids_df["parish_normalized"])

event_year_df.to_csv(OUTPUT, index=False)