#!/usr/bin/env python3

# input
import csv
import json

import numpy as np
import pandas as pd

record_parishes_csv = "data/csv/Moving_record_parishes_with_formats_v2.csv"
record_formats_csv = "data/csv/Moving_record_formats_v2.csv"
migration_data_dir = "data/migration-data-csv-release-v1"

# output
book_ids_json = "outputs/json/book_ids.json"

record_parishes_rows = []
with open(record_parishes_csv) as fp:
    reader = csv.reader(fp)
    for row in reader:
        record_parishes_rows.append(row)
record_parishes_columns = record_parishes_rows.pop(0)
record_parishes_rows = record_parishes_rows[:-1]

record_formats_rows = []
with open(record_formats_csv) as fp:
    reader = csv.reader(fp)
    for row in reader:
        record_formats_rows.append(row)
record_formats_columns = record_formats_rows.pop(0)


record_parishes_end_col = record_parishes_columns.index("example")
record_parishes_columns = record_parishes_columns[:record_parishes_end_col]
record_parishes_rows = [
    r[:record_parishes_end_col] for r in record_parishes_rows
]

record_formats_col_def = record_formats_columns.index("sarakkeet")
record_formats_columns = record_formats_columns[: record_formats_col_def + 1]

for i in range(len(record_formats_rows)):
    r = record_formats_rows[i]
    r_cols = [c for c in r[record_formats_col_def:] if len(c) > 0]
    r = r[:record_formats_col_def]
    r.append(r_cols)
    record_formats_rows[i] = r


record_parishes_df = pd.DataFrame(
    record_parishes_rows, columns=record_parishes_columns
)
record_formats_df = pd.DataFrame(
    record_formats_rows, columns=record_formats_columns
)

record_parishes_df = record_parishes_df.replace("", np.nan)
record_formats_df = record_formats_df.replace("", np.nan)

parish_ids = record_parishes_df["parish_id"].unique().tolist()

record_parishes_df = record_parishes_df.drop(
    ["additional", "year_added", "kartta", "notes"], axis="columns"
)
record_parishes_df["classification"] = record_parishes_df["classification"].str.lower()
record_parishes_df["year_start"] = record_parishes_df["year_start"].astype(int)
record_parishes_df["year_end"] = record_parishes_df["year_end"].astype(int)

record_parishes_df["book_id"] = (
    record_parishes_df["url"].str.split("bid=").str[1]
)
book_id_to_book = (
    record_parishes_df[
        [
            "parish_normalized",
            "year_start",
            "year_end",
            "source",
            "book_id",
            "doc",
        ]
    ]
    .set_index("book_id")
    .to_dict("index")
)

with open(book_ids_json, "w") as fp:
    json.dump(book_id_to_book, fp, ensure_ascii=False)
