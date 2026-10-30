#!/usr/bin/env python
import argparse
import pandas as pd
from pathlib import Path
from natsort import natsorted
from src.csv_pages_to_names import read_files
from collections import defaultdict
from tqdm.cli import tqdm
from src.extract_migrations_with_metadata.extract_from_row_columns import(
    build_row,
    gather_file_paths,
    parish_output_path,
    parse_book_name
)
import json

argparser = argparse.ArgumentParser()
argparser.add_argument("-b", "--books")
argparser.add_argument("-y", "--year-predictions")
argparser.add_argument("-m", "--books-metadata")
argparser.add_argument("-o", "--output")
args = argparser.parse_args()

# "outputs/csv/migration-data-table_fix/"
CSV_DATA_DIRECTORY = args.books
# "outputs/csv/year_predictions_extended.csv"
YEAR_PREDICTIONS = args.year_predictions
# "outputs/csv/migration_books_metadata.csv"
BOOKS_METADATA = args.books_metadata
# "outputs/json/all_migration_events_parish_splits"
OUTPUT_DIR = Path(args.output)


# Metadata

## General

event_year_df = pd.read_csv(YEAR_PREDICTIONS)
book_migration_metadata = pd.read_csv(BOOKS_METADATA)

book_migration_metadata["book"] = (
    book_migration_metadata["parish_normalized"].astype(str)
    + "_"
    + book_migration_metadata["doctype"].astype(str)
    + "_"
    + book_migration_metadata["year_start"].astype(str)
    + "-"
    + book_migration_metadata["year_end"].astype(str)
    + "_"
    + book_migration_metadata["source"].astype(str)
)

### Print type columns

print_type_columns = pd.read_csv("outputs/csv/print_type_columns.csv")
with open("outputs/json/print_type_column_names_categorized.json", "r") as fp:
    categorize_column_name = json.load(fp)


print_type_columns.loc[print_type_columns["side"].isna(), "side"] = "both"
print_type_and_side_to_cols = {
    print_type.split(" ")[1]: dict(zip(group["side"], group["sarakkeet"]))
    for print_type, group in print_type_columns.groupby("print type")
}

print_type_and_side_to_cols = {
    print_type: {
        side: [] if cols == "[]" else cols[2:-2].split("', '")
        for side, cols in sides.items()
    }
    for print_type, sides in print_type_and_side_to_cols.items()
}

print_type_and_side_to_categorized_cols = {
    print_type: {
        side: tuple(categorize_column_name[column_name] for column_name in cols)
        for side, cols in sides.items()
    }
    for print_type, sides in print_type_and_side_to_cols.items()
}

## Gather migrations

BOOK_TYPE_SEPARATORS = ("muuttaneet", "sekalaiset")
GATHER_COLUMNS = [
    "source",
    "book",
    "side",
    "in_out",
    "img_range_start",
    "img_range_end",
]

all_file_paths = gather_file_paths(
    csv_data_directory=CSV_DATA_DIRECTORY,
    book_migration_metadata=BOOKS_METADATA,
    gather_columns=GATHER_COLUMNS,
    book_type_separators=BOOK_TYPE_SEPARATORS
)

book_to_print_type = (
    book_migration_metadata[
        [
            "book", "in_out", "print"
        ]
    ].set_index(["book", "in_out"])["print"].to_dict()
)

book_page_side_to_year = event_year_df[~(event_year_df["book_id"] == 0)]
book_page_side_to_year["key"] = (
    book_page_side_to_year["book"].astype(str)
    + "/" + book_page_side_to_year["page"].astype(str)
    + "/" + book_page_side_to_year["side"]
)
book_page_side_to_year = book_page_side_to_year.set_index("key")
book_page_side_to_year = book_page_side_to_year["year"].to_dict()

parish_to_parish_id = (
    book_migration_metadata[["parish_id", "parish_normalized"]]
    .drop_duplicates()
    .set_index("parish_normalized")["parish_id"]
    .to_dict()
)

def get_columns(print_type, side):
    cols = print_type_and_side_to_cols.get(print_type, {}).get(side, [])
    cols_categorized = print_type_and_side_to_categorized_cols.get(
        print_type, {}
    ).get(side, [])
    return cols, cols_categorized

def iter_book_rows(book, image_ranges):
    parish, book_source, year_start, year_end = parse_book_name(
        book, BOOK_TYPE_SEPARATORS
    )
    parish_id = parish_to_parish_id[parish]

    for image_range, directions in image_ranges.items():
        range_start, range_end = image_range.split("-")
        for in_out, file_paths in directions.items():
            pages, _ = read_files(file_paths)
            print_type = book_to_print_type[(book, in_out)]

            for file_path, page in zip(file_paths, pages):
                split_path = file_path.split("_")
                page_n = int(split_path[-3])
                side = split_path[-2]
                year = book_page_side_to_year.get(f"{book}/{page_n}/{side}", 0)
                cols, cols_categorized = get_columns(print_type, side)
                metadata = {
                    "year_prediction": int(year),
                    "file_path": file_path.replace(CSV_DATA_DIRECTORY, ""),
                    "parish_id": parish_id,
                    "book": book,
                    "print_type": print_type,
                    "book_source": book_source,
                    "book_year_start": year_start,
                    "book_year_end": year_end,
                    "image_range_start": range_start,
                    "image_range_end": range_end,
                    "page_n": page_n,
                    "in/out": in_out,
                    "side": side,
                }
                for row_idx, row in enumerate(page):
                    yield parish_id, build_row(
                        row_idx, row, cols, cols_categorized, metadata
                    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for parish_id in parish_to_parish_id.values():
    parish_output_path(parish_id).write_text("")

for book, image_ranges in tqdm(list(all_file_paths.items())):
    lines_by_parish = {}
    for parish_id, row in iter_book_rows(book, image_ranges):
        lines_by_parish.setdefault(parish_id, []).append(
            json.dumps(row, ensure_ascii=False) + "\n"
        )
    for parish_id, lines in lines_by_parish.items():
        with open(parish_output_path(parish_id), "a") as fp:
            fp.writelines(lines)
