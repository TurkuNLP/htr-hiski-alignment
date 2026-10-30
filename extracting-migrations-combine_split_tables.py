#!/usr/bin/env python

import argparse
from src.fix_split_single_table_books import (
    collect_files_to_fix,
    load_single_table_books,
    replace_with_merged
)
from tqdm.cli import tqdm

argparser = argparse.ArgumentParser()

# "data/migration-data-csv-release-v1/"
argparser.add_argument("-i", "--input")
# "outputs/csv/migration-data-table_fix/"
argparser.add_argument("-o", "--output")
# "outputs/csv/migration_books_metadata.csv"
argparser.add_argument("-m", "--books-metadata")

args = argparser.parse_args()

INPUT_DIR = args.input
OUTPUT_DIR = args.output
BOOK_METADATA_PATH = args.books_metadata

# Copy contents of `INPUT_DIR` to `OUTPUT_DIR`, then run this script to fix
# tables in `OUTPUT_DIR`.

books = load_single_table_books(BOOK_METADATA_PATH)
files_to_fix = collect_files_to_fix(books, INPUT_DIR)

for page_ranges in tqdm(files_to_fix.values()):
    for in_out_files in page_ranges.values():
        for files in in_out_files.values():
            replace_with_merged(files, INPUT_DIR, OUTPUT_DIR)
