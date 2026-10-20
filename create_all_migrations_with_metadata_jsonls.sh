#!/bin/bash

INPUT_DIR="data/migration-data-csv-release-v1"
OUTPUT_DIR="outputs/csv/migration-data-table_fix"

rm -r $OUTPUT_DIR
cp -r $INPUT_DIR $OUTPUT_DIR

./extracting-migrations-combine_split_tables.py

./extracting-migrations-all_migrations_with_metadata.py