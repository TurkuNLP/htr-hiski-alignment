#!/bin/bash

INPUT_DIR="data/migration-data-csv-release-v1"
OUTPUT_DIR="outputs/csv/migration-data-table_fix"

rm -r $OUTPUT_DIR
cp -r $INPUT_DIR $OUTPUT_DIR

echo "Getting book ids"
./miscellanious-get_book_ids.py

echo "Processing books metadata and print types"
./miscellanious-book_metadata.py \
    --print-types-csv "data/csv/print_types.csv" \
    --books-csv "data/csv/parish_books_with_metadata.csv" \
    --output-books-csv "outputs/csv/migration_books_metadata.csv" \
    --output-print-types-csv "outputs/csv/print_types_metadata.csv"

echo "Extending year prediction data"
./extend_year_prediction_data.py \
    --year-predictions "data/year-predictions" \
    --books "data/migration-data-csv-release-v1" \
    --book-ids "outputs/json/book_ids.json" \
    --output "outputs/csv/year_predictions_extended.csv"

echo "Combining mistakenly split tables"
./extracting-migrations-combine_split_tables.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR \
    --books-metadata "outputs/csv/migration_books_metadata.csv"
