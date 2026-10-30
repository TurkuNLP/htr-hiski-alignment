#!/bin/bash

INPUT_DIR="data/migration-data-csv-release-v1"
OUTPUT_DIR="outputs/csv/migration-data-table_fix"

rm -r $OUTPUT_DIR
cp -r $INPUT_DIR $OUTPUT_DIR

echo "Extending year prediction data"
./extend_year_prediction_data.py \
    --year-predictions "data/year_predicitons" \
    --books "data/migration-data-csv-release-v1" \
    --book_ids "outputs/json/book_ids.json" \
    --output "outputs/csv/year_predictions_extended.csv"

echo "Combining mistakenly split tables"
./extracting-migrations-combine_split_tables.py \
    --input $INPUT_DIR \
    --output $OUTPUT_DIR \
    --books-metadata "outputs/csv/migration_books_metadata.csv"
