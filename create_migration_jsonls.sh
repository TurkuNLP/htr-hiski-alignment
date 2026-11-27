#!/usr/bin/bash

source venv/bin/activate

echo "Combining events with metadata"
./extracting-migrations-all_migrations_with_metadata.py \
    --books "outputs/csv/migration-data-table_fix/" \
    --year-predictions "outputs/csv/year_predictions_extended.csv" \
    --books-metadata "outputs/csv/migration_books_metadata.csv" \
    --output-dir "outputs/json/migration_jsonls/0-with-metadata"

echo "Filling columns"
./fill_column_values.py \
    --input-dir "outputs/json/migration_jsonls/0-with-metadata" \
    --output-dir "outputs/json/migration_jsonls/1-with-filled_columns"

echo "Finding duplicates"
./extracting-migrations-overlap_discovery.py \
    --input-dir "outputs/json/migration_jsonls/1-with-filled_columns"
    --output-dir "outputs/json/migration_jsonls/2-with-duplicate-info" \
    --scorer "ratio" \
    --scorer-threshold 75 \
    --max-year-difference 0 \
    --min-info-count 2 \
    --allow-invalid-years \
    --allow-unknown-directions

echo "Finding toponym columns"
./find_toponym_columns.py \
    --input-dir "outputs/json/migration_jsonls/2-with-duplicate-info" \
    --output-dir "outputs/json/migration_jsonls/3-final" \
    --valid-toponyms "data/json/placenames.jsonl"
