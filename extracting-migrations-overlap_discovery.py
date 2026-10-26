#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz
from tqdm.cli import tqdm

from src.duplicate_page_discovery import DuplicateConfig, process_parish

INPUT_DIR = Path(
    "outputs/json/all_migration_events_parish_splits"
)
OUTPUT_DIR = Path(
    "outputs/json/all_migration_events_parish_splits_duplicate_info/"
)
SCORER = fuzz.ratio
SCORER_THRESHOLD = 75
MAX_YEAR_DIFFERENCE = 1
VALID_YEAR_BOUNDS = (1600, 1980)
JOIN_CHAR="|"

# Load data
book_migration_directions = pd.read_csv(
    "outputs/csv/book_migration_direction_info.csv"
)
with open("outputs/json/book_ids.json", "r") as fp:
    book_id_to_info = json.load(fp)

# Run duplicate page discovery
jsonls = tuple(INPUT_DIR.glob("*.jsonl"))
config = DuplicateConfig(
    scorer=SCORER,
    threshold=SCORER_THRESHOLD,
    max_year_diff=MAX_YEAR_DIFFERENCE,
    valid_year_prediction_bounds=VALID_YEAR_BOUNDS,
    allow_invalid_years=True,
    allow_unknown_directions=True,
    min_info_count=5,
    join_char=JOIN_CHAR,
)

for parish_jsonl in tqdm(jsonls):
    process_parish(
        parish_jsonl,
        OUTPUT_DIR,
        config
    )