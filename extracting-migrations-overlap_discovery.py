#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz
from tqdm.cli import tqdm

from src.duplicate_page_discovery import DuplicateConfig, process_parish

def parse_args():
    parser = argparse.ArgumentParser(description="Process parish migration events for duplicates")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/json/all_migration_events_parish_splits"),
        help="Input directory containing JSON files to process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/json/all_migration_events_parish_splits_duplicate_info/"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--scorer",
        type=str,
        default="ratio",
        help="Fuzzy matching scorer to use (default: ratio)"
    )
    parser.add_argument(
        "--scorer-threshold",
        type=int,
        default=75,
        help="Threshold for fuzzy matching (0-100, default: 75)"
    )
    parser.add_argument(
        "--max-year-difference",
        type=int,
        default=0,
        help="Maximum allowed year difference for matching (default: 1)"
    )
    parser.add_argument(
        "--min-info-count",
        type=int,
        default=0,
        help="Minimum number of events with has_information==True required on a page for matching (default: 0)"
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=1600,
        help="Minimum valid year (default: 1600)"
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=1980,
        help="Maximum valid year (default: 1980)"
    )
    parser.add_argument(
        "--join-char",
        type=str,
        default="|",
        help="Character used for joining strings (default: \"|\")"
    )
    parser.add_argument(
        "--allow-invalid-years",
        action="store_true"
    )
    parser.add_argument(
        "--allow-unknown-directions",
        action="store_true"
    )

    return parser.parse_args()

args = parse_args()

SCORER = getattr(fuzz, args.scorer)
INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
SCORER_THRESHOLD = args.scorer_threshold
MAX_YEAR_DIFFERENCE = args.max_year_difference
MIN_INFO_COUNT = args.min_info_count
VALID_YEAR_BOUNDS = (args.min_year, args.max_year)
JOIN_CHAR = args.join_char
ALLOW_INVALID_YEARS=args.allow_invalid_years
ALLOW_UNKNOWN_DIRECTIONS=args.allow_unknown_directions

# Load data
book_migration_directions = pd.read_csv(
    "outputs/csv/book_migration_direction_info.csv"
)
with open("outputs/json/book_ids.json", "r") as fp:
    book_id_to_info = json.load(fp)

# Duplicate page discovery
jsonls = tuple(INPUT_DIR.glob("*.jsonl"))
config = DuplicateConfig(
    scorer=SCORER,
    threshold=SCORER_THRESHOLD,
    max_year_diff=MAX_YEAR_DIFFERENCE,
    valid_year_prediction_bounds=VALID_YEAR_BOUNDS,
    allow_invalid_years=ALLOW_INVALID_YEARS,
    allow_unknown_directions=ALLOW_UNKNOWN_DIRECTIONS,
    min_info_count=MIN_INFO_COUNT,
    join_char=JOIN_CHAR,
)

for parish_jsonl in tqdm(jsonls):
    process_parish(
        parish_jsonl,
        OUTPUT_DIR,
        config
    )