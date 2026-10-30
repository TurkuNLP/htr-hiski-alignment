#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
from tqdm.cli import tqdm
from itertools import zip_longest

from src.toponym_column_detection import fill_values, transpose_page_table

argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--input-dir", default=None)
argparser.add_argument("-o", "--output-dir", default=None)
args = argparser.parse_args()

# "outputs/json/all_migration_events_parish_splits/"
INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALUES_TO_FILL = {
    '"', '-"-' '-"', '"-', "1o", "2o", "2i", "do", "Do", "jo", "J"
}


def transpose_page_table(page_data):
    return [
        list(col) for col in zip_longest(*page_data, fillvalue="")
    ]

def fill_values(column, values_to_fill={'"', "1o", "do", "Do", "jo", "J"}):
    filled_cells = []
    last_value = None
    for cell in column:
        cell = cell.strip()
        if cell == "":
            filled_cells.append(cell)
        elif cell in values_to_fill and last_value is not None:
            index = len(filled_cells) - 1
            while index >= 0 and filled_cells[index] == "":
                filled_cells[index] = last_value
                index -= 1
            filled_cells.append(last_value)
        else:
            filled_cells.append(cell)
            if cell not in values_to_fill:
                last_value = cell
    return filled_cells


jsons = tuple(INPUT_DIR.glob("*.jsonl"))
for events_json in tqdm(jsons):
    events = pd.read_json(events_json, lines=True)

    row_data_filled = [None] * len(events)

    for (book, page_n, side, direction), group_df in events.groupby(
        ["book", "page_n", "side", "in/out"]
    ):
        group_row_data = group_df["row_data"].to_list()
        group_cols = transpose_page_table(group_row_data)
        group_cols_filled = [fill_values(col, VALUES_TO_FILL) for col in group_cols]
        row_data_filled_cols = transpose_page_table(group_cols_filled)

        for idx, row in zip(group_df.index, row_data_filled_cols):
            row_data_filled[idx] = row

    events["row_data_filled"] = pd.Series(row_data_filled, index=events.index)

    output_filename = OUTPUT_DIR / f"{events_json.stem}.jsonl"
    events.to_json(
        output_filename, orient="records", lines=True, force_ascii=False
    )