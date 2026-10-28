#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import json
import rapidfuzz
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm.cli import tqdm

from src.toponym_column_detection import compute_column_features, fill_values, find_toponym_columns, get_matches, transpose_page_table


# Load data

print_columns_df = pd.read_csv("outputs/csv/print_type_columns.csv")
print_columns_df = print_columns_df.rename(
    {"print type": "print_type", "sarakkeet": "column"}, axis="columns"
)
print_columns_df["column"] = print_columns_df["column"].apply(
    lambda cols: json.loads(cols.replace("'", '"'))
)
print_columns_df = print_columns_df.explode("column").reset_index(drop=True)
print_columns_df["column_idx"] = print_columns_df.groupby(
    "print_type"
).cumcount()

with open("outputs/json/print_type_column_names_categorized.json", "r") as fp:
    print_column_name_to_categorized = json.load(fp)

print_columns_df["column_categorized"] = print_columns_df["column"].apply(
    print_column_name_to_categorized.get
)


with open("data/json/placenames.jsonl", "r") as fp:
    valid_toponyms = set(json.loads(line)["spelling"] for line in fp)

with open("data/etunimi_split.txt", "r") as fp:
    valid_first_names = set(name.strip() for name in fp)
with open("data/sukunimi_split.txt", "r") as fp:
    valid_last_names = set(name.strip() for name in fp)
with open("data/patron_split.txt", "r") as fp:
    valid_patronyms = set(name.strip() for name in fp)

valid_names = valid_first_names | valid_last_names | valid_patronyms
valid_months = {
    "Tammikuu", "tammikuu", "tammi", "januari",
    "Helmikuu", "helmikuu", "helmi", "februari",
    "Maaliskuu", "maaliskuu", "maalis", "mars",
    "Huhtikuu", "huhtikuu", "huhti", "april",
    "Toukokuu", "toukokuu", "touko", "maj",
    "Kesäkuu", "kesäkuu", "kesä", "juni",
    "Heinäkuu", "heinäkuu", "heinä", "juli",
    "Elokuu", "elokuu", "elo", "augusti",
    "Syyskuu", "syyskuu", "syys", "september",
    "Lokakuu", "lokakuu", "loka", "oktober",
    "Marraskuu", "marraskuu", "marras", "november",
    "Joulukuu", "joulukuu", "joulu", "december",
    "Tammik.", "Helmik.", "Maalisk.", "Huhtik.",
    "Toukok.", "Kesäk.", "Heinäk.", "Elok.",
    "Syysk.", "Lokak.", "Marrask.", "Jouluk.",
    "tammik.", "helmik.", "maalisk.", "huhtik.",
    "toukok.", "kesäk.", "heinäk.", "elok.",
    "syysk.", "lokak.", "marrask.", "jouluk.",
}
valid_marital_statuses = {
    "Nainut",
    "nainut",
    "Naimaton",
    "naimaton",
    "Leski",
    "leski",
    "Lapsi",
    "lapsi",
}


#scorer = rapidfuzz.distance.Levenshtein.normalized_similarity
SCORER = rapidfuzz.fuzz.ratio
SCORE_CUTOFF = 60
MIN_SCORE = 60
GAP_THRESHOLD = 10
MAX_MEAN_N_WORDS = 3.0
VALUES_TO_FILL = {'"', '-"-', "1o", "do", "Do", "jo", "J"}

jsons = tuple(Path(
    "outputs/json/all_migration_events_parish_splits/"
).glob("*.jsonl"))

valid_toponyms = tuple(valid_toponyms)
valid_months = tuple(valid_months)
valid_marital_statuses= tuple(valid_marital_statuses)

for events_json in tqdm(jsons):
    events = pd.read_json(events_json, lines=True)
    all_values = tuple(set([
        value for row in events["row_data"].to_list()
        for value in row
    ]))

    # Create similarity matrix for all pairs

    toponym_scores = get_matches(
        all_values, valid_toponyms,
        scorer=SCORER, score_cutoff=SCORE_CUTOFF,
        workers=-1
    )
    month_scores = get_matches(
        all_values, valid_months,
        scorer=SCORER, score_cutoff=SCORE_CUTOFF,
        workers=-1
    )
    marital_status_scores = get_matches(
        all_values, valid_marital_statuses,
        scorer=SCORER, score_cutoff=SCORE_CUTOFF,
        workers=-1
    )

    book_outputs = []
    for (book, page_n, side), group_df in events.groupby(["book", "page_n", "side"]):
        group_row_data = group_df["row_data"].to_list()
        group_cols = transpose_page_table(group_row_data)

        group_cols_filled = [
            fill_values(col, VALUES_TO_FILL) for col in group_cols
        ]

        page_features = compute_column_features(
                filled_columns=group_cols_filled,
                toponym_scores=toponym_scores,
                month_scores=month_scores,
                marital_status_scores=marital_status_scores
            ) 
        group_toponym_cols = find_toponym_columns(
            page_features,
            min_score=MIN_SCORE,
            gap_threshold=GAP_THRESHOLD,
            max_mean_n_words=MAX_MEAN_N_WORDS
        )

        output_data = {
            "book": book,
            "page_n": page_n,
            "side": side,
            "toponym_columns": []
        }

        for col_idx in group_toponym_cols:
            filled = group_cols_filled[col_idx]
            non_empty_values = list(filter(bool, filled))
            output_data["toponym_columns"].append({
                "column_index": col_idx,
                "values": non_empty_values
            })

        book_outputs.append(output_data)

    output_filename = f"outputs/toponym_cols/toponyms_{book}.jsonl"
    with open(output_filename, 'w') as f:
        f.writelines([
            json.dumps(page_side_output, ensure_ascii=False) + "\n"
            for page_side_output in book_outputs
        ])
