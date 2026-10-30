import numpy as np
import rapidfuzz
from itertools import zip_longest

# Helper functions

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

def should_score_cell(cell: str, isdigit_threshold=0.5) -> bool:
    """
    Filter out cells that are unlikely to be toponyms regardless of content.
    """
    if not cell:
        return False
    if cell.isdigit():
        return False
    if sum(c.isdigit() for c in cell) / len(cell) >= isdigit_threshold:
        return False
    return True

def column_mean_value_len(column):
    return sum(map(len, column)) / len(column)


def get_matches(
    queries, choices, scorer, score_cutoff, workers
) -> dict[str, tuple[str, int]]:
    sim_m = rapidfuzz.process.cdist(
        queries, choices,
        scorer=scorer, score_cutoff=score_cutoff,
        workers=workers
    )
    matches = {}
    for i, value in enumerate(queries):
        row = sim_m[i]
        row_argmax = np.argmax(row)
        toponym = choices[row_argmax]
        score = row[row_argmax]
        matches[value] = (
            toponym if score > 0 else "", score
        )

    return matches


# Column-level aggregation

def aggregate_column_score(raw_scores, filled_col, percentile=75):
    """
    Aggregate per-cell scores into a single column score.

    Uses the given percentile over non-empty cells, so that a column with a few
    high-scoring toponyms and many empty cells is not unfairly penalised, while
    a column with a few lucky matches among mostly non-toponyms does not score
    artificially high.
    """
    non_empty_scores = [s for s, c in zip(raw_scores, filled_col) if c]
    if not non_empty_scores:
        return 0.0
    return float(np.percentile(non_empty_scores, percentile))

def compute_column_features(
    filled_columns: list[list],
    toponym_scores: dict,
    month_scores: dict,
    marital_status_scores: dict,
):
    features = []
    for filled_col in filled_columns:
        non_empty_only_alphabetic = [
            "".join(filter(str.isalpha, c)) for c in filled_col
        ]
        should_score_flags = [should_score_cell(cell) for cell in filled_col]
        col_toponym_scores = [
            toponym_scores.get(cell, (None, 0.0))[1] if flag else 0.0
            for cell, flag in zip(filled_col, should_score_flags)
        ]
        col_month_scores = [
            month_scores.get(cell, (None, 0.0))[1] if flag else 0.0
            for cell, flag in zip(filled_col, should_score_flags)
        ]
        col_marital_status_scores = [
            marital_status_scores.get(cell, (None, 0.0))[1] if flag else 0.0
            for cell, flag in zip(filled_col, should_score_flags)
        ]

        features.append(
            {
                "toponym_score": aggregate_column_score(
                    col_toponym_scores, filled_col
                ),
                "month_score": aggregate_column_score(
                    col_month_scores, filled_col
                ),
                "marital_status_score": aggregate_column_score(
                    col_marital_status_scores, filled_col
                ),
                "mean_n_words": (
                    float(np.mean([len(c.split()) for c in filled_col]))
                    if filled_col else 0.0
                ),
                "mean_alphabetic_value_len": column_mean_value_len(
                    non_empty_only_alphabetic
                )
            }
        )

    return features

def find_toponym_columns(
    column_features,
    min_score=0.5,
    gap_threshold=0.1,
    max_mean_n_words=2.0,
    min_mean_alphabetic_value_len=3,
    marital_status_score_threshold=0.75,
    month_score_threshold=0.75
):
    scores = np.array([f["toponym_score"] for f in column_features])
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]

    selected = []
    for i, (idx, score) in enumerate(zip(sorted_indices, sorted_scores)):
        if score < min_score:
            break
        if i > 0 and (sorted_scores[i - 1] - score) > gap_threshold:
            break

        # Filter away column that is most likely for names
        if column_features[idx]["mean_n_words"] > max_mean_n_words:
            continue

        # Filter away columns with only short strings
        if (
            column_features[idx]["mean_alphabetic_value_len"]
            < min_mean_alphabetic_value_len
        ):
            continue

        # Filter away marital status column
        marital_status_score = column_features[idx]["marital_status_score"]
        if (
            marital_status_score > score
            or marital_status_score > marital_status_score_threshold
        ):
            continue
        # Filter away month column
        month_score = column_features[idx]["month_score"]
        if (
            month_score > score
            or month_score > month_score_threshold
        ):
            continue
        selected.append((int(idx), float(score)))

    return sorted(selected, key=lambda x: x[0])
