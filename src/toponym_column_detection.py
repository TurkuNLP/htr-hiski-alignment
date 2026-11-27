import orjson as json
from functools import lru_cache
import fastpercentile
import numpy as np
import rapidfuzz
from itertools import zip_longest


# Helper functions


def load_events(path):
    with open(path, "rb") as fp:
        return [json.loads(line) for line in fp if line.strip()]

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


def is_place_category(category: str) -> bool:
    return (
        category == "place"
        or category.startswith("source")
        or category.startswith("destination")
    )


def categorize_toponym_columns(
    toponym_cols: list[int], col_names_categorized: list[str]
) -> list[tuple[int, str]]:
    """
    Pair found columns with the place categories of the page, left to right.
    """
    place_categories = [
        c for c in col_names_categorized if is_place_category(c)
    ]
    if place_categories:
        return list(zip(toponym_cols, place_categories))
    return [
        (
            col_idx, col_names_categorized[col_idx]
            if col_idx < len(col_names_categorized)
            else "unknown",
        )
        for col_idx in toponym_cols
    ]


@lru_cache(maxsize=None)
def should_score_cell(cell: str, isdigit_threshold=0.5, min_len=2) -> bool:
    """
    Filter out cells that are unlikely to be toponyms regardless of content.
    """
    cell_len = len(cell)
    if cell_len < min_len:
        return False
    if cell.isdigit():
        return False
    if cell.isalpha():
        return True

    if sum(c.isdigit() for c in cell) / cell_len >= isdigit_threshold:
        return False

    return True

def column_mean_value_len(column):
    return sum(map(len, column)) / len(column)

def get_score_matrix(queries, choices, scorer, score_cutoff, workers):
    return rapidfuzz.process.cdist(
        queries, choices,
        scorer=scorer, score_cutoff=score_cutoff,
        workers=workers,
        dtype=np.float32
    )


def get_matches_from_matrix(queries, sim_m, choices_arr):
    best_idx = np.ascontiguousarray(sim_m).argmax(axis=1)
    best_scores = sim_m[np.arange(sim_m.shape[0]), best_idx]
    names = np.where(best_scores > 0, choices_arr[best_idx], "")
    return dict(zip(queries, zip(names, best_scores)))


def get_matches(queries, choices, scorer, score_cutoff, workers):
    sim_m = get_score_matrix(queries, choices, scorer, score_cutoff, workers)
    return get_matches_from_matrix(queries, sim_m, choices)


def aggregate_column_score(scores: np.ndarray, non_empty: np.ndarray, percentile=75) -> float:
    """
    Aggregate per-cell scores into a single column score.

    Uses the given percentile over non-empty cells, so that a column with a few
    high-scoring toponyms and many empty cells is not unfairly penalised, while
    a column with a few lucky matches among mostly non-toponyms does not score
    artificially high.
    """
    selected = scores[non_empty]
    if selected.size == 0:
        return 0.0
    return float(fastpercentile.percentile(selected, percentile))

def compute_column_features(
    filled_columns: list[list],
    value_pos: dict,
    toponym_best: np.ndarray,
    month_best: np.ndarray,
    marital_best: np.ndarray,
):
    features = []
    for filled_col in filled_columns:
        n = len(filled_col)
        non_empty = np.fromiter((bool(c) for c in filled_col), dtype=bool, count=n)
        flags = np.fromiter(
            (should_score_cell(c) for c in filled_col), dtype=bool, count=n
        )

        idx = np.fromiter(
            # Unknown or unscored cells select 0.0
            (value_pos.get(c, -1) for c in filled_col),
            dtype=np.intp, count=n
        )
        idx = np.where(flags, idx, -1)

        non_empty_only_alphabetic = [
            "".join(filter(str.isalpha, c)) for c in filled_col
        ]

        features.append(
            {
                "toponym_score": aggregate_column_score(
                    toponym_best[idx], non_empty
                ),
                "month_score": aggregate_column_score(
                    month_best[idx], non_empty
                ),
                "marital_status_score": aggregate_column_score(
                    marital_best[idx], non_empty
                ),
                "mean_n_words": (
                    float(np.mean([len(c.split()) for c in filled_col]))
                    if filled_col
                    else 0.0
                ),
                "mean_alphabetic_value_len": column_mean_value_len(
                    non_empty_only_alphabetic
                ),
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


def build_possibly_related_toponyms(
    group_toponym_cols: list[tuple],   # [(col_idx, score), …] — all found columns
    row_data_filled: list[str],
    col_names_categorized: list[str],
) -> list[list]:
    """
    Map found toponym columns onto the ordered list of place-categories from
    col_names_categorized.

    Rules:
    - place_categories  = ordered subset of col_names_categorized values that
                          are either "place" or starts with
                          "source"/"destination"
    - If place_categories is empty, no filtering is done and
      col_names_categorized[col_idx] used for each found column as before.
    - If len(found) > len(place_categories), the top-n by score (n =
      len(place_categories)) are kept and original left-to-right order
      is restored.
    - Categories are assigned positionally: 1st kept column becomes
      place_categories[0], etc.
    """
    def value_at(col_idx: int) -> str:
        return row_data_filled[col_idx] if col_idx < len(row_data_filled) else ""

    all_entries = [
        (col_idx, value_at(col_idx), score)
        for col_idx, score in group_toponym_cols
    ]

    place_categories = [
        category for category in col_names_categorized
        if is_place_category(category)
    ]

    if not place_categories:
        return [
            [
                col_idx,
                value,
                col_names_categorized[col_idx]
                    if col_idx < len(col_names_categorized) else "unknown",
                score,
            ]
            for col_idx, value, score in all_entries
        ]

    n = len(place_categories)

    if len(all_entries) > n:
        top_n_positions = sorted(
            range(len(all_entries)),
            key=lambda i: -all_entries[i][2],
        )[:n]
        selected = sorted(top_n_positions)
        filtered = [all_entries[i] for i in selected]
    else:
        filtered = all_entries

    return [
        [col_idx, value, place_categories[i], score]
        for i, (col_idx, value, score) in enumerate(filtered)
        if i < n
    ]

