def get_possible_name(row_columns, cols_categorized) -> str | None:
    possible_name = None
    if len(cols_categorized) == 0:
        return possible_name
    if cols_categorized != []:
        if "name" in cols_categorized and (
            name_index := cols_categorized.index("name")
        ) < len(row_columns):
            possible_name = row_columns[name_index]

    return possible_name


def get_possibly_related_toponyms(
    row_columns, cols_categorized
) -> list[tuple[int, str, str]] | list:
    if len(cols_categorized) == 0:
        return []

    row_len = len(row_columns)

    toponym_indexes = [
        (i, category) for i, category in enumerate(cols_categorized)
        if category in ("place", "source", "destination")
    ]

    if len(toponym_indexes) == 0:
        return []

    possibly_related_toponyms = [
        (i, row_columns[i], category) for i, category in toponym_indexes
        if i < row_len
    ]

    return possibly_related_toponyms


def get_has_information(row_columns, threshold) -> bool:
    joined_cols = "".join(row_columns)
    if len(tuple(c for c in joined_cols if c.isalpha())) > threshold:
        return True
    else:
        return False
