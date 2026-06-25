from copy import deepcopy


def remove_overlaps_handle_page_breaks(
    connections: list[list],
    a_range_start_idx: int,
    a_range_end_idx: int,
    b_range_end_idx: int,
    a_rows_idx: int,
    b_rows_idx: int,
    page_idx: int,
) -> list[list]:
    if not connections:
        return []

    overlaps_removed = []

    overlaps_removed.append(deepcopy(connections[0]))

    for latest in connections[1:]:
        current = overlaps_removed[-1]

        if (
            latest[a_range_start_idx] <= current[a_range_end_idx]
            and latest[page_idx] == current[page_idx]
        ):

            current[a_range_end_idx] = max(
                current[a_range_end_idx], latest[a_range_end_idx]
            )
            current[b_range_end_idx] = max(
                current[b_range_end_idx], latest[b_range_end_idx]
            )

            existing_pairs = set(zip(current[a_rows_idx], current[b_rows_idx]))

            for a_val, b_val in zip(latest[a_rows_idx], latest[b_rows_idx]):
                if (a_val, b_val) not in existing_pairs:
                    current[a_rows_idx].append(a_val)
                    current[b_rows_idx].append(b_val)
                    existing_pairs.add((a_val, b_val))
        else:
            overlaps_removed.append(deepcopy(latest))

    for i in range(len(overlaps_removed)):
        overlaps_removed[i] = [
            overlaps_removed[i][a_rows_idx],
            overlaps_removed[i][b_rows_idx],
            overlaps_removed[i][page_idx],
        ]

    return overlaps_removed

def score_pairs_with_pages(connections, a_names, b_names, scorer):
    pair_scores_and_page = []
    for c in connections:
        for i, j in zip(c[0], c[1]):
            pair_scores_and_page.append(
                [
                    (a_names[i], b_names[j]),
                    scorer(a_names[i], b_names[j]),
                    c[2],
                ]
            )
    return pair_scores_and_page
