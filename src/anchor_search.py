from src.similarity_matrix import sim_m_max_threshold_cols_rows


def look_for_anchors(
    sim_m,
    min_n_of_anchors,
    threshold_step: int = 1,
    threshold_start: int = 100,
    threshold_stop: int = 0,
):
    t = threshold_start + threshold_step
    previous_n_anchors = 0

    while (t := t - threshold_step) > threshold_stop:
        m, idxs = sim_m_max_threshold_cols_rows(sim_m, threshold=t / 100)

        n_anchors = m.shape[1]

        if n_anchors >= min_n_of_anchors:
            break
        else:
            previous_n_anchors = n_anchors

    return m, idxs, t, n_anchors, previous_n_anchors
