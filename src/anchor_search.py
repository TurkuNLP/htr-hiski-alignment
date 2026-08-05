import numpy as np
from src.similarity_matrix import sim_m_max_threshold_cols_rows


def threshold_drop(
    sim_m,
    min_n_of_anchors: int = 0,
    threshold_step: int = 1,
    threshold_start: int = 100,
    threshold_stop: int = 0,
):
    t = threshold_start + threshold_step
    previous_n_anchors = 0

    while (t := t - threshold_step) >= threshold_stop:
        m, idxs = sim_m_max_threshold_cols_rows(sim_m, threshold=t / 100)

        n_anchors = m.shape[1]

        if n_anchors >= min_n_of_anchors:
            break
        else:
            previous_n_anchors = n_anchors

    return m, idxs, t, n_anchors, previous_n_anchors


def threshold_raise(
    sim_m,
    max_diff_between_x_y_length,
    threshold_step: int = 1,
    threshold_start: int = 0,
    threshold_stop: int = 100,
):
    t = threshold_start + threshold_step
    previous_diff_between_x_y_length = abs(sim_m.shape[0] - sim_m.shape[1])

    while (t := t + threshold_step) < threshold_stop:
        m, idxs = sim_m_max_threshold_cols_rows(sim_m, threshold=t / 100)

        diff_between_x_y_length = abs(m.shape[0] - m.shape[1])

        if diff_between_x_y_length <= max_diff_between_x_y_length:
            break
        else:
            previous_diff_between_x_y_length = diff_between_x_y_length

    return m, idxs, t, diff_between_x_y_length, previous_diff_between_x_y_length


def sliding_argmax(sim_m, window_size, step_size):
    m, n = sim_m.shape
    window_argmaxes = []

    for i in range(0, m - window_size, step_size):
        for j in range(0, n - window_size, step_size):
            window = sim_m[i : i + window_size, j : j + window_size]

            local_idx = np.unravel_index(
                window.argmax(), (window_size, window_size)
            )
            global_coords = (i + local_idx[0], j + local_idx[1])

            window_argmaxes.append(global_coords)

    window_argmaxes = np.array(window_argmaxes)

    anchors = np.zeros_like(sim_m)
    anchors[window_argmaxes[:, 0], window_argmaxes[:, 1]] = sim_m[
        window_argmaxes[:, 0], window_argmaxes[:, 1]
    ]

    return anchors
