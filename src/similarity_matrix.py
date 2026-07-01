import evaluate
from matplotlib import pyplot as plt
import numpy as np
import sacrebleu
from tqdm.notebook import tqdm
import seaborn as sns

from src.pairwise_align import (
    align_score_and_coordinates_pairs,
    align_score_and_coordinates_str,
)


def create_slices(
    a_rows,
    b_rows,
    a_block_size=1,
    b_block_size=1,
    a_step=1,
    b_step=1,
    a_range_start=0,
    b_range_start=0,
    a_range_end=None,
    b_range_end=None,
):
    a_range_end = a_range_end if a_range_end else len(a_rows)
    b_range_end = b_range_end if b_range_end else len(b_rows)

    a_ranges = [
        (i, i + a_block_size)
        for i in range(a_range_start, a_range_end - a_block_size, a_step)
    ]
    b_ranges = [
        (i, i + b_block_size)
        for i in range(
            b_range_start, b_range_end - b_block_size, b_step
        )
    ]

    a_slices = [a_rows[i:j] for i, j in a_ranges]
    b_slices = [b_rows[i:j] for i, j in b_ranges]

    return a_slices, b_slices, a_ranges, b_ranges


def score_pairs(target, query, scorer):
    scores = []
    for t, q in zip(target, query):
        if t == "":
            t = " "
        if q == "":
            q = " "
        scores.append(scorer(t.lower(), q.lower()))

    scores = [score / 100 for score in scores]
    avg_score = sum(scores) / len(target)

    return avg_score, scores


def create_sim_m(a_slices, b_slices, scorer, join_char=None):
    m = np.zeros(shape=(len(a_slices), len(b_slices)))
    individual_scores = []
    for i in tqdm(range(len(a_slices))):
        for j in range(len(b_slices)):
            if join_char:
                s1 = join_char.join(a_slices[i])
                s2 = join_char.join(b_slices[j])
                if len(s1.strip()) == 0 or len(s2.strip()) == 0:
                    m[i, j] = 0
                else:
                    m[i, j] = scorer(s1, s2) / 100  # max(len(s1), len(s2))
            else:
                m[i, j], scores = score_pairs(
                    a_slices[i], b_slices[j], scorer
                )
                individual_scores.append(scores)

    return m, individual_scores


def create_sim_m_chrf(a_slices, b_slices, join_char=None):
    m = np.zeros(shape=(len(a_slices), len(b_slices)))
    for i in tqdm(range(len(a_slices))):
        for j in range(len(b_slices)):
            pred = join_char.join(a_slices[i])
            ref = join_char.join(b_slices[j])
            m[i, j] = sacrebleu.corpus_chrf(
                hypotheses=[pred], references=[[ref]], word_order=0
            ).score

    return m


def create_sim_m_and_alignments_m(
    a_slices, b_slices, aligner, join_char=None
):
    m = np.zeros(shape=(len(a_slices), len(b_slices)))
    coordinates = [
        [0 for j in range(len(b_slices))] for i in range(len(a_slices))
    ]
    individual_scores = []
    for i in tqdm(range(len(a_slices))):
        for j in range(len(b_slices)):
            if join_char:
                m[i, j], coordinates[i][j] = align_score_and_coordinates_str(
                    a_slices[i], b_slices[j], aligner, join_char
                )
            else:
                m[i, j], coordinates[i][j], scores = (
                    align_score_and_coordinates_pairs(
                        a_slices[i], b_slices[j], aligner
                    )
                )
                individual_scores.append(scores)

    return m, coordinates, individual_scores


def create_sim_m_heatmap(m, cmap=None, ax=None, figsize=(20, 10), xlabel="ocr block", ylabel="db block", zeros_together=False):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)

    ax = sns.heatmap(m, ax=ax, square=True, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if zeros_together:
        ax.invert_yaxis()

    return ax


def two_way_max_m(sim_m):
    row_indices = np.arange(sim_m.shape[0])
    col_indices = np.arange(sim_m.shape[1])
    row_argmax = sim_m.argmax(axis=0)
    col_argmax = sim_m.argmax(axis=1)

    sim_m_max = np.zeros(sim_m.shape)
    sim_m_max[row_indices, col_argmax] = sim_m[row_indices, col_argmax]
    sim_m_max[row_argmax, col_indices] = sim_m[row_argmax, col_indices]

    return sim_m_max


def sim_m_max_threshold_cols_rows(
    sim_m_max, threshold, threshold_cols=True, threshold_rows=True
):
    sim_m_max_indices = np.indices(sim_m_max.shape).transpose((1, 2, 0))

    if threshold_cols:
        col_mask = sim_m_max.max(axis=0) > threshold
    else:
        col_mask = np.ones(sim_m_max.shape[0], dtype=bool)
    if threshold_rows:
        row_mask = sim_m_max.max(axis=1) > threshold
    else:
        col_mask = np.ones(sim_m_max.shape[1], dtype=bool)

    sim_m_max_thresholded = sim_m_max[row_mask][:, col_mask]
    sim_m_max_thresholded_indices = sim_m_max_indices[row_mask][:, col_mask]

    return sim_m_max_thresholded, sim_m_max_thresholded_indices
