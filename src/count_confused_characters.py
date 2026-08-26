from collections import Counter

import numpy as np


def update_confusion_counter(
    counter, place_spelling_alignments
):
    for place_name, spelling in place_spelling_alignments:
        for i in range(len(place_name)):
            counter[place_name[i]][spelling[i]] += 1


def align_characters(place, place_spellings, aligner):
    place_as_list = list(place)
    place_alignments = []
    spelling_alignments = []
    for p_s in place_spellings:
        alignment = aligner.align(place_as_list, p_s)[0]
        place_alignments.append(alignment[0])
        spelling_alignments.append(alignment[1])

    return place_alignments, spelling_alignments

def pad_alignments(place_alignments, spelling_alignments):
    place_alignments_padded, spelling_alignments_padded = [], []

    max_len = max([max(len(x[0]), len(x[1])) for x in zip(place_alignments, spelling_alignments)])
    for place_alignment, spelling_alignment in zip(place_alignments, spelling_alignments):
        place_alignments_padded.append(
            place_alignment + (max_len - len(place_alignment)) * [None]
        )
        spelling_alignments_padded.append(
            spelling_alignment + (max_len - len(spelling_alignment)) * [None]
        )

    return place_alignments, spelling_alignments

def counter_to_matrix(chars, confusion_counter, normalize=True):
    matrix = np.zeros((len(chars), len(chars)))

    for c_1 in chars:
        for c_2 in chars:
            value = confusion_counter[c_1][c_2]
            if normalize:
                value = value / max(1, confusion_counter[c_1].total())
            matrix[chars.index(c_1), chars.index(c_2)] = value

    return matrix
