def alignments_to_tuples(coordinates):
    target = [
        (coordinates[0][i * 2], coordinates[0][i * 2 + 1])
        for i in range(len(coordinates[0]) // 2)
    ]
    query = [
        (coordinates[1][i * 2], coordinates[1][i * 2 + 1])
        for i in range(len(coordinates[0]) // 2)
    ]

    return (target, query)


def align_score_and_coordinates_str(target, query, aligner, join_char):
    t = " " + join_char.join(target).lower()
    q = " " + join_char.join(query).lower()

    alignments = aligner(t, q)

    longer_str_len = max(len(t), len(q))

    score = alignments[0].score / longer_str_len
    coordinates = alignments_to_tuples(alignments[0].coordinates)

    return score, coordinates

def align_score_and_coordinates_pairs(target, query, aligner):
    alignments = []
    for t, q in zip(target, query):
        if t == "":
            t = " "
        if q == "":
            q = " "
        alignments.append(aligner(t.lower(), q.lower()))

    scores = [
        a.score / max(len(t), len(q))
        for a, t, q in zip(alignments, target, query)
    ]
    avg_score = sum(scores) / len(target)

    coordinates = []
    for a in alignments:
        if len(a) > 0:
            coordinates.append(alignments_to_tuples(a[0].coordinates))
        else:
            coordinates.append([])

    return avg_score, coordinates, scores
