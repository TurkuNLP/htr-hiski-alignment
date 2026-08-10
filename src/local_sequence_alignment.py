"""
Local sequence alignment (Smith-Waterman) with affine gap penalties (Gotoh),
driven by an *arbitrary* M x N similarity matrix.

The algorithm does not care where the per-column scores come from: you supply a
dense M x N array ``S`` where ``S[i, j]`` is the similarity of element ``i`` of
sequence A to element ``j`` of sequence B. There is no alphabet and no
substitution matrix. This makes it usable for embeddings, feature vectors,
time-series points, custom kernels, etc.

Scoring model
-------------
Three matrices are maintained (Gotoh affine gaps):

    E[i,j] = max( H[i,j-1] - (gap_open + gap_extend),   E[i,j-1] - gap_extend )   # gap in A
    F[i,j] = max( H[i-1,j] - (gap_open + gap_extend),   F[i-1,j] - gap_extend )   # gap in B
    H[i,j] = max( 0,  H[i-1,j-1] + S[i-1,j-1],  E[i,j],  F[i,j] )

``gap_open`` and ``gap_extend`` are POSITIVE penalties. A gap of length L costs
``gap_open + L * gap_extend``. For a plain linear gap penalty ``g`` per residue,
set ``gap_open=0.0`` and ``gap_extend=g``.

IMPORTANT (scale): locality only emerges when typical/background pairs score
negative. If your raw similarities are all positive, subtract a threshold tau
from S first (S = S - tau), and keep gap penalties on the same numeric scale as
S, or the alignment degenerates to (near) global.

Public API
----------
    smith_waterman(S, gap_open=1.0, gap_extend=1.0) -> Alignment
    format_alignment(A, B, alignment)               -> str   (optional pretty-print)

The heavy loops are JIT-compiled with Numba. The first call pays a one-time
compilation cost (~1 s); subsequent calls run at compiled-code speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

__all__ = ["Alignment", "smith_waterman", "format_alignment"]

# Sentinel for "no gap state possible here" (matrix boundary). Large negative,
# but finite, so Numba arithmetic stays well-defined (no inf/NaN propagation).
_NEG = -1.0e18


@dataclass
class Alignment:
    """Result of a local alignment.

    Attributes
    ----------
    score : float
        Score of the optimal local alignment (0.0 if the empty alignment is best).
    end : (int, int)
        0-based (a_index, b_index) of the last aligned residue pair, i.e. the
        cell where the maximal score was attained. (-1, -1) if empty.
    start : (int, int)
        0-based (a_index, b_index) of the first aligned residue. (-1, -1) if empty.
    path_a, path_b : np.ndarray[int64]
        Per-column indices, ordered start -> end. ``path_a[k]`` indexes sequence A
        (a row of S) or is -1 if that column is a gap in A. Likewise ``path_b``.
    aligned_pairs : list[(Optional[int], Optional[int])]
        Same information as path_a/path_b but as Python tuples with None for gaps.
    H : Optional[np.ndarray]
        The (M+1)x(N+1) score matrix, only if ``return_matrices=True``.
    """

    score: float
    end: Tuple[int, int]
    start: Tuple[int, int]
    path_a: np.ndarray
    path_b: np.ndarray
    aligned_pairs: List[Tuple[Optional[int], Optional[int]]]
    H: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Core DP (Numba)                                                             #
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _fill(S, gap_open, gap_extend):
    """Fill H/E/F and traceback pointer matrices. Returns everything needed.

    Traceback codes:
        trace_H: 0 = stop (local start), 1 = diagonal (match/mismatch),
                 2 = came from E, 3 = came from F
        trace_E: 0 = gap opened from H[i,j-1], 1 = gap extended from E[i,j-1]
        trace_F: 0 = gap opened from H[i-1,j], 1 = gap extended from F[i-1,j]
    """
    m, n = S.shape
    H = np.zeros((m + 1, n + 1), dtype=np.float64)
    E = np.full((m + 1, n + 1), _NEG, dtype=np.float64)
    F = np.full((m + 1, n + 1), _NEG, dtype=np.float64)

    trace_H = np.zeros((m + 1, n + 1), dtype=np.int8)
    trace_E = np.zeros((m + 1, n + 1), dtype=np.int8)
    trace_F = np.zeros((m + 1, n + 1), dtype=np.int8)

    open_ext = gap_open + gap_extend

    best = 0.0
    bi = 0
    bj = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # E: gap in A (consume b_{j-1}); horizontal move
            e_open = H[i, j - 1] - open_ext
            e_ext = E[i, j - 1] - gap_extend
            if e_open >= e_ext:
                E[i, j] = e_open
                trace_E[i, j] = 0
            else:
                E[i, j] = e_ext
                trace_E[i, j] = 1

            # F: gap in B (consume a_{i-1}); vertical move
            f_open = H[i - 1, j] - open_ext
            f_ext = F[i - 1, j] - gap_extend
            if f_open >= f_ext:
                F[i, j] = f_open
                trace_F[i, j] = 0
            else:
                F[i, j] = f_ext
                trace_F[i, j] = 1

            # H: best of stop / diagonal / E / F
            diag = H[i - 1, j - 1] + S[i - 1, j - 1]
            e_val = E[i, j]
            f_val = F[i, j]

            h = 0.0
            code = np.int8(0)  # stop
            if diag > h:
                h = diag
                code = np.int8(1)
            if e_val > h:
                h = e_val
                code = np.int8(2)
            if f_val > h:
                h = f_val
                code = np.int8(3)

            H[i, j] = h
            trace_H[i, j] = code

            if h > best:
                best = h
                bi = i
                bj = j

    return H, E, F, trace_H, trace_E, trace_F, best, bi, bj


@njit(cache=True)
def _traceback(trace_H, trace_E, trace_F, bi, bj, m, n):
    """Walk pointers back from the max cell. Returns (path_a, path_b) ordered
    start -> end, with -1 marking a gap. Capacity m+n is the max possible length.
    """
    cap = m + n
    pa = np.empty(cap, dtype=np.int64)
    pb = np.empty(cap, dtype=np.int64)
    k = 0  # number of columns collected (in reverse order for now)

    i = bi
    j = bj
    state = 0  # 0 = in H, 1 = in E, 2 = in F

    while True:
        if state == 0:  # H
            c = trace_H[i, j]
            if c == 0:
                break  # local start reached
            elif c == 1:  # diagonal: aligned pair (i-1, j-1)
                pa[k] = i - 1
                pb[k] = j - 1
                k += 1
                i -= 1
                j -= 1
            elif c == 2:  # switch into E at (i, j); no emit yet
                state = 1
            else:  # c == 3: switch into F at (i, j); no emit yet
                state = 2
        elif state == 1:  # E: column is (gap, j-1)
            pa[k] = -1
            pb[k] = j - 1
            k += 1
            if trace_E[i, j] == 0:  # opened from H[i, j-1]
                state = 0
            # else extended from E[i, j-1]: stay in E
            j -= 1
        else:  # state == 2, F: column is (i-1, gap)
            pa[k] = i - 1
            pb[k] = -1
            k += 1
            if trace_F[i, j] == 0:  # opened from H[i-1, j]
                state = 0
            # else extended from F[i-1, j]: stay in F
            i -= 1

    # reverse the first k entries into start -> end order
    ra = np.empty(k, dtype=np.int64)
    rb = np.empty(k, dtype=np.int64)
    for t in range(k):
        ra[t] = pa[k - 1 - t]
        rb[t] = pb[k - 1 - t]
    return ra, rb


# --------------------------------------------------------------------------- #
# Public wrapper (plain Python: validation + packaging)                       #
# --------------------------------------------------------------------------- #
def smith_waterman(
    S,
    gap_open: float = 1.0,
    gap_extend: float = 1.0,
    *,
    return_matrices: bool = False,
) -> Alignment:
    """Optimal local alignment of two sequences given an M x N similarity matrix.

    Parameters
    ----------
    S : array-like, shape (M, N)
        Similarity scores. ``S[i, j]`` scores element i of A vs element j of B.
        Any real values; need not be symmetric or metric. Center it (subtract a
        threshold) so background pairs are negative -- see module docstring.
    gap_open : float, default 1.0
        Positive gap-opening penalty (extra cost for the first residue of a gap).
        Set 0.0 for a linear (non-affine) gap penalty.
    gap_extend : float, default 1.0
        Positive per-residue gap penalty. A length-L gap costs
        ``gap_open + L * gap_extend``.
    return_matrices : bool, keyword-only
        If True, attach the full H matrix to the result.

    Returns
    -------
    Alignment
    """
    S = np.ascontiguousarray(S, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D (M x N); got shape {S.shape}")
    if not np.all(np.isfinite(S)):
        raise ValueError("S contains non-finite values (nan/inf).")
    if gap_open < 0 or gap_extend < 0:
        raise ValueError("gap_open and gap_extend must be non-negative penalties.")

    m, n = S.shape

    H, E, F, tH, tE, tF, best, bi, bj = _fill(
        S, float(gap_open), float(gap_extend)
    )

    if best <= 0.0 or bi == 0 or bj == 0:
        # Empty alignment is optimal.
        empty = np.empty(0, dtype=np.int64)
        return Alignment(
            score=0.0,
            end=(-1, -1),
            start=(-1, -1),
            path_a=empty,
            path_b=empty.copy(),
            aligned_pairs=[],
            H=H if return_matrices else None,
        )

    path_a, path_b = _traceback(tH, tE, tF, bi, bj, m, n)

    aligned_pairs = [
        (None if a < 0 else int(a), None if b < 0 else int(b))
        for a, b in zip(path_a, path_b)
    ]

    # start = first aligned residue indices (skip leading gaps, though a local
    # alignment never starts on a gap, this is just defensive)
    sa = next((int(a) for a in path_a if a >= 0), -1)
    sb = next((int(b) for b in path_b if b >= 0), -1)

    return Alignment(
        score=float(best),
        end=(bi - 1, bj - 1),
        start=(sa, sb),
        path_a=path_a,
        path_b=path_b,
        aligned_pairs=aligned_pairs,
        H=H if return_matrices else None,
    )


# --------------------------------------------------------------------------- #
# Optional pretty-printer (when you do have actual sequences)                 #
# --------------------------------------------------------------------------- #
def format_alignment(
    A: Sequence,
    B: Sequence,
    alignment: Alignment,
    gap: str = "-",
) -> str:
    """Render an alignment as three lines (A / markers / B).

    ``A`` and ``B`` are the actual sequences (e.g. strings or lists of symbols)
    behind the rows and columns of S. The marker line shows '|' where the two
    aligned symbols are equal, '.' where they differ, and ' ' for gaps. For
    non-string symbols a space-joined layout is used.
    """
    if alignment.score <= 0.0:
        return "(empty alignment)"

    top, mid, bot = [], [], []
    stringy = isinstance(A, str) and isinstance(B, str)
    pad = "" if stringy else " "

    def cell(x):
        return str(x)

    for a, b in alignment.aligned_pairs:
        sa = gap if a is None else cell(A[a])
        sb = gap if b is None else cell(B[b])
        w = max(len(sa), len(sb), 1)
        top.append(sa.rjust(w))
        bot.append(sb.rjust(w))
        if a is None or b is None:
            marker = " "
        elif A[a] == B[b]:
            marker = "|"
        else:
            marker = "."
        mid.append(marker.rjust(w))

    return (
        f"score = {alignment.score:.4g}   "
        f"A[{alignment.start[0]}..{alignment.end[0]}]  "
        f"B[{alignment.start[1]}..{alignment.end[1]}]\n"
        + pad.join(top) + "\n"
        + pad.join(mid) + "\n"
        + pad.join(bot)
    )


# --------------------------------------------------------------------------- #
# Demo / self-test                                                            #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ---- 1) Classic DNA sanity check, built from sequences via a score fn ----
    A = "GGTTGACTA"
    B = "TGTTACGG"

    match, mismatch = 3.0, -3.0
    Sdna = np.array(
        [[match if a == b else mismatch for b in B] for a in A],
        dtype=np.float64,
    )
    aln = smith_waterman(Sdna, gap_open=0.0, gap_extend=2.0)  # linear gap = 2
    print("DNA example (match +3 / mismatch -3 / linear gap 2):")
    print(format_alignment(A, B, aln))
    print()

    # ---- 2) Arbitrary similarity matrix, no alphabet at all -----------------
    rng = np.random.default_rng(0)
    M, N = 8, 10
    Sraw = rng.normal(size=(M, N))           # arbitrary similarities
    # plant a diagonal "signal" so there's something to find
    for k in range(5):
        Sraw[2 + k, 3 + k] += 3.0
    aln2 = smith_waterman(Sraw, gap_open=1.0, gap_extend=0.5, return_matrices=True)
    print("Arbitrary similarity matrix:")
    print("  score        =", round(aln2.score, 4))
    print("  end (a,b)    =", aln2.end)
    print("  start (a,b)  =", aln2.start)
    print("  aligned pairs=", aln2.aligned_pairs)
    print("  H shape      =", aln2.H.shape)
