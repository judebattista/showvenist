"""Basic assignment (spec §11.5): files × (episodes + private unmatched columns)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from showvenist.scoring.types import Assignment

# Stand-in for −∞: linear_sum_assignment rejects infeasible matrices with real infinities.
NEG = -1e9
_INFEASIBLE = NEG / 2  # any chosen cell at or below this means the problem had no real solution


def assign_basic(S: np.ndarray, forbidden: set[tuple[int, int]] | None = None) -> Assignment:
    """Maximize Σ S over a one-to-one matching where any file may stay unmatched (score 0).

    `forbidden` holds (file, candidate) pairs scored −∞; candidate −1 forbids that file's
    unmatched column. column[i] is the candidate index or −1. If the constraints leave no
    feasible matching, total is −inf.
    """
    S = np.asarray(S, dtype=float)
    F, E = S.shape
    if F == 0:
        return Assignment(column=[], total=0.0)
    M = np.full((F, E + F), NEG)
    M[:, :E] = S
    M[np.arange(F), E + np.arange(F)] = 0.0
    for i, j in forbidden or ():
        M[i, E + i if j == -1 else j] = NEG
    rows, cols = linear_sum_assignment(M, maximize=True)
    column = [-1] * F
    for r, c in zip(rows, cols, strict=True):
        column[r] = int(c) if c < E else -1
    chosen = M[rows, cols]
    total = float("-inf") if (chosen <= _INFEASIBLE).any() else float(chosen.sum())
    return Assignment(column=column, total=total)
