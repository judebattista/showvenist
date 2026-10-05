"""Confidence by exact re-solve (spec §11.7): Δ_i = best − best with (i, σ(i)) forbidden."""

from __future__ import annotations

import math

import numpy as np

from showvenist.assign.basic import assign_basic
from showvenist.scoring.types import Assignment, FileConfidence, level_of


def confidence(S: np.ndarray, assignment: Assignment) -> list[FileConfidence]:
    """One FileConfidence per file. Forbidding an unmatched σ(i) forces file i onto an episode;
    the runner-up is what file i takes in the re-solve (candidate index or −1)."""
    out: list[FileConfidence] = []
    for i, col in enumerate(assignment.column):
        alt = assign_basic(S, {(i, col)})
        if not math.isfinite(alt.total) or not math.isfinite(assignment.total):
            delta, runner, rdelta = math.inf, None, None
        else:
            delta = max(0.0, assignment.total - alt.total)  # ≥ 0 up to float noise
            runner, rdelta = alt.column[i], delta
        out.append(FileConfidence(delta, level_of(delta), runner, rdelta))
    return out

