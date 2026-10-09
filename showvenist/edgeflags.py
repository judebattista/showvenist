"""Advisory edge-case flags (spec §14): `play_all` and `possible_two_parter`.

Report-only: nothing here changes the assignment or the confidence.
"""

from __future__ import annotations

import numpy as np

from showvenist.scoring.duration import double_length_starts, known_median
from showvenist.scoring.types import Candidate, Fused, ScoreResult

PLAY_ALL_RATIO = 1.8  # file duration / median episode runtime at or above which a file is 'long'
# A window points strongly at an episode when its best per-window LLR (refsubs, nats) beats
# that window's runner-up by more than this (about 10:1). `play_all` needs two such windows
# whose best episodes differ.
PLAY_ALL_WINDOW_MARGIN = 2.3
DIALOGUE_SCORERS = ("refsubs", "synopsis", "guests", "cues", "llm")


def play_all(
    i: int, duration_s: float, candidates: list[Candidate], results: dict[str, ScoreResult]
) -> bool:
    """A long file (>= PLAY_ALL_RATIO x the median runtime) whose dialogue windows strongly point
    to different episodes. Uses the per-window refsubs scores; windows without dialogue (all
    zero) are ignored and fewer than two windows with dialogue means no flag."""
    median = known_median([c.episode.runtime_min for c in candidates])
    ref = results.get("refsubs")
    if median is None or ref is None or ref.window_llr is None or len(candidates) < 2:
        return False
    if duration_s / 60.0 < PLAY_ALL_RATIO * median:
        return False
    win = np.asarray(ref.window_llr[i], dtype=float)  # (W, E)
    win = win[np.any(win != 0.0, axis=1)]
    if len(win) < 2:
        return False
    top2 = np.sort(win, axis=1)[:, -2:]
    strong = (top2[:, 1] - top2[:, 0]) > PLAY_ALL_WINDOW_MARGIN
    best = np.argmax(win, axis=1)
    return len({int(b) for b in best[strong]}) >= 2


def possible_two_parter(
    i: int, col: int, duration_s: float, candidates: list[Candidate], fused: Fused
) -> bool:
    """The file's length is about two consecutive runtimes (the duration scorer's double-length
    rule), the assigned episode is the first of that pair, and a dialogue-based scorer gives
    positive evidence for both episodes of the pair."""
    if col < 0 or col + 1 >= len(candidates):
        return False
    if col not in double_length_starts(duration_s, [c.episode.runtime_min for c in candidates]):
        return False
    for c in (col, col + 1):
        if not any(
            fused.contributions[n][i, c] > 0 for n in DIALOGUE_SCORERS if n in fused.contributions
        ):
            return False
    return True
