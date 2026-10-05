"""Duration scorer (spec §9.2): Gaussian on file duration vs. episode runtime against a broad
background density, best of 1.0x and PAL 0.959x speed. Returns raw LLR in nats."""

from __future__ import annotations

import math
import statistics

import numpy as np

from showvenist.scoring.types import FileInfo, ScoreContext, ScoreResult

SPEED_FACTORS = (1.0, 0.959)  # film/NTSC-rate, PAL speed-up (a 44 min episode runs ~42.2 min)
SIGMA_VARIED_MIN = 2.0  # runtimes in the season differ: believe them
SIGMA_UNIFORM_MIN = 4.0  # all runtimes equal: likely a placeholder, be looser
# Background density of file durations (minutes): one wide Gaussian centred on the season median.
# At a runtime match this yields about +2.5 nats (sigma 2) / +1.8 nats (sigma 4).
BG_SIGMA_MIN = 25.0
LLR_LIMIT = 50.0  # internal clip so nothing is NaN/inf


def _log_norm(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2 * math.pi))


class DurationScorer:
    name = "duration"
    can_be_negative = True

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        res = ScoreResult.empty(len(files), ctx.n_candidates)
        runtimes = [c.episode.runtime_min for c in ctx.candidates]
        known = [float(r) for r in runtimes if r]
        if not known:
            return res
        median = float(statistics.median(known))
        sigma = SIGMA_VARIED_MIN if len(set(known)) > 1 else SIGMA_UNIFORM_MIN
        rts = [float(r) if r else median for r in runtimes]
        for i, f in enumerate(files):
            d = f.duration_s / 60.0
            log_bg = _log_norm(d, median, BG_SIGMA_MIN)
            for j, r in enumerate(rts):
                best = max(_log_norm(d, s * r, sigma) for s in SPEED_FACTORS)
                v = best - log_bg
                ok = math.isfinite(v)
                res.llr[i, j] = float(np.clip(v, -LLR_LIMIT, LLR_LIMIT)) if ok else 0.0
            res.available[i] = True
        return res
