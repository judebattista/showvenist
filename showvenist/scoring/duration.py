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
# TMDB runtimes are often the broadcast SLOT (Hulk: 60 min for files of 48.6 min), so the
# scorer also tries a batch-calibrated scale k = median(file durations in the plausible band) /
# median(runtime). Found by the first library eval (M3): without it, every Hulk S02 episode was
# rejected as an extra at high confidence. k is estimated from the files alone, never from labels.
BAND = (0.55, 1.2)  # file/median-runtime ratios that count as 'episode-like' for calibrating k
MIN_BAND_FILES = 3  # fewer episode-like files than this: no calibration, wider sigma instead
K_MIN_DEVIATION = 0.03  # |k - 1| below this: runtimes are accurate, keep the sharp sigma
SIGMA_REL_SLOT = 0.15  # wide component (fraction of runtime) when k cannot be estimated
W_DOUBLE = 0.05  # prior weight of "this file holds this episode and the next one"
DOUBLE_SIGMA_MIN = 6.0  # looser: two runtimes add up, and their errors with them
W_ACCURATE = 0.5  # prior weight of "the runtime is accurate" in that two-component mixture


def _log_norm(x: float, mu: float, sigma: float) -> float:
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma * math.sqrt(2 * math.pi))


def calibrate_scale(durations_min: list[float], median_runtime: float) -> tuple[float | None, bool]:
    """(k, enough): k is None when runtimes look accurate or there are too few episode-like files.
    `enough` says whether at least MIN_BAND_FILES episode-like files exist."""
    lo, hi = BAND[0] * median_runtime, BAND[1] * median_runtime
    band = [d for d in durations_min if lo <= d <= hi]
    if len(band) < MIN_BAND_FILES:
        return None, False
    k = float(statistics.median(band)) / median_runtime
    return (k if abs(k - 1) >= K_MIN_DEVIATION else None), True


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
        k, enough = calibrate_scale([f.duration_s / 60.0 for f in files], median)
        scales = SPEED_FACTORS if k is None else (*SPEED_FACTORS, k, k * SPEED_FACTORS[1])
        log_w = {
            True: (math.log(1 - W_DOUBLE), None, math.log(W_DOUBLE)),
            False: (
                math.log((1 - W_DOUBLE) * W_ACCURATE),
                math.log((1 - W_DOUBLE) * (1 - W_ACCURATE)),
                math.log(W_DOUBLE),
            ),
        }[enough]
        for i, f in enumerate(files):
            d = f.duration_s / 60.0
            log_bg = _log_norm(d, median, BG_SIGMA_MIN)
            for j, r in enumerate(rts):
                r2 = r + (rts[j + 1] if j + 1 < len(rts) else r)  # this episode + the next
                parts = [log_w[0] + max(_log_norm(d, sc * r, sigma) for sc in scales)]
                if log_w[1] is not None:  # too few files to calibrate: the slot component
                    wide = max(sigma, SIGMA_REL_SLOT * r)
                    parts.append(log_w[1] + max(_log_norm(d, sc * r, wide) for sc in scales))
                # two-part episode muxed as one file (spec §14): not an extra, left to dialogue
                dbl = max(_log_norm(d, sc * r2, DOUBLE_SIGMA_MIN) for sc in scales)
                parts.append(log_w[2] + dbl)
                v = float(np.logaddexp.reduce(parts)) - log_bg
                ok = math.isfinite(v)
                res.llr[i, j] = float(np.clip(v, -LLR_LIMIT, LLR_LIMIT)) if ok else 0.0
            res.available[i] = True
        return res
