"""Fusion of evidence from multiple scorers (spec §11.1, §11.2, §11.4).

S(i, j) = log_prior + Σ_k w_k · clamp_k(LLR_k(i, j)) − history(i, j); S(i, unmatched) = 0.
"""

from __future__ import annotations

import math

import numpy as np

from showvenist.config import WeightsConfig
from showvenist.scoring.types import (
    DEFAULT_CLAMPS,
    PI_X_DEFAULT,
    REGIME_SCORERS,
    Fused,
    Regime,
    ScoreResult,
)

# Scorers that can't produce negative evidence (§9.2 "for only"). Their lower clamp is forced to
# 0 whatever `clamps` says, so they never lower S.
FOR_ONLY = frozenset({"title", "titlecard", "synopsis", "guests", "cues"})


def log_prior(n_candidates: int, pi_x: float = PI_X_DEFAULT) -> float:
    """log((1 − π_x)/E) − log π_x: the evidence an episode must beat to win over 'extra' (§11.2)."""
    if n_candidates <= 0:
        raise ValueError("log_prior needs at least one candidate")
    if not 0.0 < pi_x < 1.0:
        raise ValueError("pi_x must be in (0, 1)")
    return math.log((1.0 - pi_x) / n_candidates) - math.log(pi_x)


def _weight(weights: WeightsConfig, regime: Regime, name: str) -> float:
    src = weights.llm if regime == "llm" else weights
    return float(getattr(src, name, 1.0))


def _range(name: str, clamps: dict[str, tuple[float, float]]) -> tuple[float, float]:
    lo, hi = clamps.get(name, DEFAULT_CLAMPS.get(name, (-np.inf, np.inf)))
    if name in FOR_ONLY:
        lo = max(lo, 0.0)
    return float(lo), float(hi)


def fuse(
    results: dict[str, ScoreResult],
    weights: WeightsConfig,
    clamps: dict[str, tuple[float, float]] = DEFAULT_CLAMPS,
    pi_x: float = PI_X_DEFAULT,
    history_penalty: np.ndarray | None = None,
) -> Fused:
    """Fuse raw scorer LLRs into S (F×E).

    A file is in the "llm" regime iff results["llm"] marks it available; then the LLM replaces
    synopsis, guests and cues and the weights come from `weights.llm`. Scorers absent from
    `results`, or unavailable for a file, contribute 0. `contributions[name]` is the signed
    weighted, clamped term actually added, so S = log_prior + Σ contributions. A history
    penalty (non-negative magnitudes) is subtracted and recorded as contributions["history"]
    (its negative).
    """
    shapes = {r.llr.shape for r in results.values()}
    if len(shapes) > 1:
        raise ValueError(f"scorer results disagree on shape: {shapes}")
    if shapes:
        n_files, n_cands = next(iter(shapes))
    elif history_penalty is not None:
        n_files, n_cands = history_penalty.shape
    else:
        raise ValueError("fuse needs at least one scorer result or a history penalty")

    llm = results.get("llm")
    llm_avail = (
        np.asarray(llm.available, dtype=bool) if llm is not None else np.zeros(n_files, bool)
    )
    regimes: list[Regime] = ["llm" if a else "base" for a in llm_avail]

    prior = log_prior(n_cands, pi_x) if n_cands > 0 else 0.0
    S = np.full((n_files, n_cands), prior, dtype=float)
    contributions: dict[str, np.ndarray] = {}

    names = {n for r in REGIME_SCORERS.values() for n in r}
    for name in sorted(names):
        res = results.get(name)
        if res is None:
            continue
        lo, hi = _range(name, clamps)
        clipped = np.clip(np.nan_to_num(np.asarray(res.llr, dtype=float)), lo, hi)
        avail = np.asarray(res.available, dtype=bool)
        contrib = np.zeros((n_files, n_cands))
        for i, regime in enumerate(regimes):
            if name not in REGIME_SCORERS[regime] or not avail[i]:
                continue
            contrib[i] = _weight(weights, regime, name) * clipped[i]
        if name in FOR_ONLY:
            # guard against a negative weight too: for-only evidence never lowers S
            contrib = np.maximum(contrib, 0.0)
        contributions[name] = contrib
        S += contrib

    if history_penalty is not None:
        hp = np.asarray(history_penalty, dtype=float)
        if hp.shape != (n_files, n_cands):
            raise ValueError(f"history_penalty shape {hp.shape} != {(n_files, n_cands)}")
        contributions["history"] = -hp
        S -= hp

    return Fused(S=S, contributions=contributions, regimes=regimes, log_prior=prior)
