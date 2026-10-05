"""M3 contract: scorer protocol, shared inputs and defaults (spec §9, §11.1).

Every scorer module implements the `Scorer` protocol below, returns RAW log-likelihood ratios
(nats, unclamped, unweighted) and never applies the clamp or the weight: fusion does both. A
missing observation is exactly 0. Scorers never read each other's evidence (§9.1).

Module map (one task each; implement exactly these names):

scoring/duration.py   DurationScorer      (spec §9.2 Duration)
scoring/title.py      TitleScorer
scoring/refsubs.py    RefSubsScorer
scoring/synopsis.py   SynopsisScorer      (also owns scoring/names.py helpers: cast_names())
scoring/characters.py GuestScorer
scoring/cues.py       CuesScorer
scoring/llm.py        LlmScorer           (+ adapters/llm_cues.py for ingest-time cues)
fusion.py             fuse(), log_prior()
assign/basic.py       assign_basic()
assign/confidence.py  confidence()
report.py             build_report(), render_tables(), exit_code()   (schema in report_schema.py)
eval/harness.py       read_manifest() resolve_labels() evaluate()     (types in eval/types.py)
eval/fit.py           fit_weights()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from showvenist.analysis.types import Dialogue
from showvenist.bundle.schema import Bundle, Episode
from showvenist.config import Config

Regime = Literal["base", "llm"]

# ---- defaults (placeholders until `eval --fit`, spec §9.1) -----------------------------------

DEFAULT_CLAMPS: dict[str, tuple[float, float]] = {
    "duration": (-8.0, 2.0),
    "title": (0.0, 6.0),
    "refsubs": (-4.0, 12.0),
    "synopsis": (0.0, 4.0),
    "guests": (0.0, 6.0),
    "cues": (0.0, 4.0),
    # decision 40: the only clamp is the p clip to [0.01, 0.99]; with n candidates the evidence
    # lies in [-log(99) + log(n-1), +log(99) + log(n-1)]. The fusion clamp is a wide safety net.
    "llm": (-12.0, 12.0),
}
DEFAULT_WEIGHTS: dict[str, float] = {k: 1.0 for k in DEFAULT_CLAMPS}

# Which scorers are summed in each regime (§11.1). A file is in the "llm" regime iff the LLM
# scorer reports it `available`; the LLM then REPLACES synopsis, guests and cues.
REGIME_SCORERS: dict[Regime, tuple[str, ...]] = {
    "base": ("duration", "title", "refsubs", "synopsis", "guests", "cues"),
    "llm": ("duration", "title", "refsubs", "llm"),
}

PI_X_DEFAULT = 0.15  # expected share of extras (§11.2)
HIGH_DELTA = 4.6  # ≈ 99:1 (§11.7)
MEDIUM_DELTA = 2.2  # ≈ 9:1
HISTORY_PENALTY_DEFAULT = float(np.log(10))  # §11.4


# ---- inputs ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """An episode as numbered in the ordering being solved (§11.3). `index` is its column."""

    index: int
    episode: Episode
    season: int
    number: int

    @property
    def id(self) -> str:
        return self.episode.id

    @property
    def label(self) -> str:
        return f"S{self.season:02d}E{self.number:02d}"


@dataclass
class FileInfo:
    """Everything the scorers may read about one file (stage-1/2 results from M2)."""

    path: Path
    fingerprint: str
    duration_s: float
    title_tag: str | None = None
    disc: str | None = None
    dialogue: Dialogue | None = None  # None or track=None -> dialogue scorers report unavailable
    ocr_quality: float | None = None


@dataclass
class ScoreContext:
    bundle: Bundle
    candidates: list[Candidate]
    config: Config
    bundle_dir: Path | None = None  # reference subtitle paths are relative to it
    language: str = "en"
    vocabulary: set[str] = field(default_factory=set)  # normalized words of bundle text/subs

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


# ---- scorer protocol --------------------------------------------------------------------------


@dataclass
class ScoreResult:
    """One scorer's raw evidence for F files × E candidates."""

    llr: np.ndarray  # (F, E) float; 0 = no information
    available: np.ndarray  # (F,) bool; False = scorer had no observation for the file
    notes: dict[tuple[int, int], dict] = field(default_factory=dict)  # (file, cand) -> details
    # per-window scores for dialogue scorers, (F, W, E) or None (play-all detection, §14)
    window_llr: np.ndarray | None = None

    @classmethod
    def empty(cls, n_files: int, n_cands: int) -> ScoreResult:
        return cls(np.zeros((n_files, n_cands)), np.zeros(n_files, dtype=bool))


class Scorer(Protocol):
    name: str  # key into DEFAULT_CLAMPS / weights
    can_be_negative: bool  # signals that can't penalize must never return negative values

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult: ...


# ---- fusion and assignment ---------------------------------------------------------------------


@dataclass
class Fused:
    """S(i, j) = log_prior(j) + Σ_k w_k · clamp_k(LLR_k(i, j)); unmatched = 0 (§11.1)."""

    S: np.ndarray  # (F, E)
    contributions: dict[str, np.ndarray]  # weighted+clamped (F, E) per scorer actually summed
    regimes: list[Regime]  # per file
    log_prior: float


@dataclass
class Assignment:
    """column[i] is the candidate index assigned to file i, or -1 for 'unmatched' (extra)."""

    column: list[int]
    total: float


@dataclass
class FileConfidence:
    delta: float  # Δ_i (§11.7); inf if no alternative exists
    level: Literal["high", "medium", "low"]
    runner_up: int | None  # candidate index (or -1 for unmatched) of the best alternative
    runner_up_delta: float | None


def level_of(delta: float) -> Literal["high", "medium", "low"]:
    if delta >= HIGH_DELTA:
        return "high"
    return "medium" if delta >= MEDIUM_DELTA else "low"
