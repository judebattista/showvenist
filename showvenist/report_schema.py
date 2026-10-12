"""Report JSON v1 (spec §12.3). The schema is stable and versioned: scripts and eval read it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REPORT_VERSION = 1


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EpisodeRef(_M):
    episode_id: str
    season: int
    episode: int
    title: str


class Decision(_M):
    kind: Literal["episode", "extra"]
    episode: EpisodeRef | None = None  # None for extras


class Confidence(_M):
    level: Literal["high", "medium", "low"]
    delta: float | None  # None = no alternative (infinite)
    odds: str  # e.g. "120:1"


class SignalContribution(_M):
    name: str
    available: bool
    winner: float  # weighted, clamped contribution for the chosen option (0 for extras)
    runner_up: float | None = None


class CandidateDetail(_M):
    """One candidate episode in the evidence pane (§12.4)."""

    episode_id: str
    label: str  # S01E03 in the ordering used
    title: str
    score: float  # fused score S(i, j)
    delta: float  # best score in the pane minus this one (0 for the best)
    odds: str  # of the best against this candidate, e.g. "14:1"
    signals: list[SignalContribution] = []  # this candidate's contribution (runner_up unused)
    notes: list[str] = []  # matched lines, guests, terms, title-card readings, LLM quotes
    synopsis: str | None = None
    runtime_min: int | None = None


class FileDetail(_M):
    path: str
    duration_s: float
    regime: Literal["base", "llm"]
    candidates: list[CandidateDetail] = []
    # title-order tie-breaker line (§11.5, decision 63) for a file on a disc with title indexes
    title_order: str | None = None

    @property
    def signals(self) -> list[list[SignalContribution]]:
        return [c.signals for c in self.candidates]

    @property
    def notes(self) -> list[list[str]]:
        return [c.notes for c in self.candidates]

    @property
    def runtimes(self) -> list[int | None]:
        return [c.runtime_min for c in self.candidates]


class FileReport(_M):
    path: str
    fingerprint: str
    disc: str | None = None
    duration_s: float
    regime: Literal["base", "llm"]
    decision: Decision
    confidence: Confidence
    runner_up: Decision | None = None
    signals: list[SignalContribution] = []
    flags: list[str] = []  # duplicate, out_of_scope, possible_two_parter, play_all, not_ready, ...
    # Duplicates (decision 56): path of the primary copy this file duplicates, and this copy's
    # 0-based rank among that primary's copies by quality, then path. None for non-duplicates.
    duplicate_of: str | None = None
    duplicate_rank: int | None = None
    # The user gave this path as a file (FileNode.explicit): the planner renames it in place.
    # In-memory only: not part of the JSON schema.
    explicit: bool = Field(default=False, exclude=True)


class Summary(_M):
    files: int
    high: int
    medium: int
    low: int
    extras: int


class Report(_M):
    report_version: Literal[1] = REPORT_VERSION
    tool_version: str
    show_slug: str
    bundle_revision: str
    ordering: str
    season: int | None = None
    files: list[FileReport] = []
    summary: Summary
    notices: list[str] = []  # one-line advice for the user (e.g. no evidence source available)
