"""Report JSON v1 (spec §12.3). The schema is stable and versioned: scripts and eval read it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

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
