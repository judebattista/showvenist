"""Eval contract (spec §16.1, decision 41): manifest rows, resolved labels, metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LabelKind = Literal["episode", "extra", "episode?"]


@dataclass(frozen=True)
class ManifestRow:
    """One CSV row: path,show,label,order,disc. `label` is raw text: S01E02 | extra | episode?"""

    path: str
    show: str  # bundle slug
    label: str
    order: str = "aired"
    disc: str = ""


@dataclass(frozen=True)
class ResolvedLabel:
    row: ManifestRow
    kind: LabelKind
    episode_id: str | None = None  # set for kind == "episode"
    duplicate_set: str | None = None  # files sharing a label (§14); kept out of precision
    multi: bool = False  # `extra-multi`: several episodes on one track; an extra for scoring
    # a two-part episode muxed as one file, labeled "S01E01+S01E02": correct if either is chosen
    episode_ids: tuple[str, ...] = ()


@dataclass
class Problem:
    row: ManifestRow
    reason: str  # reported and left out, never counted as an error


@dataclass
class RateWithBound:
    errors: int
    n: int
    upper_95: float  # one-sided Clopper-Pearson upper bound on the error rate


@dataclass
class MetricSet:
    files: int = 0
    top1: float | None = None
    precision_high: RateWithBound | None = None
    coverage_high: float | None = None
    unmatched_precision: float | None = None
    unmatched_recall: float | None = None
    reliability: dict[str, float] = field(default_factory=dict)  # level -> correct share
    # decision 41: the separate `episode?` line
    episode_unknown_files: int = 0
    episode_unknown_high_as_episode: int = 0
    episode_unknown_flagged: int = 0
    episode_unknown_provable_errors: int = 0
    # `extra-multi` files (several episodes on one track): scored as extras, reported separately
    extra_multi_files: int = 0
    extra_multi_as_extra: int = 0
    extra_multi_as_episode_high: int = 0  # the error that matters: a play-all taken for an episode
    cost_usd: float = 0.0


@dataclass
class EvalResult:
    overall: MetricSet
    per_season: dict[str, MetricSet] = field(default_factory=dict)  # "<slug>-S01"
    per_show: dict[str, MetricSet] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)


@dataclass(frozen=True)
class FileDecision:
    """What the tool decided for one file; the metrics function compares it with the label.

    The pipeline (analyze -> score -> fuse -> assign -> confidence) produces these; the
    harness never runs it itself, so metrics are testable on hand-built decisions.
    """

    path: str
    kind: Literal["episode", "extra"]
    episode_id: str | None  # set when kind == "episode"
    level: Literal["high", "medium", "low"]
    delta: float | None = None
    duplicate: bool = False  # flagged as a duplicate (§14)
