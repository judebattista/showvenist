"""Plan, apply, recovery and undo types (spec §13, decisions 56-59). The M5 contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from showvenist.state.db import OpKind, RenameOp

SkipReason = Literal[
    "named",            # already named as an episode (§5.5): never renamed or moved
    "ignored",          # inside a .showvenist-ignore folder (decision 51)
    "not_ready",        # still being written (§5.2)
    "unreadable",
    "below_high",       # medium or low and not pinned: needs review
    "out_of_scope",     # evidence points to another season (§11.3) and not pinned
    "extras_disabled",  # identified extra but move_extras is off
    "already_in_place", # the target is the file's own path
    "confirmed_extra",  # already a confirmed extra in history (re-runs skip it)
    "duplicate_primary_missing",
]


@dataclass(frozen=True)
class PlannedMove:
    """One rename or move. A sidecar move points at its video's move through `parent`."""

    kind: OpKind
    old: Path
    new: Path
    fingerprint: str
    show_slug: str
    episode_id: str | None = None  # None for extras and sidecars of extras
    season: int | None = None  # in the show's naming order (§13.1)
    episode: int | None = None
    title: str | None = None
    parent: int | None = None  # index into Plan.moves of the video this sidecar follows
    dupe_index: int | None = None  # X of `dupe_X` (kind 'duplicate')
    collision_index: int | None = None  # n of a numeric suffix `_n` added to avoid a collision
    pinned: bool = False
    confidence: Literal["high", "medium", "low"] | None = None
    reason: str = ""  # one line for the plan listing


@dataclass(frozen=True)
class SkippedFile:
    path: Path
    reason: SkipReason
    detail: str = ""


@dataclass
class PlanOptions:
    naming_order: str | None = None  # ordering whose S/E numbers name the files; None = bundle's
    template: str = "{show_snake}_s{s:02}e{e:02}"
    move_extras: bool = True
    # paths the user unticked on the plan screen or excluded some other way
    exclude: set[Path] = field(default_factory=set)


@dataclass
class Plan:
    show_slug: str
    naming_order: str
    moves: list[PlannedMove] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    # Preflight failures (permission, cross-filesystem, unresolvable target...). Any entry means
    # nothing may be applied (§13.3).
    errors: list[str] = field(default_factory=list)
    cross_filesystem: list[PlannedMove] = field(default_factory=list)  # need their own confirmation

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ApplyResult:
    run_id: int
    done: list[PlannedMove] = field(default_factory=list)
    failed: list[tuple[PlannedMove, str]] = field(default_factory=list)
    recovered: RecoveryReport | None = None  # leftovers reconciled before this run
    cancelled: bool = False  # the user declined the confirmation


@dataclass
class RecoveryReport:
    completed: list[RenameOp] = field(default_factory=list)  # planned -> done (target exists)
    rolled_back: list[RenameOp] = field(default_factory=list)  # planned dropped (source exists)
    unresolved: list[RenameOp] = field(default_factory=list)  # neither path exists: left planned


@dataclass
class UndoResult:
    run_id: int | None
    undone: list[RenameOp] = field(default_factory=list)
    skipped: list[tuple[RenameOp, str]] = field(default_factory=list)  # file gone / old path taken
    cancelled: bool = False
