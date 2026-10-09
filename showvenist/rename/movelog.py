"""Apply a plan with a move log, and crash recovery (spec §13.4, decision 59). Contract; the
bodies are task sv-o4f.3."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from showvenist.rename.types import ApplyResult, Plan, PlannedMove, RecoveryReport
from showvenist.state.db import StateDb


def recover(db: StateDb, *, exists: Callable[[Path], bool] = os.path.exists) -> RecoveryReport:
    """Reconcile leftover `planned` rows: new path exists and old does not -> `done`; old path
    exists and new does not -> the row is dropped (rolled back); neither -> left `planned` and
    reported in `unresolved`. Called at the start of every command that could move files
    (`identify --apply`, `assign --rename`, `undo`)."""
    raise NotImplementedError("sv-o4f.3")


def apply_plan(
    plan: Plan,
    db: StateDb,
    *,
    command: str,
    args: str,
    confirm: Callable[[Plan], bool],
    rename: Callable[[Path, Path], None] = os.rename,
) -> ApplyResult:
    """`recover` first; refuse a plan with `errors`; ask `confirm(plan)`; then for each move in
    order: insert the `rename_op` row `planned` (committed), `rename(old, new)`, mark it `done`
    and write the match (confirmed) in the same transaction. A move that raises is recorded in
    `failed` and the run continues with moves that do not depend on it (a sidecar follows its
    video). Never overwrites, deletes or removes folders. `rename` is injectable so a crash
    test can kill the process between the two steps."""
    raise NotImplementedError("sv-o4f.3")


def match_args(move: PlannedMove) -> dict:
    """Keyword arguments for `History.confirm` describing what a move decided."""
    return dict(
        episode_id=move.episode_id, season=move.season, episode=move.episode, title=move.title,
        source="manual" if move.pinned else "auto",
    )
