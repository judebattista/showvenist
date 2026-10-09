"""Undo a run (spec §13.5, decision 59). Contract; the body is task sv-o4f.3."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from showvenist.rename.types import UndoResult
from showvenist.state.db import RenameOp, StateDb


def undo_run(
    db: StateDb,
    run_id: int | None,
    *,
    confirm: Callable[[list[RenameOp]], bool],
    exists: Callable[[Path], bool] = os.path.exists,
    rename: Callable[[Path, Path], None] = os.rename,
) -> UndoResult:
    """`recover` first. `run_id` None = the latest run with `done` moves. Lists the run's `done`
    moves, asks `confirm`, then moves files back in reverse order. A move whose file is no longer
    at `new_path`, or whose `old_path` is now occupied, is skipped and reported; the rest are
    still undone. Nothing is overwritten. Moves become `undone`; the run's confirmed matches
    become `undone` via `History.mark_undone`."""
    raise NotImplementedError("sv-o4f.3")
