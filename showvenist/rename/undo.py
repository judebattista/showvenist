"""Undo a run (spec §13.5, decision 59).

Choices: a video is moved back only when the file at the logged new path still has the logged
fingerprint (sidecars carry their video's, so they are checked by existence only). A move found
already back at its old path (an earlier undo interrupted before it could log) is marked
`undone` without touching the file. Folders created by apply are never removed. Each undo
records a `run` row (command `undo`).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from showvenist.rename.movelog import _op, _same_file, recover, rename_noreplace
from showvenist.rename.types import UndoResult
from showvenist.state.db import RenameOp, StateDb
from showvenist.state.fingerprint import fingerprint as _fingerprint
from showvenist.state.history import History


def latest_run(db: StateDb) -> int | None:
    """The latest run that has `done` moves."""
    with db.transaction() as c:
        r = c.execute("select max(run_id) from rename_op where status='done'").fetchone()
    return None if r[0] is None else int(r[0])


def done_ops(db: StateDb, run_id: int) -> list[RenameOp]:
    with db.transaction() as c:
        return [_op(r) for r in c.execute(
            "select * from rename_op where run_id=? and status='done' order by id", (run_id,))]


def undo_run(
    db: StateDb,
    run_id: int | None,
    *,
    confirm: Callable[[list[RenameOp]], bool],
    exists: Callable[[Path], bool] = os.path.exists,
    rename: Callable[[Path, Path], None] = rename_noreplace,
    fingerprint_of: Callable[[Path], str] = _fingerprint,
) -> UndoResult:
    """`recover` first. `run_id` None = the latest run with `done` moves. Lists the run's `done`
    moves, asks `confirm`, then moves files back in reverse order. A move whose file is no longer
    at `new_path`, or whose `old_path` is now occupied, is skipped and reported; the rest are
    still undone. Nothing is overwritten. Moves become `undone`; the run's confirmed matches
    become `undone` via `History.mark_undone`.

    Nothing to undo returns an empty result without asking (`run_id` None when no run has
    `done` moves)."""
    recover(db, exists=exists, fingerprint_of=fingerprint_of)
    if run_id is None:
        run_id = latest_run(db)
        if run_id is None:
            return UndoResult(run_id=None)
    ops = done_ops(db, run_id)
    if not ops:
        return UndoResult(run_id=run_id)
    if not confirm(ops):
        return UndoResult(run_id=run_id, cancelled=True)
    hist = History(db)
    hist.begin_run("undo", str(run_id))
    res = UndoResult(run_id=run_id)

    def mark(op: RenameOp) -> None:
        with db.transaction() as c:
            c.execute("update rename_op set status='undone', undone_at=? where id=?",
                      (datetime.now(UTC).isoformat(timespec="seconds"), op.id))
        res.undone.append(op)

    for op in reversed(ops):
        old, new = Path(op.old_path), Path(op.new_path)
        if not exists(new):
            if exists(old) and _same_file(old, op, fingerprint_of) and op.kind != "sidecar":
                mark(op)  # already back (an interrupted undo)
            else:
                res.skipped.append((op, f"no longer at {new}"))
            continue
        if not _same_file(new, op, fingerprint_of):
            res.skipped.append((op, f"{new} now holds a different file"))
            continue
        if exists(old):
            res.skipped.append((op, f"{old} is occupied"))
            continue
        try:
            old.parent.mkdir(parents=True, exist_ok=True)
            rename(new, old)
        except FileExistsError:
            res.skipped.append((op, f"{old} is occupied"))
            continue
        except OSError as e:
            res.skipped.append((op, f"could not move back: {e}"))
            continue
        mark(op)
    hist.mark_undone(run_id)
    return res
