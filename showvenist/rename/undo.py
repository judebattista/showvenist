"""Undo a run (spec §13.5, decision 59).

Choices:
- A video is moved back only when the file at the logged new path still has the logged
  fingerprint (sidecars carry their video's, so they are checked by existence only).
- Order: the run's videos in reverse order, each video before its own sidecars (a sidecar
  belongs to the closest earlier video op of the run with its fingerprint). A sidecar whose
  video could not be moved back stays with its video (skipped, with the reason).
- A move found already back at its old path (an earlier undo interrupted before it could log,
  or the user moved it back by hand) is marked `undone` without touching the file; for a video
  only when its fingerprint matches, for a sidecar on existence alone.
- `--last` (`run_id` None) picks the latest run that still has `done` moves AND has not been
  undone before (no `run` row with command `undo` for it). A run whose undo skipped some moves
  therefore stops being "the last one": those leftovers are retried only by naming the run
  (`undo RUN_ID`), and `--last` moves on to the run before it.
- Folders created by apply are never removed. Each undo records a `run` row (command `undo`,
  args = the undone run's id).
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

UNDO_COMMAND = "undo"


def latest_run(db: StateDb) -> int | None:
    """The latest run with `done` moves that no earlier `undo` was applied to (see the module
    docstring)."""
    with db.transaction() as c:
        r = c.execute(
            "select max(run_id) from rename_op where status='done' and cast(run_id as text)"
            " not in (select args from run where command=?)", (UNDO_COMMAND,)).fetchone()
    return None if r[0] is None else int(r[0])


def done_ops(db: StateDb, run_id: int) -> list[RenameOp]:
    with db.transaction() as c:
        return [_op(r) for r in c.execute(
            "select * from rename_op where run_id=? and status='done' order by id", (run_id,))]


def undo_order(ops: list[RenameOp]) -> list[RenameOp]:
    """Reverse apply order by video, each video first and then its sidecars."""
    groups: list[list[RenameOp]] = []
    video_group: dict[str, list[RenameOp]] = {}
    for op in ops:  # id order
        if op.kind == "sidecar" and op.fingerprint in video_group:
            video_group[op.fingerprint].append(op)
            continue
        g = [op]
        groups.append(g)
        if op.kind != "sidecar":
            video_group[op.fingerprint] = g
    return [op for g in reversed(groups) for op in g]


def undo_run(
    db: StateDb,
    run_id: int | None,
    *,
    confirm: Callable[[list[RenameOp]], bool],
    exists: Callable[[Path], bool] = os.path.exists,
    rename: Callable[[Path, Path], None] = rename_noreplace,
    fingerprint_of: Callable[[Path], str] = _fingerprint,
) -> UndoResult:
    """`recover` first. `run_id` None = `latest_run`. Lists the run's `done` moves (in the
    order they will be reversed), asks `confirm`, then moves files back (`undo_order`). A move
    whose file is no longer at `new_path`, or whose `old_path` is now occupied, is skipped and
    reported; the rest are still undone. Nothing is overwritten. Moves become `undone`; the
    run's confirmed matches become `undone` via `History.mark_undone`.

    Nothing to undo returns an empty result without asking (`run_id` None when no run
    qualifies)."""
    recovered = recover(db, exists=exists, fingerprint_of=fingerprint_of)
    if run_id is None:
        run_id = latest_run(db)
        if run_id is None:
            return UndoResult(run_id=None, recovered=recovered)
    ops = undo_order(done_ops(db, run_id))
    if not ops:
        return UndoResult(run_id=run_id, recovered=recovered)
    if not confirm(ops):
        return UndoResult(run_id=run_id, cancelled=True, recovered=recovered)
    hist = History(db)
    hist.begin_run(UNDO_COMMAND, str(run_id))
    res = UndoResult(run_id=run_id, recovered=recovered)
    held: set[str] = set()  # fingerprints of videos that stay at their new path

    def mark(op: RenameOp) -> None:
        with db.transaction() as c:
            c.execute("update rename_op set status='undone', undone_at=? where id=?",
                      (datetime.now(UTC).isoformat(timespec="seconds"), op.id))
        res.undone.append(op)
        if op.kind != "sidecar":
            held.discard(op.fingerprint)

    def skip(op: RenameOp, why: str) -> None:
        res.skipped.append((op, why))
        if op.kind != "sidecar":
            held.add(op.fingerprint)

    for op in ops:
        old, new = Path(op.old_path), Path(op.new_path)
        if op.kind == "sidecar" and op.fingerprint in held:
            skip(op, "its video was not moved back: it stays with the video")
            continue
        if not exists(new):
            if exists(old) and _same_file(old, op, fingerprint_of):
                mark(op)  # already back (an interrupted undo, or by hand)
            else:
                skip(op, f"no longer at {new}")
            continue
        if not _same_file(new, op, fingerprint_of):
            skip(op, f"{new} now holds a different file")
            continue
        if exists(old):
            skip(op, f"{old} is occupied")
            continue
        try:
            old.parent.mkdir(parents=True, exist_ok=True)
            rename(new, old)
        except FileExistsError:
            skip(op, f"{old} is occupied")
            continue
        except OSError as e:
            skip(op, f"could not move back: {e}")
            continue
        mark(op)
    hist.mark_undone(run_id, frozenset(
        op.fingerprint for op, _ in res.skipped if op.kind != "sidecar"))
    return res
