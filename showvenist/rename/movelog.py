"""Apply a plan with a move log, and crash recovery (spec §13.4, decision 59).

Journal protocol per move: insert the `rename_op` row `planned` and commit; create the target
folder (never removed later); rename; then in ONE transaction mark the row `done`, record the
file and the confirmed match, and link the row to the match. A crash between the steps leaves a
`planned` row that `recover` reconciles from what exists on disk.

Choices:
- The default rename is `rename_noreplace`: Linux `renameat2(RENAME_NOREPLACE)`, so a file that
  appears at the target between the check and the rename is never overwritten; on a filesystem
  without it, an existence check right before `os.rename`.
- Cross-filesystem moves (`Plan.cross_filesystem`) run only when `confirm_cross_fs` agrees;
  they copy to a hidden partial file next to the target, fsync, check the size, link it into
  place without replacing anything, and only then remove the original (`copy_move`).
- Recovery marks a completed move `done` but writes no match for it (the plan that carried
  season, episode and title is gone); the file is at its new name, so the next run picks it up.
  A `planned` row whose new path holds a file with another fingerprint, or where both paths
  exist, is left `planned` and reported: nothing is guessed.
- A failure to write state.db after a rename stops the run (later moves are not attempted);
  the row stays `planned` and the next recovery completes it.
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from showvenist.rename import naming
from showvenist.rename.types import ApplyResult, Plan, PlannedMove, RecoveryReport
from showvenist.state.db import RenameOp, StateDb
from showvenist.state.fingerprint import fingerprint as _fingerprint
from showvenist.state.history import History

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_renameat2 = None
try:  # glibc >= 2.28
    _libc = ctypes.CDLL(None, use_errno=True)
    _renameat2 = _libc.renameat2
    _renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                           ctypes.c_uint]
    _renameat2.restype = ctypes.c_int
except (OSError, AttributeError):  # pragma: no cover - other libcs
    _renameat2 = None


class PlanRefused(ValueError):
    """apply_plan was handed a plan with preflight errors (§13.3): nothing was moved."""


def rename_noreplace(old: Path, new: Path) -> None:
    """Rename `old` to `new`, raising FileExistsError instead of replacing an existing `new`."""
    if _renameat2 is not None:
        r = _renameat2(_AT_FDCWD, os.fsencode(old), _AT_FDCWD, os.fsencode(new),
                       _RENAME_NOREPLACE)
        if r == 0:
            return
        e = ctypes.get_errno()
        if e == errno.EEXIST:
            raise FileExistsError(e, os.strerror(e), str(new))
        if e not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
            raise OSError(e, os.strerror(e), str(old))
    if os.path.lexists(new):  # no RENAME_NOREPLACE on this filesystem: check, then rename
        raise FileExistsError(errno.EEXIST, "target exists", str(new))
    os.rename(old, new)


def copy_move(old: Path, new: Path) -> None:
    """A move across filesystems that never overwrites and never loses the original before a
    complete copy is in place: copy to a hidden partial file beside `new`, fsync, check the
    size, hard-link it to `new` (fails if `new` exists), remove the partial, then the original."""
    old, new = Path(old), Path(new)
    part = new.with_name(f".{new.name}.showvenist-partial")
    if os.path.lexists(new):
        raise FileExistsError(errno.EEXIST, "target exists", str(new))
    if os.path.lexists(part):
        raise FileExistsError(errno.EEXIST, "a partial copy is in the way", str(part))
    try:
        shutil.copy2(old, part)
        with open(part, "rb") as fh:
            os.fsync(fh.fileno())
        if os.stat(part).st_size != os.stat(old).st_size:
            raise OSError(errno.EIO, "copy is incomplete", str(part))
        try:
            os.link(part, new)
        except FileExistsError:
            raise
        except OSError:  # no hard links here: same-filesystem no-replace rename instead
            rename_noreplace(part, new)
    finally:
        if os.path.lexists(part):
            os.unlink(part)  # our own partial copy (or the second link to the finished one)
    if _fingerprint(new) != _fingerprint(old):  # head, tail and size must agree before we let go
        os.unlink(new)  # the copy we just made, never the original
        raise OSError(errno.EIO, "the copy does not match the original; the original is kept",
                      str(new))
    os.unlink(old)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _op(r) -> RenameOp:
    return RenameOp(**{k: r[k] for k in r.keys()})


def _same_file(path: Path, op: RenameOp, fingerprint_of: Callable[[Path], str]) -> bool:
    """A sidecar carries its video's fingerprint, so only videos are checked."""
    if op.kind == "sidecar":
        return True
    try:
        return fingerprint_of(path) == op.fingerprint
    except OSError:
        return False


def recover(
    db: StateDb,
    *,
    exists: Callable[[Path], bool] = os.path.exists,
    fingerprint_of: Callable[[Path], str] = _fingerprint,
) -> RecoveryReport:
    """Reconcile leftover `planned` rows: new path exists and old does not -> `done`; old path
    exists and new does not -> the row is dropped (rolled back); neither -> left `planned` and
    reported in `unresolved`. Called at the start of every command that could move files
    (`identify --apply`, `assign --rename`, `undo`).

    Also left `planned` (unresolved): both paths exist, or the new path holds a video whose
    fingerprint is not the logged one. A recovered `done` row gets no match (see the module
    docstring)."""
    rep = RecoveryReport()
    with db.transaction() as c:
        rows = [_op(r) for r in c.execute(
            "select * from rename_op where status='planned' order by id")]
    for op in rows:
        old, new = Path(op.old_path), Path(op.new_path)
        o, n = exists(old), exists(new)
        if n and not o and _same_file(new, op, fingerprint_of):
            with db.transaction() as c:
                c.execute("update rename_op set status='done', applied_at=? where id=?",
                          (_now(), op.id))
            rep.completed.append(replace(op, status="done"))
        elif o and not n:
            with db.transaction() as c:
                c.execute("delete from rename_op where id=? and status='planned'", (op.id,))
            rep.rolled_back.append(op)
        else:
            rep.unresolved.append(op)
    return rep


def apply_plan(
    plan: Plan,
    db: StateDb,
    *,
    command: str,
    args: str,
    confirm: Callable[[Plan], bool],
    rename: Callable[[Path, Path], None] = rename_noreplace,
    confirm_cross_fs: Callable[[list[PlannedMove]], bool] | None = None,
    move_across: Callable[[Path, Path], None] = copy_move,
    exists: Callable[[Path], bool] = os.path.exists,
    fingerprint_of: Callable[[Path], str] = _fingerprint,
) -> ApplyResult:
    """`recover` first; refuse a plan with `errors`; ask `confirm(plan)`; then for each move in
    order: insert the `rename_op` row `planned` (committed), `rename(old, new)`, mark it `done`
    and write the match (confirmed) in the same transaction. A move that raises is recorded in
    `failed` and the run continues with moves that do not depend on it (a sidecar follows its
    video). Never overwrites, deletes or removes folders. `rename` is injectable so a crash
    test can kill the process between the two steps.

    Refusals raise `PlanRefused` before anything is written. Moves in `plan.cross_filesystem`
    are asked about once through `confirm_cross_fs` (after `confirm`); without it, or when it
    declines, they are recorded as failed and nothing else changes. A declined `confirm`
    returns `cancelled` with `run_id` 0 (no run is recorded). Right before each rename the
    source must still exist and the target must still be free; a target taken since planning
    gets the next free collision suffix instead (the plan's other targets stay reserved)."""
    if not plan.ok:
        raise PlanRefused("the plan has errors: " + "; ".join(plan.errors))
    recovered = recover(db, exists=exists, fingerprint_of=fingerprint_of)
    if not plan.moves:
        return ApplyResult(run_id=0, recovered=recovered)
    if not confirm(plan):
        return ApplyResult(run_id=0, recovered=recovered, cancelled=True)
    cross_ids = {id(m) for m in plan.cross_filesystem}
    allow_cross = bool(cross_ids) and confirm_cross_fs is not None and bool(
        confirm_cross_fs(list(plan.cross_filesystem)))
    hist = History(db)
    run = hist.begin_run(command, args)
    result = ApplyResult(run_id=run.id, recovered=recovered)
    reserved = {m.new for m in plan.moves}
    actual: dict[int, Path] = {}  # move index -> where its file went
    match_of: dict[int, int | None] = {}
    for i, move in enumerate(plan.moves):
        if move.parent is not None and move.parent not in actual:
            result.failed.append((move, "its video was not moved"))
            continue
        if id(move) in cross_ids and not allow_cross:
            result.failed.append((move, "crosses filesystems: not confirmed"))
            continue
        new, n_used = move.new, move.collision_index
        if move.parent is not None and actual[move.parent] != plan.moves[move.parent].new:
            new = naming.sidecar_target(plan.moves[move.parent].old, move.old,
                                        actual[move.parent])
        if not exists(move.old):
            result.failed.append((move, "the file is gone"))
            continue
        reserved.discard(move.new)
        base = new
        n = 1
        while exists(new) or new in reserved:
            n += 1
            new = naming.with_collision_index(base, n)
            n_used = n
        with db.transaction() as c:
            cur = c.execute(
                "insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status,"
                " show_slug, episode_id, dupe_index) values (?,?,?,?,?,'planned',?,?,?)",
                (run.id, move.fingerprint, move.kind, str(move.old), str(new), move.show_slug,
                 move.episode_id if move.kind in ("episode", "duplicate") else None,
                 move.dupe_index),
            )
            op_id = int(cur.lastrowid)
        mover = move_across if id(move) in cross_ids else rename
        error: str | None = None
        for _attempt in range(100):
            try:
                new.parent.mkdir(parents=True, exist_ok=True)
                mover(move.old, new)
                break
            except FileExistsError:  # appeared after the check: never overwrite
                n += 1
                new = naming.with_collision_index(base, n)
                n_used = n
                while exists(new) or new in reserved:
                    n += 1
                    new = naming.with_collision_index(base, n)
                    n_used = n
                with db.transaction() as c:
                    c.execute("update rename_op set new_path=? where id=?", (str(new), op_id))
            except OSError as e:
                error = str(e)
                break
        else:
            error = "no free target name"
        if error is not None:
            if exists(move.old) and not exists(new):  # nothing happened: drop the journal row
                with db.transaction() as c:
                    c.execute("delete from rename_op where id=? and status='planned'", (op_id,))
            result.failed.append((move, error))
            continue
        try:
            with db.transaction() as c:
                match_id = _record(c, hist, plan, move, new, run.id, match_of)
                c.execute(
                    "update rename_op set status='done', applied_at=?, match_id=? where id=?",
                    (_now(), match_id, op_id))
        except (sqlite3.Error, OSError) as e:
            actual[i] = new
            result.failed.append((
                move, f"moved to {new}, but the move log could not be updated ({e}); the "
                "next run's recovery completes it"))
            break
        actual[i] = new
        match_of[i] = match_id
        done = move if (new == move.new) else replace(move, new=new, collision_index=n_used)
        result.done.append(done)
    return result


def _record(c, hist: History, plan: Plan, move: PlannedMove, new: Path, run_id: int,
            match_of: dict[int, int | None]) -> int | None:
    """File and match rows for a finished move, inside the caller's transaction."""
    if move.kind == "sidecar":
        return match_of.get(move.parent) if move.parent is not None else None
    hist.touch_file_in(c, move.fingerprint, size=os.stat(new).st_size, path=str(new),
                       show_slug=move.show_slug, disc_label=move.disc)
    if move.confirmed:  # decided earlier: link that decision, write none (undo leaves it)
        row = c.execute(
            "select id from match where fingerprint=? and show_slug=? and status='confirmed'"
            " order by id desc limit 1", (move.fingerprint, move.show_slug)).fetchone()
        return int(row[0]) if row else None
    m = hist.confirm_in(c, move.fingerprint, move.show_slug, **match_args(move), run_id=run_id,
                        confidence=move.confidence, bundle_revision=plan.bundle_revision)
    return m.id


def match_args(move: PlannedMove) -> dict:
    """Keyword arguments for `History.confirm` describing what a move decided."""
    return dict(
        episode_id=move.episode_id, season=move.season, episode=move.episode, title=move.title,
        source="manual" if move.pinned else "auto",
    )
