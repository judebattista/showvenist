"""Apply a plan with a move log, and crash recovery (spec §13.4, decision 59).

Journal protocol per move: re-fingerprint the source (a video that changed since planning is
not moved); insert the `rename_op` row `planned` with what the move decides (episode, season,
number, title, source, confidence: schema v2) and commit; create the target folder (never
removed later); rename; then in ONE transaction mark the row `done`, record the file and the
confirmed match, and link the row to the match. A crash between the steps leaves a `planned`
row that `recover` reconciles from what exists on disk, confirming the match from the row.

Choices:
- The default rename is `rename_noreplace`: Linux `renameat2(RENAME_NOREPLACE)`, so a file that
  appears at the target between the check and the rename is never overwritten; on a filesystem
  without it, an existence check right before `os.rename`.
- Cross-filesystem moves (`Plan.cross_filesystem`) run only when `confirm_cross_fs` agrees;
  they copy to a hidden partial file next to the target, fsync, check the size, link it into
  place without replacing anything, and only then remove the original (`copy_move`).
- Recovery marks a completed move `done` and confirms its match from the journaled row (a
  sidecar links its video's match). It writes no match for a row journaled before schema v2
  (no decision recorded) or for a file that a later run decided again; both are reported in
  `unconfirmed`. A `planned` row whose new path holds a file with another fingerprint, or
  where both paths exist, is left `planned` and reported: nothing is guessed. The one
  exception is a cross-filesystem row (`cross_fs`) whose copy at the new path verifies against
  the original and the logged fingerprint: `copy_move` was interrupted after the copy was in
  place, so recovery finishes it by removing the original.
- Recovery removes our own `.showvenist-partial` copy of a `planned` row, but only while the
  original still exists.
- A duplicate (`PlannedMove.primary`) is renamed only after its primary's move succeeded, so a
  copy never carries `_dupe_N` while its primary keeps its old name.
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
from dataclasses import fields, replace
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


def partial_path(new: Path) -> Path:
    """The hidden partial copy `copy_move` writes beside `new`."""
    new = Path(new)
    return new.with_name(f".{new.name}.showvenist-partial")


def copy_move(old: Path, new: Path) -> None:
    """A move across filesystems that never overwrites and never loses the original before a
    complete copy is in place: copy to a hidden partial file beside `new`, fsync, check the
    size, hard-link it to `new` (fails if `new` exists), remove the partial, then the original."""
    old, new = Path(old), Path(new)
    part = partial_path(new)
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


_OP_FIELDS = tuple(f.name for f in fields(RenameOp))


def _op(r) -> RenameOp:
    """A RenameOp from a `select *` row (columns RenameOp does not carry are ignored)."""
    return RenameOp(**{k: r[k] for k in _OP_FIELDS})


def _same_file(path: Path, op: RenameOp, fingerprint_of: Callable[[Path], str]) -> bool:
    """A sidecar carries its video's fingerprint, so only videos are checked."""
    if op.kind == "sidecar":
        return True
    try:
        return fingerprint_of(path) == op.fingerprint
    except OSError:
        return False


def _verified_copy(old: Path, new: Path, op: RenameOp,
                   fingerprint_of: Callable[[Path], str]) -> bool:
    """The file at `new` is a finished copy of `old` (and, for a video, the logged file)."""
    try:
        fo, fn = fingerprint_of(old), fingerprint_of(new)
    except OSError:
        return False
    return fo == fn and (op.kind == "sidecar" or fo == op.fingerprint)


def _recovered_match(c, hist: History, op: RenameOp, row) -> tuple[int | None, bool]:
    """(match id, confirmed?) for a move recovery completed, inside its transaction."""
    if op.kind == "sidecar":
        r = c.execute(
            "select match_id from rename_op where run_id=? and fingerprint=? and kind<>'sidecar'"
            " and status='done' and id<? order by id desc limit 1",
            (op.run_id, op.fingerprint, op.id)).fetchone()
        return (r[0] if r else None), True
    if row["match_source"] is None or op.show_slug is None:
        return None, False  # journaled before schema v2: the decision is not on record
    later = c.execute(
        "select 1 from match where fingerprint=? and show_slug=? and status='confirmed'"
        " and run_id>?", (op.fingerprint, op.show_slug, op.run_id)).fetchone()
    if later:
        return None, False  # decided again by a later run: that decision stands
    move = PlannedMove(
        kind=op.kind, old=Path(op.old_path), new=Path(op.new_path), fingerprint=op.fingerprint,
        show_slug=op.show_slug, episode_id=op.episode_id if op.kind != "extra" else None,
        season=row["season"], episode=row["episode"], episode_end=row["episode_end"],
        title=row["title"],
        pinned=row["match_source"] == "manual", confidence=row["confidence"],
        confirmed=bool(row["prior_match"]), disc=row["disc_label"],
    )
    mid = _record(c, hist, row["bundle_revision"], move, Path(op.new_path), op.run_id, {})
    return mid, True


def recover(
    db: StateDb,
    *,
    exists: Callable[[Path], bool] = os.path.exists,
    fingerprint_of: Callable[[Path], str] = _fingerprint,
    remove: Callable[[Path], None] = os.unlink,
) -> RecoveryReport:
    """Reconcile leftover `planned` rows (§13.4): new path exists and old does not -> `done`
    and the match confirmed from the row; old path exists and new does not -> the row is
    dropped (rolled back); neither -> left `planned` and reported in `unresolved`. Called at
    the start of every command that could move files (`identify --apply`, `assign --rename`,
    `undo`), and again (a no-op by then) inside `apply_plan` and `undo_run`.

    Also left `planned` (unresolved): the new path holds a video whose fingerprint is not the
    logged one; both paths exist, unless the row is a cross-filesystem move whose copy
    verifies (then the original is removed and the row is done). A `.showvenist-partial` copy
    of a planned row is removed while its original exists, and kept (reported) otherwise.
    Every unresolved row gets a line in `notes`. Nothing is guessed."""
    rep = RecoveryReport()
    hist = History(db)
    with db.transaction() as c:
        rows = list(c.execute("select * from rename_op where status='planned' order by id"))

    def complete(op: RenameOp, row) -> None:
        with db.transaction() as c:
            c.execute("update rename_op set status='done', applied_at=? where id=?",
                      (_now(), op.id))
            mid, confirmed = _recovered_match(c, hist, op, row)
            if mid is not None:
                c.execute("update rename_op set match_id=? where id=?", (mid, op.id))
        done = replace(op, status="done", match_id=mid)
        rep.completed.append(done)
        if not confirmed:
            rep.unconfirmed.append(done)
            rep.notes.append(f"{op.new_path}: move completed, but its match was not recorded "
                             "(not in the move log, or decided again since): identify it again")

    def unresolved(op: RenameOp, why: str) -> None:
        rep.unresolved.append(op)
        rep.notes.append(f"left as planned: {op.old_path} -> {op.new_path}: {why}")

    for row in rows:
        op = _op(row)
        old, new = Path(op.old_path), Path(op.new_path)
        part = partial_path(new)
        if os.path.lexists(part):
            if exists(old):
                try:
                    remove(part)
                    rep.removed_partials.append(part)
                except OSError as e:
                    rep.notes.append(f"{part}: could not remove our partial copy ({e})")
            else:
                rep.notes.append(f"{part}: partial copy kept, its original {old} is gone")
        o, n = exists(old), exists(new)
        if n and not o:
            if _same_file(new, op, fingerprint_of):
                complete(op, row)
            else:
                unresolved(op, "the new path holds another file")
        elif o and not n:
            with db.transaction() as c:
                c.execute("delete from rename_op where id=? and status='planned'", (op.id,))
            rep.rolled_back.append(op)
        elif o and n:
            if not row["cross_fs"]:
                unresolved(op, "both paths exist; check which one to keep by hand")
            elif not _verified_copy(old, new, op, fingerprint_of):
                unresolved(op, "both paths exist and the copy does not verify against the "
                               "original; check by hand")
            else:
                try:
                    remove(old)  # the copy is complete and verified: finish the move
                except OSError as e:
                    unresolved(op, f"the copy verifies but the original could not be removed "
                                   f"({e})")
                    continue
                rep.notes.append(f"{old}: finished an interrupted cross-filesystem move (the "
                                 f"copy at {new} verified, the original was removed)")
                complete(op, row)
        else:
            unresolved(op, "neither path exists")
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
    run_id: int | None = None,
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
    gets the next free collision suffix instead (the plan's other targets stay reserved), and
    a video must still have the planned fingerprint (re-read right before its move, cross-
    filesystem copies included): a changed file fails with a reason and is not moved.

    `run_id`: an open `run` row to record the moves under (one row per invocation, §7.2, when
    a caller applies several plans); None = begin a run here after `confirm`. A duplicate is
    skipped (in `failed`, with the reason) when its primary was not moved."""
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
    if run_id is None:
        run_id = hist.begin_run(command, args).id
    result = ApplyResult(run_id=run_id, recovered=recovered)
    reserved = {m.new for m in plan.moves}
    actual: dict[int, Path] = {}  # move index -> where its file went
    match_of: dict[int, int | None] = {}
    for i, move in enumerate(plan.moves):
        if move.parent is not None and move.parent not in actual:
            result.failed.append((move, "its video was not moved"))
            continue
        if move.primary is not None and move.primary not in actual:
            result.failed.append((move, f"not renamed: its primary "
                                  f"{plan.moves[move.primary].old.name} was not moved"))
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
        if move.kind != "sidecar":  # the file must still be the one that was planned
            try:
                now_fp = fingerprint_of(move.old)
            except OSError as e:
                result.failed.append((move, f"cannot read it to check it ({e}): not moved"))
                continue
            if now_fp != move.fingerprint:
                result.failed.append((move, "the file changed since it was planned (its "
                                      "fingerprint differs): not moved; plan again"))
                continue
        reserved.discard(move.new)
        base = new
        n = 1
        while exists(new) or new in reserved:
            n += 1
            new = naming.with_collision_index(base, n)
            n_used = n
        with db.transaction() as c:
            video = move.kind != "sidecar"
            cur = c.execute(
                "insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status,"
                " show_slug, episode_id, dupe_index, season, episode, title, match_source,"
                " confidence, bundle_revision, disc_label, prior_match, cross_fs, episode_end)"
                " values (?,?,?,?,?,'planned',?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, move.fingerprint, move.kind, str(move.old), str(new), move.show_slug,
                 move.episode_id if move.kind in ("episode", "duplicate") else None,
                 move.dupe_index,
                 move.season if video else None, move.episode if video else None,
                 move.title if video else None,
                 ("manual" if move.pinned else "auto") if video else None,
                 move.confidence if video else None,
                 plan.bundle_revision if video else None,
                 move.disc if video else None,
                 int(move.confirmed) if video else None,
                 int(id(move) in cross_ids),
                 move.episode_end if video else None),
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
                match_id = _record(c, hist, plan.bundle_revision, move, new, run_id, match_of)
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


def _record(c, hist: History, bundle_revision: str | None, move: PlannedMove, new: Path,
            run_id: int, match_of: dict[int, int | None]) -> int | None:
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
                        confidence=move.confidence, bundle_revision=bundle_revision)
    return m.id


def match_args(move: PlannedMove) -> dict:
    """Keyword arguments for `History.confirm` describing what a move decided."""
    return dict(
        episode_id=move.episode_id, season=move.season, episode=move.episode, title=move.title,
        source="manual" if move.pinned else "auto", episode_end=move.episode_end,
    )
