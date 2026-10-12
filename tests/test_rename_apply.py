"""Apply with the move log, crash recovery and undo (spec §13.3-13.5, decision 59).

Everything runs on tiny files under tmp_path. Safety invariants checked throughout: no file is
deleted, nothing is overwritten, no folder is removed, and undo restores the original set of
files byte for byte."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from showvenist.rename.movelog import (
    PlanRefused,
    apply_plan,
    copy_move,
    recover,
    rename_noreplace,
)
from showvenist.rename.undo import undo_run
from showvenist.state.db import StateDb
from showvenist.state.history import History
from tests.test_rename_plan import SLUG, plan_for, row, season, snapshot, touch

REPO = Path(__file__).resolve().parent.parent


def ops(db) -> list[tuple]:
    return [tuple(r) for r in db.conn.execute(
        "select kind, status, old_path, new_path, match_id is not null from rename_op order by id")]


def statuses(db) -> list[str]:
    return [r[0] for r in db.conn.execute("select status from rename_op order by id")]


def build(tmp_path: Path):
    """A season with two discs: two episodes (one pinned, one with a sidecar), an extra, and a
    duplicate of episode 1."""
    s = season(tmp_path)
    f = {
        "e1": touch(s / "Disc 1" / "t00.mkv"),
        "e2": touch(s / "Disc 2" / "t00.mkv"),
        "x": touch(s / "Disc 2" / "t01.mkv"),
        "d": touch(s / "Disc 2" / "t02.mkv", (s / "Disc 1" / "t00.mkv").read_bytes()[:-1] + b"!"),
    }
    touch(s / "Disc 1" / "t00.en.srt", b"1\n00:00:01,000 --> 00:00:02,000\nhello\n")
    rows = [row(f["e1"], 1), row(f["e2"], 2, level="medium", flags=["pinned"]),
            row(f["x"], None), row(f["d"], 1, dup_of=f["e1"], rank=0, flags=["duplicate"])]
    return s, f, rows


def test_rename_noreplace_and_copy_move_never_overwrite(tmp_path):
    a, b = touch(tmp_path / "a", b"A"), touch(tmp_path / "b", b"B")
    with pytest.raises(FileExistsError):
        rename_noreplace(a, b)
    with pytest.raises(FileExistsError):
        copy_move(a, b)
    assert a.read_bytes() == b"A" and b.read_bytes() == b"B"
    rename_noreplace(a, tmp_path / "c")
    assert not a.exists() and (tmp_path / "c").read_bytes() == b"A"
    (tmp_path / "sub").mkdir()
    copy_move(tmp_path / "c", tmp_path / "sub" / "d")
    assert (tmp_path / "sub" / "d").read_bytes() == b"A" and not (tmp_path / "c").exists()
    assert sorted(p.name for p in (tmp_path / "sub").iterdir()) == ["d"]  # no partial left


def test_apply_happy_path_journal_and_history(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows)
    files0, dirs0 = snapshot(tmp_path)
    db = StateDb(tmp_path / "state" / "state.db")
    asked = []
    res = apply_plan(plan, db, command="identify --apply", args="x",
                     confirm=lambda p: asked.append(p) or True)
    assert asked == [plan] and not res.failed and not res.cancelled
    assert [m.kind for m in res.done] == ["episode", "sidecar", "episode", "extra", "duplicate"]
    assert sorted(p.relative_to(s).as_posix() for p in s.rglob("*") if p.is_file()) == [
        "Disc 2/synth_show_s01e01_dupe_1.mkv",
        "extras/Disc 2__t01.mkv",
        "synth_show_s01e01.en.srt",
        "synth_show_s01e01.mkv",
        "synth_show_s01e02.mkv",
    ]
    files1, dirs1 = snapshot(tmp_path)
    assert sorted(files0.values()) == sorted(v for k, v in files1.items()
                                             if not k.startswith("state"))  # same bytes
    assert dirs0 <= dirs1  # no folder removed (extras/ added)
    assert statuses(db) == ["done"] * 5
    assert all(o[4] for o in ops(db))  # every op linked to a match (the sidecar to its video's)
    rows_ = db.conn.execute("select fingerprint, episode_id, status, source, confidence, run_id,"
                            " bundle_revision from match order by id").fetchall()
    assert [(r[1], r[2], r[3], r[4]) for r in rows_] == [
        ("tmdb:100", "confirmed", "auto", "high"),
        ("tmdb:101", "confirmed", "manual", "medium"),
        (None, "confirmed", "auto", "high"),
        ("tmdb:100", "confirmed", "auto", "high"),
    ]
    assert {r[5] for r in rows_} == {res.run_id} and rows_[0][6] == plan.bundle_revision
    h = History(db)
    mf = h.get_file(plan.moves[0].fingerprint)
    assert mf.last_path == str(s / "synth_show_s01e01.mkv") and mf.disc_label == "Disc 1"
    assert h.highest_dupe_index(SLUG, "tmdb:100") == 1
    assert db.conn.execute("select command from run").fetchall()[0][0] == "identify --apply"


def test_refused_and_declined_plans_move_and_write_nothing(tmp_path):
    s, f, rows = build(tmp_path)
    before = snapshot(tmp_path)
    db = StateDb(tmp_path / "state.db")
    bad = plan_for(*rows)
    bad.errors.append("no write permission")
    with pytest.raises(PlanRefused):
        apply_plan(bad, db, command="c", args="", confirm=lambda p: True)
    res = apply_plan(plan_for(*rows), db, command="c", args="", confirm=lambda p: False)
    assert res.cancelled and res.run_id == 0 and not res.done
    assert db.conn.execute("select count(*) from run").fetchone()[0] == 0
    assert db.conn.execute("select count(*) from rename_op").fetchone()[0] == 0
    assert db.conn.execute("select count(*) from match").fetchone()[0] == 0
    files, dirs = snapshot(tmp_path)
    assert {k: v for k, v in files.items() if not k.startswith("state.db")} == before[0]


def test_target_taken_after_planning_gets_a_suffix(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows)
    intruder = touch(s / "synth_show_s01e01.mkv", b"appeared after planning")
    db = StateDb(tmp_path / "state.db")
    real = rename_noreplace
    raced = []

    def racing_rename(old, new):  # the user drops a file at the target mid-move
        if new.name == "synth_show_s01e02.mkv" and not raced:
            raced.append(new)
            new.write_bytes(b"raced in")
        real(old, new)

    res = apply_plan(plan, db, command="c", args="", confirm=lambda p: True,
                     rename=racing_rename)
    assert not res.failed
    assert intruder.read_bytes() == b"appeared after planning"
    assert (s / "synth_show_s01e02.mkv").read_bytes() == b"raced in"
    done = {m.old: m for m in res.done}
    assert done[f["e1"]].new == s / "synth_show_s01e01_2.mkv"
    assert done[f["e1"]].collision_index == 2
    assert (s / "synth_show_s01e01_2.en.srt").exists()  # the sidecar follows the suffixed name
    assert done[f["e2"]].new == s / "synth_show_s01e02_2.mkv"
    logged = {Path(o[2]): Path(o[3]) for o in ops(db)}
    assert logged[f["e1"]] == s / "synth_show_s01e01_2.mkv"
    assert logged[f["e2"]] == s / "synth_show_s01e02_2.mkv"


def test_failed_and_vanished_moves_do_not_stop_the_run(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows)
    f["x"].unlink()  # the extra vanished after planning (test setup, not the code under test)
    db = StateDb(tmp_path / "state.db")

    def flaky(old, new):
        if old == f["e1"]:
            raise PermissionError(13, "Permission denied", str(old))
        rename_noreplace(old, new)

    res = apply_plan(plan, db, command="c", args="", confirm=lambda p: True, rename=flaky)
    why = {m.old.name: w for m, w in res.failed}
    assert "Permission denied" in why["t00.mkv"]
    assert why["t00.en.srt"] == "its video was not moved"
    assert why["t01.mkv"] == "the file is gone"
    assert "its primary t00.mkv was not moved" in why["t02.mkv"]  # no _dupe_1 beside t00
    assert [m.old for m in res.done] == [f["e2"]]
    assert f["e1"].exists() and (s / "Disc 1" / "t00.en.srt").exists() and f["d"].exists()
    assert statuses(db) == ["done"]  # the failed move's journal row was dropped


def test_recover_reconciles_planned_rows(tmp_path):
    db = StateDb(tmp_path / "state.db")
    h = History(db)
    run = h.begin_run("c", "")
    moved = touch(tmp_path / "new1.mkv")
    stayed = touch(tmp_path / "old2.mkv")
    both_a, both_b = touch(tmp_path / "old4.mkv"), touch(tmp_path / "new4.mkv")
    other = touch(tmp_path / "new5.mkv", b"a different file")
    from showvenist.state.fingerprint import fingerprint

    rows = [
        (fingerprint(moved), tmp_path / "old1.mkv", moved),  # rename happened
        (fingerprint(stayed), stayed, tmp_path / "new2.mkv"),  # rename never happened
        ("fp3", tmp_path / "old3.mkv", tmp_path / "new3.mkv"),  # neither exists
        (fingerprint(both_a), both_a, both_b),  # both exist: never guessed
        ("fp-original", tmp_path / "old5.mkv", other),  # another file sits at the target
    ]
    for fp, old, new in rows:
        db.conn.execute(
            "insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status)"
            " values (?,?,?,?,?,'planned')", (run.id, fp, "episode", str(old), str(new)))
    db.conn.commit()
    before = snapshot(tmp_path)
    rep = recover(db)
    assert [Path(o.new_path).name for o in rep.completed] == ["new1.mkv"]
    assert [Path(o.old_path).name for o in rep.rolled_back] == ["old2.mkv"]
    assert [Path(o.new_path).name for o in rep.unresolved] == ["new3.mkv", "new4.mkv",
                                                               "new5.mkv"]
    assert [tuple(r) for r in db.conn.execute(
        "select new_path, status, applied_at is not null from rename_op order by id")] == [
        (str(moved), "done", 1), (str(tmp_path / "new3.mkv"), "planned", 0),
        (str(both_b), "planned", 0), (str(other), "planned", 0)]
    def media(snap):
        return {k: v for k, v in snap[0].items() if not k.startswith("state.db")}

    assert media(snapshot(tmp_path)) == media(before)  # recovery never touches files


def test_undo_round_trip(tmp_path):
    s, f, rows = build(tmp_path)
    files0, dirs0 = snapshot(s)
    db = StateDb(tmp_path / "state.db")
    res = apply_plan(plan_for(*rows), db, command="c", args="", confirm=lambda p: True)
    shown = []
    u = undo_run(db, None, confirm=lambda o: shown.append(o) or True)
    assert u.run_id == res.run_id and not u.skipped and len(u.undone) == 5
    # reverse order, each video before its sidecar (shown in the order it happens)
    assert [o.kind for o in shown[0]] == ["duplicate", "extra", "episode", "episode", "sidecar"]
    assert [o.kind for o in u.undone] == ["duplicate", "extra", "episode", "episode", "sidecar"]
    files1, dirs1 = snapshot(s)
    assert files1 == files0  # every file back where it was, byte for byte
    # the extras/ folder the run created is empty again: undo removes it (sv-g7y)
    assert dirs1 == dirs0 and "extras" not in dirs1
    assert u.removed_dirs == [s / "extras"]
    assert statuses(db) == ["undone"] * 5
    assert {r[0] for r in db.conn.execute("select status from match")} == {"undone"}
    assert [r[0] for r in db.conn.execute("select command from run order by id")] == ["c", "undo"]
    assert undo_run(db, None, confirm=lambda o: True).run_id is None  # nothing left
    # the dupe index is never reused after an undo
    assert History(db).highest_dupe_index(SLUG, "tmdb:100") == 1
    again = plan_for(*rows, history=History(db))
    assert [m.dupe_index for m in again.moves if m.kind == "duplicate"] == [2]


def test_undo_declined_and_skips(tmp_path):
    s, f, rows = build(tmp_path)
    db = StateDb(tmp_path / "state.db")
    res = apply_plan(plan_for(*rows), db, command="c", args="", confirm=lambda p: True)
    snap = snapshot(s)
    assert undo_run(db, res.run_id, confirm=lambda o: False).cancelled
    assert snapshot(s) == snap and statuses(db) == ["done"] * 5
    # the user moved one file away and put something at another's old path
    (s / "synth_show_s01e02.mkv").rename(tmp_path / "elsewhere.mkv")
    blocker = touch(f["x"], b"new rip at the old path")
    u = undo_run(db, res.run_id, confirm=lambda o: True)
    why = {Path(o.new_path).name: w for o, w in u.skipped}
    assert set(why) == {"synth_show_s01e02.mkv", "Disc 2__t01.mkv"}
    assert "no longer at" in why["synth_show_s01e02.mkv"] and "occupied" in why["Disc 2__t01.mkv"]
    assert blocker.read_bytes() == b"new rip at the old path"
    assert (s / "extras" / "Disc 2__t01.mkv").exists()
    assert f["e1"].exists() and f["d"].exists() and (s / "Disc 1" / "t00.en.srt").exists()
    assert statuses(db) == ["undone", "undone", "done", "done", "undone"]
    assert (tmp_path / "elsewhere.mkv").exists()


def test_cross_filesystem_moves_need_their_own_confirmation(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows[:3], device=lambda p: 2 if "Disc 2" in str(p) else 1)
    cross = [m.old.name for m in plan.cross_filesystem]
    assert plan.ok and cross == ["t00.mkv", "t01.mkv"]  # Disc 2's episode and extra
    db = StateDb(tmp_path / "state.db")
    res = apply_plan(plan, db, command="c", args="", confirm=lambda p: True)
    assert {m.old for m, w in res.failed} == {f["e2"], f["x"]}
    assert all("not confirmed" in w for _, w in res.failed)
    assert f["e2"].exists() and f["x"].exists() and not f["e1"].exists()
    plan = plan_for(*rows[1:3], device=lambda p: 2 if "Disc 2" in str(p) else 1)
    used = []

    def across(old, new):
        used.append(old)
        copy_move(old, new)

    res = apply_plan(plan, db, command="c", args="", confirm=lambda p: True,
                     confirm_cross_fs=lambda ms: True, move_across=across)
    assert not res.failed and used == [f["e2"], f["x"]]
    assert (s / "synth_show_s01e02.mkv").exists() and not f["e2"].exists()


CHILD = textwrap.dedent("""
    import os, sys
    from pathlib import Path
    sys.path.insert(0, sys.argv[4])
    from showvenist.rename.movelog import apply_plan
    from showvenist.state.db import StateDb
    from tests.test_rename_apply import build
    from tests.test_rename_plan import plan_for

    tmp, variant, crash_at = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    s, f, rows = build(tmp)
    plan = plan_for(*rows)
    calls = []

    def dying_rename(old, new):
        calls.append(old)
        if len(calls) == crash_at:
            if variant == "after":
                os.rename(old, new)
            os._exit(1)
        os.rename(old, new)

    apply_plan(plan, StateDb(tmp / "state.db"), command="crash", args=variant,
               confirm=lambda p: True, rename=dying_rename)
    os._exit(0)
""")


@pytest.mark.parametrize("variant", ["after", "before"])
def test_real_crash_between_journal_and_rename_is_recovered(tmp_path, variant):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    work = tmp_path / "w"
    work.mkdir()
    env = {**os.environ, "XDG_DATA_HOME": str(tmp_path / "xdg")}
    r = subprocess.run([sys.executable, str(script), str(work), variant, "3", str(REPO)],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, r.stderr  # died inside the third rename
    db = StateDb(work / "state.db")
    # moves 1-2 are done (episode + its sidecar); the third (the pinned episode) is planned
    assert statuses(db) == ["done", "done", "planned"]
    third = db.conn.execute("select old_path, new_path from rename_op where id=3").fetchone()
    old, new = Path(third[0]), Path(third[1])
    assert old.exists() == (variant == "before") and new.exists() == (variant == "after")
    assert db.conn.execute("select count(*) from match").fetchone()[0] == 1  # only move 1's

    rep = recover(db)
    if variant == "after":
        assert [o.id for o in rep.completed] == [3] and not rep.rolled_back
        assert statuses(db) == ["done", "done", "done"]
    else:
        assert [o.id for o in rep.rolled_back] == [3] and not rep.completed
        assert statuses(db) == ["done", "done"]
    assert not rep.unresolved
    # consistency: every done row's file is at its new path, never at its old one
    for o, n, st in db.conn.execute("select old_path, new_path, status from rename_op"):
        assert st == "done" and Path(n).exists() and not Path(o).exists()
    # ... and undo brings back the original layout
    u = undo_run(db, None, confirm=lambda o: True)
    assert not u.skipped
    s = work / "Synth Show" / "Season 1"
    assert sorted(p.relative_to(s).as_posix() for p in s.rglob("*") if p.is_file()) == [
        "Disc 1/t00.en.srt", "Disc 1/t00.mkv", "Disc 2/t00.mkv", "Disc 2/t01.mkv",
        "Disc 2/t02.mkv"]


def test_copy_move_keeps_the_original_when_the_copy_does_not_match(tmp_path, monkeypatch):
    from showvenist.rename import movelog

    old, new = tmp_path / "a.mkv", tmp_path / "sub" / "b.mkv"
    new.parent.mkdir()
    old.write_bytes(b"x" * 300_000)
    real = movelog.shutil.copy2

    def bad_copy(src, dst, **kw):  # a copy that comes out different (same size)
        real(src, dst, **kw)
        with open(dst, "r+b") as fh:
            fh.write(b"corrupt")

    monkeypatch.setattr(movelog.shutil, "copy2", bad_copy)
    with pytest.raises(OSError):
        movelog.copy_move(old, new)
    assert old.exists() and old.stat().st_size == 300_000
    assert not new.exists() and not list(new.parent.iterdir())
