"""Review fixes to M5 renaming (spec §7.2, §13.2-13.5): duplicates follow their primary, one run
per invocation, re-fingerprinting before a move, undo order and `--last`, recovery that
confirms the match, recovery at the start of every moving command, cross-filesystem leftovers.
tmp_path only."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from showvenist import cli
from showvenist.rename.movelog import apply_plan, partial_path, recover, rename_noreplace
from showvenist.rename.types import PlanOptions
from showvenist.rename.undo import latest_run, undo_run
from showvenist.state import migrations
from showvenist.state.db import StateDb, state_db_path
from showvenist.state.fingerprint import fingerprint
from showvenist.state.history import History
from tests.test_rename_apply import build, statuses
from tests.test_rename_cli import AFTER, files_under, identify, two_discs  # noqa: F401
from tests.test_rename_plan import SLUG, plan_for, row, snapshot, touch

runner = CliRunner()


class Crash(BaseException):
    """The process dies right after the rename (not caught by apply_plan's OSError handler)."""


def crash_after(target_name: str):
    def rename(old, new):
        rename_noreplace(old, new)
        if new.name == target_name:
            raise Crash
    return rename


# --- 1. duplicates only follow a primary that is planned and moved --------------------------


def test_copies_of_an_unticked_or_out_of_scope_primary_are_left_alone(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows, opts=PlanOptions(exclude={f["e1"]}))
    assert "duplicate" not in [m.kind for m in plan.moves]
    why = {k.path: k for k in plan.skipped}
    assert why[f["e1"]].reason == "excluded"
    assert why[f["d"]].reason == "duplicate_primary_missing"
    assert "not moved in this plan" in why[f["d"]].detail
    # out of scope only at the target stage: the episode is not in this season folder
    other = tmp_path / "Synth Show" / "Season 2"
    p = touch(other / "Disc 1" / "t00.mkv")
    d = touch(other / "Disc 1" / "t01.mkv", p.read_bytes()[:-1] + b"?")
    plan = plan_for(row(p, 1), row(d, 1, dup_of=p, rank=0, flags=["duplicate"]))
    assert plan.moves == []
    assert [k.reason for k in plan.skipped] == ["out_of_scope", "duplicate_primary_missing"]


def test_duplicate_points_at_its_primary_move(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows)
    dup = next(m for m in plan.moves if m.kind == "duplicate")
    assert plan.moves[dup.primary].old == f["e1"]


def test_apply_skips_a_copy_whose_primary_is_not_moved(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows)
    db = StateDb(tmp_path / "state.db")

    def refuse_e1(old, new):
        if old == f["e1"]:
            raise PermissionError(13, "Permission denied", str(old))
        rename_noreplace(old, new)

    res = apply_plan(plan, db, command="c", args="", confirm=lambda p: True, rename=refuse_e1)
    why = {m.old: w for m, w in res.failed}
    assert "its primary t00.mkv was not moved" in why[f["d"]]
    assert f["d"].exists() and not list(s.rglob("*_dupe_*"))


# --- 2. one run per invocation ----------------------------------------------------------------


def test_several_plans_under_one_run_are_undone_together(tmp_path):
    s, f, rows = build(tmp_path)
    before = snapshot(s)
    db = StateDb(tmp_path / "state.db")
    run = History(db).begin_run("identify --apply", "both seasons")
    p1, p2 = plan_for(rows[0]), plan_for(rows[1])
    for p in (p1, p2):
        res = apply_plan(p, db, command="identify --apply", args="", confirm=lambda p: True,
                         run_id=run.id)
        assert res.run_id == run.id and not res.failed
    assert db.conn.execute("select count(*) from run").fetchone()[0] == 1
    assert {r[0] for r in db.conn.execute("select run_id from rename_op")} == {run.id}
    u = undo_run(db, None, confirm=lambda o: True)
    assert u.run_id == run.id and len(u.undone) == 3 and not u.skipped
    assert snapshot(s) == before


def test_cli_apply_records_one_run(two_discs, monkeypatch):  # noqa: F811
    root, show, s = two_discs
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    r = identify(root, show, "--apply", input="yes\n")
    assert r.exit_code == 0, r.output
    with StateDb.open_default() as db:
        assert [tuple(x) for x in db.conn.execute("select id, command from run")] == [
            (1, "identify --apply")]
        assert {x[0] for x in db.conn.execute("select run_id from rename_op")} == {1}


# --- 4. re-fingerprint right before each move ------------------------------------------------


def test_a_file_changed_since_planning_is_not_moved(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows)
    f["e1"].write_bytes(f["e1"].read_bytes() + b"remuxed since")  # changed after planning
    db = StateDb(tmp_path / "state.db")
    res = apply_plan(plan, db, command="c", args="", confirm=lambda p: True)
    why = {m.old.name if m.old != f["e1"] else "e1": w for m, w in res.failed}
    assert "fingerprint differs" in why["e1"]
    assert why["t00.en.srt"] == "its video was not moved"
    assert "its primary" in why["t02.mkv"]
    assert f["e1"].exists() and (s / "Disc 1" / "t00.en.srt").exists() and f["d"].exists()
    assert str(f["e1"]) not in {r[0] for r in db.conn.execute("select old_path from rename_op")}


def test_a_changed_cross_filesystem_source_is_not_copied(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(rows[1], device=lambda p: 2 if "Disc 2" in str(p) else 1)
    assert plan.cross_filesystem
    f["e2"].write_bytes(b"changed")
    used = []
    res = apply_plan(plan, StateDb(tmp_path / "state.db"), command="c", args="",
                     confirm=lambda p: True, confirm_cross_fs=lambda ms: True,
                     move_across=lambda o, n: used.append(o))
    assert used == [] and "fingerprint differs" in res.failed[0][1]


# --- 5. undo order, sidecars already back, --last ---------------------------------------------


def test_undo_sidecar_already_back_and_sidecar_stays_with_its_video(tmp_path):
    s, f, rows = build(tmp_path)
    db = StateDb(tmp_path / "state.db")
    res = apply_plan(plan_for(*rows), db, command="c", args="", confirm=lambda p: True)
    (s / "synth_show_s01e01.en.srt").rename(s / "Disc 1" / "t00.en.srt")  # moved back by hand
    u = undo_run(db, res.run_id, confirm=lambda o: True)
    assert not u.skipped and statuses(db) == ["undone"] * 5

    s2 = tmp_path / "b"
    s, f, rows = build(s2)
    db = StateDb(s2 / "state.db")
    res = apply_plan(plan_for(*rows), db, command="c", args="", confirm=lambda p: True)
    touch(f["e1"], b"a new rip where the video was")  # the video cannot go back
    u = undo_run(db, res.run_id, confirm=lambda o: True)
    why = {Path(o.new_path).name: w for o, w in u.skipped}
    assert "occupied" in why["synth_show_s01e01.mkv"]
    assert "stays with the video" in why["synth_show_s01e01.en.srt"]
    assert (s / "synth_show_s01e01.en.srt").exists()


def test_last_moves_on_after_a_partial_undo(tmp_path):
    s, f, rows = build(tmp_path)
    db = StateDb(tmp_path / "state.db")
    r1 = apply_plan(plan_for(rows[0]), db, command="c", args="", confirm=lambda p: True)
    r2 = apply_plan(plan_for(rows[1]), db, command="c", args="", confirm=lambda p: True)
    blocker = touch(f["e2"], b"in the way")
    u = undo_run(db, None, confirm=lambda o: True)
    assert u.run_id == r2.run_id and len(u.skipped) == 1
    assert latest_run(db) == r1.run_id  # not run 2 again, forever
    u = undo_run(db, None, confirm=lambda o: True)
    assert u.run_id == r1.run_id and not u.skipped
    assert latest_run(db) is None
    blocker.unlink()  # naming the run still retries its leftovers
    u = undo_run(db, r2.run_id, confirm=lambda o: True)
    assert len(u.undone) == 1 and f["e2"].exists()


# --- 6. schema v2, recovery confirms the match ------------------------------------------------


def test_upgrade_from_schema_1_keeps_rows(tmp_path):
    path = tmp_path / "state.db"
    c = sqlite3.connect(path)
    c.executescript(migrations.MIGRATIONS[0])
    c.execute("PRAGMA user_version = 1")
    c.execute("insert into run (started_at, command, args) values ('t', 'c', '')")
    c.execute("insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status)"
              " values (1, 'fp', 'episode', '/a', '/b', 'done')")
    c.commit()
    c.close()
    db = StateDb(path)
    assert db.migrated_from == 1
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(rename_op)")}
    assert {"season", "episode", "title", "match_source", "confidence", "bundle_revision",
            "disc_label", "prior_match", "cross_fs"} <= cols
    r = db.conn.execute("select old_path, new_path, status, season from rename_op").fetchone()
    assert tuple(r) == ("/a", "/b", "done", None)


def test_recovery_confirms_the_match_from_the_journal(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(*rows)
    db = StateDb(tmp_path / "state.db")
    with pytest.raises(Crash):  # dies after renaming the extra, before logging it
        apply_plan(plan, db, command="c", args="", confirm=lambda p: True,
                   rename=crash_after("Disc 2__t01.mkv"))
    assert statuses(db)[-1] == "planned"
    rep = recover(db)
    assert len(rep.completed) == 1 and not rep.unconfirmed
    op = db.conn.execute("select status, match_id from rename_op order by id desc").fetchone()
    assert op[0] == "done" and op[1] is not None
    assert History(db).is_confirmed_extra(fingerprint(s / "extras" / "Disc 2__t01.mkv"))


def test_recovered_episode_and_its_sidecar(tmp_path):
    s, f, rows = build(tmp_path)
    plan = plan_for(rows[0])  # episode 1 and its sidecar
    db = StateDb(tmp_path / "state.db")
    with pytest.raises(Crash):  # dies after the sidecar's rename, before logging it
        apply_plan(plan, db, command="c", args="", confirm=lambda p: True,
                   rename=crash_after("synth_show_s01e01.en.srt"))
    assert statuses(db) == ["done", "planned"]
    rep = recover(db)
    assert [o.kind for o in rep.completed] == ["sidecar"]
    sc, video = db.conn.execute(
        "select a.match_id, b.match_id from rename_op a, rename_op b where a.kind='sidecar'"
        " and b.kind='episode' and b.fingerprint=a.fingerprint").fetchone()
    assert sc == video is not None
    # an episode: crash after its own rename
    s2 = tmp_path / "b"
    s, f, rows = build(s2)
    db = StateDb(s2 / "state.db")
    with pytest.raises(Crash):
        apply_plan(plan_for(rows[1]), db, command="c", args="", confirm=lambda p: True,
                   rename=crash_after("synth_show_s01e02.mkv"))
    assert db.conn.execute("select count(*) from match").fetchone()[0] == 0
    recover(db)
    m = History(db).confirmed(SLUG)
    assert [(x.episode_id, x.season, x.episode, x.title, x.source, x.confidence)
            for x in m] == [("tmdb:101", 1, 2, "Episode Title 2", "manual", "medium")]
    mf = History(db).get_file(m[0].fingerprint)
    assert mf.last_path == str(s / "synth_show_s01e02.mkv") and mf.disc_label == "Disc 2"


def test_recovery_of_a_pre_v2_row_writes_no_match(tmp_path):
    db = StateDb(tmp_path / "state.db")
    run = History(db).begin_run("c", "")
    new = touch(tmp_path / "new.mkv")
    db.conn.execute("insert into rename_op (run_id, fingerprint, kind, old_path, new_path,"
                    " status, show_slug, episode_id) values (?,?,?,?,?,'planned',?,?)",
                    (run.id, fingerprint(new), "episode", str(tmp_path / "old.mkv"), str(new),
                     SLUG, "tmdb:100"))
    db.conn.commit()
    rep = recover(db)
    assert len(rep.completed) == 1 and len(rep.unconfirmed) == 1
    assert "match was not recorded" in rep.notes[0]
    assert db.conn.execute("select count(*) from match").fetchone()[0] == 0


# --- 7. cross-filesystem leftovers; recovery at the start of every moving command -----------


def _planned(db, old, new, *, cross: bool, fp: str | None = None, kind: str = "episode"):
    run = History(db).begin_run("c", "")
    db.conn.execute(
        "insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status,"
        " show_slug, episode_id, season, episode, title, match_source, confidence, cross_fs)"
        " values (?,?,?,?,?,'planned',?,'tmdb:100',1,1,'T','auto','high',?)",
        (run.id, fp or fingerprint(old), kind, str(old), str(new), SLUG, int(cross)))
    db.conn.commit()


def test_recovery_removes_our_partial_copy_only_while_the_original_exists(tmp_path):
    db = StateDb(tmp_path / "state.db")
    old = touch(tmp_path / "a" / "old.mkv")
    new = tmp_path / "b" / "new.mkv"
    part = touch(partial_path(new), old.read_bytes()[:10])
    _planned(db, old, new, cross=True)
    rep = recover(db)
    assert rep.removed_partials == [part] and not part.exists() and old.exists()
    assert len(rep.rolled_back) == 1
    # the original is gone: keep the partial, report it, never guess
    gone = tmp_path / "a" / "gone.mkv"
    new2 = tmp_path / "b" / "new2.mkv"
    part2 = touch(partial_path(new2), b"partial")
    _planned(db, gone, new2, cross=True, fp="fp-gone")
    rep = recover(db)
    assert part2.exists() and not rep.removed_partials and len(rep.unresolved) == 1
    assert any("partial copy kept" in n for n in rep.notes)


def test_recovery_finishes_a_verified_cross_filesystem_copy(tmp_path):
    db = StateDb(tmp_path / "state.db")
    old = touch(tmp_path / "a" / "old.mkv")
    new = touch(tmp_path / "b" / "new.mkv", old.read_bytes())
    _planned(db, old, new, cross=True)
    rep = recover(db)
    assert len(rep.completed) == 1 and not old.exists() and new.exists()
    assert History(db).confirmed(SLUG)[0].episode_id == "tmdb:100"
    assert "finished an interrupted cross-filesystem move" in rep.notes[0]


@pytest.mark.parametrize("cross,content", [(True, b"a different file"), (False, None)])
def test_recovery_leaves_both_paths_when_not_verifiable(tmp_path, cross, content):
    db = StateDb(tmp_path / "state.db")
    old = touch(tmp_path / "a" / "old.mkv")
    new = touch(tmp_path / "b" / "new.mkv", content if content is not None else old.read_bytes())
    _planned(db, old, new, cross=cross)
    before = snapshot(tmp_path / "a"), snapshot(tmp_path / "b")
    rep = recover(db)
    assert len(rep.unresolved) == 1 and "both paths exist" in rep.notes[0]
    assert (snapshot(tmp_path / "a"), snapshot(tmp_path / "b")) == before
    assert statuses(db) == ["planned"]


def _leftover(tmp_path) -> Path:
    """A planned row whose rename happened (recovery completes it)."""
    db = StateDb(state_db_path())
    new = touch(tmp_path / "elsewhere" / "moved.mkv")
    _planned(db, tmp_path / "elsewhere" / "orig.mkv", new, cross=False, fp=fingerprint(new))
    db.close()
    return new


def _leftover_status() -> list[str]:
    with StateDb.open_default() as db:
        return statuses(db)


def test_identify_apply_recovers_even_with_nothing_to_move(two_discs, monkeypatch):  # noqa: F811
    root, show, s = two_discs
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    assert identify(root, show, "--apply", input="yes\n").exit_code == 0
    assert files_under(s) == AFTER
    _leftover(show.parent)
    r = identify(root, show, "--apply", input="yes\n")
    assert "Nothing to move." in r.stdout and "Recovered from an earlier" in r.output
    assert _leftover_status()[-1] == "done"
    _leftover(show.parent / "x")
    r = identify(root, show, "--apply", "--confirm", "0000-0000")
    assert "Recovered from an earlier" in r.output
    assert _leftover_status()[-1] == "done"


def test_undo_recovers_even_with_nothing_to_undo(tmp_path, monkeypatch):
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    _leftover(tmp_path)
    with StateDb.open_default() as db:  # pretend the leftover's run was undone already
        db.conn.execute("insert into run (started_at, command, args) values ('t','undo','1')")
        db.conn.commit()
    r = runner.invoke(cli.app, ["undo", "--last", "--yes"])
    assert r.exit_code == 1 and "nothing to undo" in r.output
    assert "Recovered from an earlier" in r.output
    assert _leftover_status() == ["done"]


def test_assign_rename_recovers_first(tmp_path, monkeypatch):
    from tests.test_cli_identify_eval import setup as flat_setup

    root, rips = flat_setup(tmp_path, monkeypatch)
    _leftover(tmp_path)
    r = runner.invoke(cli.app, ["assign", str(rips / "t01.mkv"), "S01E02", "--show", SLUG,
                                "-o", str(root), "--rename"], input="n\n")
    assert r.exit_code == 1 and "Recovered from an earlier" in r.output
    assert _leftover_status() == ["done"]
