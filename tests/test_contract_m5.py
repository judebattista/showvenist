"""M5 contract (spec §7.2, §13, decisions 56-59; tasks sv-o4f.1-.3)."""

import sqlite3
from pathlib import Path

import pytest

from showvenist.config import load_config
from showvenist.hierarchy.parse import parse_named
from showvenist.rename import naming
from showvenist.rename.movelog import apply_plan, match_args, recover
from showvenist.rename.plan import build_plan
from showvenist.rename.sidecars import find_sidecars
from showvenist.rename.types import Plan, PlannedMove
from showvenist.rename.undo import undo_run
from showvenist.report_schema import FileReport
from showvenist.state.db import StateDb
from showvenist.state.history import History
from showvenist.state.migrations import MIGRATIONS, SCHEMA_VERSION, SchemaTooNew, migrate


def db() -> StateDb:
    return StateDb(":memory:")


def test_schema_created_and_versioned(tmp_path):
    d = StateDb(tmp_path / "sub" / "state.db")
    assert d.migrated_from == 0
    v = d.conn.execute("PRAGMA user_version").fetchone()[0]
    assert v == SCHEMA_VERSION == len(MIGRATIONS)
    tables = {r[0] for r in d.conn.execute("select name from sqlite_master where type='table'")}
    assert {"media_file", "run", "match", "rename_op"} <= tables
    d.close()
    again = StateDb(tmp_path / "sub" / "state.db")  # reopening migrates nothing
    assert again.migrated_from == SCHEMA_VERSION
    assert again.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_newer_database_is_refused(tmp_path):
    p = tmp_path / "x.db"
    c = sqlite3.connect(p)
    c.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    c.commit()
    with pytest.raises(SchemaTooNew):
        migrate(c)


def _file(d, fp="1:a"):
    d.conn.execute("insert into media_file values (?,?,?,?,?,?,?,?,?)",
                   (fp, None, 1, "/x", "s", None, None, "t", "t"))


def test_match_status_and_extra_constraints():
    d = db()
    _file(d)
    ins = ("insert into match (fingerprint, show_slug, episode_id, season, episode, status,"
           " source, created_at) values (?,?,?,?,?,?,?,?)")
    d.conn.execute(ins, ("1:a", "s", "tmdb:1", 1, 2, "confirmed", "auto", "t"))
    d.conn.execute(ins, ("1:a", "s", None, None, None, "confirmed", "auto", "t"))  # an extra
    for bad in (("proposed", "auto"), ("confirmed", "guess")):
        with pytest.raises(sqlite3.IntegrityError):
            d.conn.execute(ins, ("1:a", "s", "tmdb:1", 1, 2, *bad, "t"))
    with pytest.raises(sqlite3.IntegrityError):  # an extra carries no season/episode
        d.conn.execute(ins, ("1:a", "s", None, 1, 2, "confirmed", "auto", "t"))
    for status in ("rejected", "undone"):
        d.conn.execute(ins, ("1:a", "s", "tmdb:1", 1, 2, status, "auto", "t"))


def test_rename_op_vocabulary():
    d = db()
    d.conn.execute("insert into run (started_at, command, args) values ('t','apply','')")
    ins = ("insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status)"
           " values (1,'1:a',?,?,?,?)")
    for kind in ("episode", "extra", "duplicate", "sidecar"):
        d.conn.execute(ins, (kind, "/a", "/b", "planned"))
    for bad in (("move", "planned"), ("episode", "pending")):
        with pytest.raises(sqlite3.IntegrityError):
            d.conn.execute(ins, (bad[0], "/a", "/b", bad[1]))


def test_history_api_is_implemented():
    h = History(db())
    assert h.begin_run("apply", "").id == 1
    assert h.highest_dupe_index("s", "tmdb:1") == 0


def test_rename_functions_are_stubs_until_sv_o4f_3(tmp_path):
    plan = Plan(show_slug="s", naming_order="aired")
    assert plan.ok
    plan.errors.append("no write permission")
    assert not plan.ok
    with pytest.raises(NotImplementedError):
        find_sidecars(tmp_path / "a.mkv")
    with pytest.raises(NotImplementedError):
        recover(db())
    with pytest.raises(NotImplementedError):
        apply_plan(plan, db(), command="identify", args="", confirm=lambda p: True)
    with pytest.raises(NotImplementedError):
        undo_run(db(), None, confirm=lambda ops: True)
    with pytest.raises(NotImplementedError):
        build_plan(None, None, None)  # type: ignore[arg-type]


def test_match_args_follow_the_move():
    m = PlannedMove("episode", Path("a"), Path("b"), "1:a", "s", episode_id="tmdb:1",
                    season=1, episode=2, title="T", pinned=True)
    assert match_args(m) == dict(episode_id="tmdb:1", season=1, episode=2, title="T",
                                 source="manual")
    assert match_args(PlannedMove("extra", Path("a"), Path("b"), "1:a", "s"))["source"] == "auto"


def test_naming_helpers():
    assert naming.snake("The West Wing") == "the_west_wing"
    assert naming.snake("Stargate SG-1 (1997)") == "stargate_sg_1_1997"
    assert naming.sanitize('a:b/c?  "d". ') == "a_b_c_ _d_"
    stem = naming.episode_stem("{show_snake}_s{s:02}e{e:02}", "The West Wing", 1, 2)
    assert stem == "the_west_wing_s01e02"
    assert naming.episode_stem(
        "{show} - {s}x{e} - {title}", "Firefly", 1, 5, "Our Mrs. Reynolds"
    ) == "Firefly - 1x5 - Our Mrs. Reynolds"
    assert naming.dupe_stem(stem, 3) == "the_west_wing_s01e02_dupe_3"
    assert naming.parse_dupe("the_west_wing_s01e02_dupe_12.mkv") == 12
    assert naming.parse_dupe("the_west_wing_s01e02.mkv") is None
    assert naming.extra_name("disc2", "Babylon 5 - Season 2_t00.mkv") == (
        "disc2__Babylon 5 - Season 2_t00.mkv")
    assert naming.extra_name(None, "t00.mkv") == "t00.mkv"
    assert naming.with_collision_index(Path("/x/a.mkv"), 2) == Path("/x/a_2.mkv")
    assert naming.sidecar_target(
        Path("/d/title_t00.mkv"), Path("/d/title_t00.en.srt"), Path("/s/show_s01e02.mkv")
    ) == Path("/s/show_s01e02.en.srt")


def test_parser_records_dupe_index_and_still_sees_an_episode():
    n = parse_named("the_west_wing_s01e02_dupe_1.mkv")
    assert (n.season, n.episode, n.dupe) == (1, 2, 1)
    assert parse_named("the_west_wing_s01e02.mkv").dupe is None
    assert parse_named("disc2__Babylon 5 - Season 2_t00.mkv") is None  # a prefixed extra


def test_report_gains_duplicate_fields_without_a_version_bump():
    f = FileReport.model_fields
    assert f["duplicate_of"].default is None and f["duplicate_rank"].default is None


def test_history_penalty_default_is_ln10():
    cfg = load_config(None, env={})
    assert abs(cfg.history.penalty - 2.302585092994046) < 1e-12
