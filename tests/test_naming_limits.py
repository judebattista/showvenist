"""Sidecar claim, readonly open, same-target confirm across runs, name limits."""

import sqlite3

import pytest

from showvenist.rename import naming
from showvenist.rename.sidecars import find_sidecars
from showvenist.state.db import SchemaOutdated, StateDb, StateDbMissing
from showvenist.state.history import History


def test_sidecar_of_longer_video_stem_is_not_claimed(tmp_path):
    for n in ["A.mkv", "A.part2.mkv", "A.en.srt", "A.part2.en.srt", "A.nfo"]:
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in find_sidecars(tmp_path / "A.mkv")] == ["A.en.srt", "A.nfo"]
    assert [p.name for p in find_sidecars(tmp_path / "A.part2.mkv")] == ["A.part2.en.srt"]


def test_readonly_open(tmp_path):
    path = tmp_path / "state.db"
    with pytest.raises(StateDbMissing):
        StateDb.open_readonly(path)
    assert not path.exists()
    StateDb(path).close()
    db = StateDb.open_readonly(path)
    with pytest.raises(sqlite3.OperationalError):
        db.conn.execute("create table x (a)")
    db.close()
    c = sqlite3.connect(path)
    c.execute("PRAGMA user_version = 0")
    c.commit()
    c.close()
    with pytest.raises(SchemaOutdated):
        StateDb.open_readonly(path)
    assert sqlite3.connect(path).execute("PRAGMA user_version").fetchone()[0] == 0


def test_same_target_across_runs_keeps_each_runs_decisions():
    h = History(StateDb(":memory:"))
    r1, r2 = h.begin_run("apply", ""), h.begin_run("apply", "")
    h.touch_file("1:a", size=1, path="/a.mkv")
    kw = dict(episode_id="e1", season=1, episode=1, title="T", source="auto")
    m1 = h.confirm("1:a", "s", run_id=r1.id, **kw)
    m2 = h.confirm("1:a", "s", run_id=r2.id, **kw)
    assert m2.id == m1.id and m2.run_id == r1.id
    assert len(h.confirmed("s")) == 1
    assert h.mark_undone(r2.id) == 0 and len(h.confirmed("s")) == 1
    assert h.mark_undone(r1.id) == 1


def test_snake_transliterates_and_drops_empty():
    assert naming.snake("Amélie") == "amelie"
    assert naming.snake("進撃の巨人") == ""
    assert naming.episode_stem("{show_snake}_s{s:02}e{e:02}", "進撃の巨人", 1, 2) == "s01e02"
    assert naming.episode_stem("{show_snake}_s{s:02}e{e:02}", "The West Wing", 1, 2) == \
        "the_west_wing_s01e02"


def test_limit_name_keeps_extension_and_suffix():
    name = "é" * 200 + "_dupe_3.mkv"
    out = naming.limit_name(name)
    assert len(out.encode()) <= 255 and out.endswith("_dupe_3.mkv")
    assert naming.limit_name("a.mkv") == "a.mkv"
    p = naming.with_collision_index(__import__("pathlib").Path("x/" + "b" * 255 + ".mkv"), 2)
    assert len(p.name.encode()) <= 255 and p.name.endswith("_2.mkv")
    long = naming.episode_stem("{show_snake}_s{s:02}e{e:02}_{title}", "Show", 1, 2, "世" * 200)
    assert len((naming.dupe_stem(long, 12) + ".mkv").encode()) <= 255
