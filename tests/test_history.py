"""History methods (sv-o4f.2) on an in-memory state.db; synthetic fingerprints only."""

import pytest

from showvenist.state.db import StateDb
from showvenist.state.history import History


@pytest.fixture
def h():
    return History(StateDb(":memory:"))


def _file(h, fp, path="/lib/a.mkv"):
    return h.touch_file(fp, size=10, path=path)


def _ep(h, fp, eid, n, run=None, source="auto", show="s"):
    _file(h, fp, f"/lib/{fp}.mkv")
    return h.confirm(fp, show, episode_id=eid, season=1, episode=n, title=f"T{n}",
                     source=source, run_id=run)


def test_begin_run_and_touch_file_upsert(h):
    r1, r2 = h.begin_run("assign", "a"), h.begin_run("apply", "b")
    assert (r1.id, r2.id, r2.command, r2.args) == (1, 2, "apply", "b")
    assert r1.started_at.endswith("+00:00")
    f1 = h.touch_file("1:a", size=1, path="/x/a.mkv", disc_label="D1")
    f2 = h.touch_file("1:a", size=1, path="/y/a.mkv", show_slug="s")
    assert f2.first_seen == f1.first_seen and f2.last_path == "/y/a.mkv"
    assert f2.last_seen >= f1.last_seen
    assert f2.disc_label == "D1" and f2.show_slug == "s"  # None keeps the stored value


def test_confirm_rejects_the_previous_confirmed_match(h):
    _ep(h, "1:a", "e1", 1)
    new = h.confirm("1:a", "s", episode_id="e2", season=1, episode=2, title="T2",
                    source="manual", run_id=None)
    assert [m.episode_id for m in h.confirmed("s")] == ["e2"]
    assert new.status == "confirmed" and new.source == "manual"
    assert [m.episode_id for m in h.rejected("s")] == ["e1"]
    # another show is untouched
    h.confirm("1:a", "other", episode_id="x", season=1, episode=1, title=None,
              source="auto", run_id=None)
    assert len(h.confirmed("s")) == 1 and len(h.confirmed("other")) == 1


def test_reconfirming_the_same_target_rejects_nothing(h):
    _ep(h, "1:a", "e1", 1)
    h.confirm("1:a", "s", episode_id="e1", season=1, episode=1, title="T1",
              source="manual", run_id=None)
    assert len(h.confirmed("s")) == 1 and h.rejected("s") == []
    assert h.confirmed("s")[0].source == "manual"


def test_extras(h):
    _file(h, "1:x")
    m = h.confirm("1:x", "s", episode_id=None, season=None, episode=None, title=None,
                  source="manual", run_id=None)
    assert m.episode_id is None and h.is_confirmed_extra("1:x")
    with pytest.raises(ValueError):
        h.confirm("1:x", "s", episode_id=None, season=1, episode=2, title=None,
                  source="manual", run_id=None)
    # extra -> episode: the extra is rejected
    h.confirm("1:x", "s", episode_id="e1", season=1, episode=1, title=None,
              source="manual", run_id=None)
    assert not h.is_confirmed_extra("1:x")
    assert [m.episode_id for m in h.rejected("s")] == [None]


def test_reads_filter_by_season_and_status(h):
    _ep(h, "1:a", "e1", 1)
    _file(h, "1:b")
    h.confirm("1:b", "s", episode_id="e9", season=2, episode=1, title=None,
              source="auto", run_id=None)
    assert len(h.confirmed("s")) == 2
    assert [m.episode_id for m in h.confirmed("s", season=2)] == ["e9"]
    assert [m.fingerprint for m in h.confirmed_for_episode("s", "e1")] == ["1:a"]
    h.reject("1:a", "s", episode_id="e5", run_id=None, season=1, episode=5)
    assert [m.episode_id for m in h.rejected("s")] == ["e5"]
    assert h.rejected("nobody") == [] and h.shows() == ["s"]


def test_mark_undone_only_touches_the_runs_confirmed_matches(h):
    r1, r2 = h.begin_run("apply", ""), h.begin_run("apply", "")
    _ep(h, "1:a", "e1", 1, run=r1.id)
    _ep(h, "1:b", "e2", 2, run=r1.id)
    _ep(h, "1:c", "e3", 3, run=r2.id)
    h.reject("1:a", "s", episode_id="e7", run_id=r1.id)
    assert h.mark_undone(r1.id) == 2
    assert [m.episode_id for m in h.confirmed("s")] == ["e3"]
    assert len(h.undone("s")) == 2
    assert [m.episode_id for m in h.rejected("s")] == ["e7"]  # not a judgement: rejects stay
    assert h.confirmed_for_episode("s", "e1") == []
    assert h.mark_undone(r1.id) == 0


def test_forget_removes_file_and_matches_but_keeps_the_move_log(h):
    run = h.begin_run("apply", "")
    m = _ep(h, "1:a", "e1", 1, run=run.id)
    _ep(h, "1:b", "e2", 2)
    h.reject("1:a", "s", episode_id="e5", run_id=None)
    h.db.conn.execute(
        "insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status, match_id,"
        " show_slug, episode_id, dupe_index) values (?,?,?,?,?,?,?,?,?,?)",
        (run.id, "1:a", "episode", "/a", "/b", "done", m.id, "s", "e1", 2))
    assert h.forget("1:a") == 2
    assert h.get_file("1:a") is None and h.matches_for_file("1:a") == []
    assert [x.fingerprint for x in h.confirmed("s")] == ["1:b"]
    row = h.db.conn.execute("select old_path, new_path, match_id from rename_op").fetchone()
    assert tuple(row) == ("/a", "/b", None)
    assert h.highest_dupe_index("s", "e1") == 2
    assert h.forget("1:zzz") == 0


def test_history_penalty_matrix(h):
    _ep(h, "1:a", "e1", 1)
    _ep(h, "1:b", "e2", 2)
    _file(h, "1:c")  # a second file for e2
    h.confirm("1:c", "s", episode_id="e2", season=1, episode=2, title=None, source="auto",
              run_id=None)
    h.reject("1:a", "s", episode_id="e3", run_id=None)  # rejected rows never penalize
    got = h.history_penalty("s", ["1:a", "1:b", "1:new"], ["e1", "e2", "e3"], 2.5)
    assert got == [
        [0.0, 2.5, 0.0],  # a: its own e1 is free, e2 is held by others
        [2.5, 2.5, 0.0],  # b: e1 held by a; e2 is also held by c
        [2.5, 2.5, 0.0],  # a new file: everything confirmed is taken
    ]
    assert h.history_penalty("other", ["1:a"], ["e1"], 1.0) == [[0.0]]
    assert h.history_penalty("s", ["1:new"], ["e1"], 1.0) == [[1.0]]


def test_highest_dupe_index(h):
    run = h.begin_run("apply", "")
    ins = ("insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status,"
           " show_slug, episode_id, dupe_index) values (?,?,?,?,?,?,?,?,?)")
    assert h.highest_dupe_index("s", "e1") == 0
    for fp, idx in (("1:a", None), ("1:b", 1), ("1:c", 3)):
        h.db.conn.execute(ins, (run.id, fp, "duplicate" if idx else "episode", "/a", "/b",
                                "planned", "s", "e1", idx))
    h.db.conn.execute(ins, (run.id, "1:d", "duplicate", "/a", "/b", "planned", "s", "e2", 9))
    assert h.highest_dupe_index("s", "e1") == 3


def test_eval_label_naming_an_episode_id():
    from showvenist.eval.harness import resolve_labels
    from showvenist.eval.types import ManifestRow
    from tests.test_session import synth_bundle

    b = synth_bundle()
    rows = [ManifestRow("/l/Season 1/a.mkv", "x", "@tmdb:102"),
            ManifestRow("/l/Season 1/b.mkv", "x", "@tmdb:999")]
    labels, problems = resolve_labels(rows, {"x": b})
    assert [(lab.kind, lab.episode_id) for lab in labels] == [("episode", "tmdb:102")]
    assert "not an episode" in problems[0].reason
