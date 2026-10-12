"""Two-episode files (decision 64): `assign S01E11+S01E12`, schema v3, naming `s01e11-e12`, both
episodes held by the file, and the range through apply, history and undo. Synthetic only."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from showvenist import cli
from showvenist.analysis.types import AnalysisCache
from showvenist.bundle.schema import Ordering, OrderingEntry
from showvenist.config import Config
from showvenist.eval.harness import resolve_labels
from showvenist.eval.history import history_labels
from showvenist.eval.types import ManifestRow
from showvenist.rename import naming
from showvenist.rename.movelog import apply_plan, recover
from showvenist.rename.render import format_ops
from showvenist.rename.types import PlanOptions
from showvenist.rename.undo import done_ops, undo_run
from showvenist.report_schema import EpisodeRef
from showvenist.session import FLAG_CONFIRMED, FLAG_HISTORY_DUPLICATE, Session
from showvenist.state import migrations
from showvenist.state.db import StateDb
from showvenist.state.fingerprint import fingerprint
from showvenist.state.history import (
    History,
    end_episode_id,
    episode_code,
    held_episode_ids,
)
from tests.synth import make_bundle
from tests.test_cli_identify_eval import setup as flat_setup
from tests.test_rename_plan import plan_for, row, season, touch
from tests.test_session import rips, synth_bundle, write_refs  # noqa: F401  (rips fixture)

SHOW = "synth-show-2000"
runner = CliRunner()


def range_row(path, first: int, *, flags=(), **kw):
    """A report row for a file holding episodes `first` and `first + 1` (make_bundle ids)."""
    r = row(path, first, flags=flags, **kw)
    ref = r.decision.episode.model_copy(update={
        "episode_end": first + 1, "episode_end_id": f"tmdb:{100 + first}"})
    return r.model_copy(update={"decision": r.decision.model_copy(update={"episode": ref})})


# --- schema v3 --------------------------------------------------------------------------------


def test_upgrade_from_schema_2_keeps_rows_as_single_episodes(tmp_path):
    path = tmp_path / "state.db"
    c = sqlite3.connect(path)
    c.executescript(migrations.MIGRATIONS[0] + migrations.MIGRATIONS[1])
    c.execute("PRAGMA user_version = 2")
    c.execute("insert into run (started_at, command, args) values ('t', 'c', '')")
    c.execute("insert into media_file values ('fp', null, 1, '/a', 's', null, null, 't', 't')")
    c.execute("insert into match (fingerprint, show_slug, episode_id, season, episode, title,"
              " status, source, run_id, created_at) values ('fp', 's', 'e1', 1, 1, 'T',"
              " 'confirmed', 'auto', 1, 't')")
    c.execute("insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status,"
              " season, episode) values (1, 'fp', 'episode', '/a', '/b', 'done', 1, 1)")
    c.commit()
    c.close()
    db = StateDb(path)
    assert db.migrated_from == 2
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == migrations.SCHEMA_VERSION == 3
    for table in ("match", "rename_op"):
        assert "episode_end" in {r[1] for r in db.conn.execute(f"PRAGMA table_info({table})")}
    (m,) = History(db).confirmed("s")
    assert (m.episode_id, m.episode, m.episode_end) == ("e1", 1, None)
    (op,) = done_ops(db, 1)
    assert (op.season, op.episode, op.episode_end) == (1, 1, None)
    assert "S01E01" in format_ops([op]) and "S01E01-" not in format_ops([op])


# --- history ---------------------------------------------------------------------------------


@pytest.fixture
def h():
    return History(StateDb(":memory:"))


def _range(h, fp, first, end, show="s", run=None):
    h.touch_file(fp, size=1, path=f"/lib/{fp}.mkv")
    return h.confirm(fp, show, episode_id=f"tmdb:{99 + first}", season=1, episode=first,
                     title="T", source="manual", run_id=run, episode_end=end)


def test_history_stores_and_reads_the_range(h):
    b = make_bundle()
    m = _range(h, "1:a", 2, 3)
    assert m.episode_end == 3 and h.confirmed("s")[0].episode_end == 3
    assert end_episode_id(b, m.episode_id, 1, 2, 3) == "tmdb:102"
    assert held_episode_ids(b, m) == ["tmdb:101", "tmdb:102"]
    assert held_episode_ids(None, m) == ["tmdb:101"]  # no bundle: the first episode only
    assert episode_code(1, 2, 3) == "S01E02-E03" and episode_code(None, None) == "extra"
    # the second episode is held by the file: duplicate rule and prior
    assert [x.fingerprint for x in h.confirmed_for_episode("s", "tmdb:102", bundle=b)] == ["1:a"]
    assert h.confirmed_for_episode("s", "tmdb:102") == []
    pen = h.history_penalty("s", ["1:a", "1:b"], ["tmdb:101", "tmdb:102", "tmdb:103"], 2.0,
                            bundle=b)
    assert pen == [[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]
    # a different range is a different decision: the old one becomes rejected
    m2 = _range(h, "1:a", 2, None)
    assert m2.id != m.id and [x.episode_end for x in h.confirmed("s")] == [None]
    assert [x.episode_end for x in h.rejected("s")] == [3]
    with pytest.raises(ValueError):
        _range(h, "1:b", 3, 3)


# --- naming ----------------------------------------------------------------------------------


def test_range_names_for_default_and_custom_templates():
    t = PlanOptions().template
    assert naming.episode_stem(t, "Burn Notice", 1, 11, "Old Friends",
                               episode_end=12) == "burn_notice_s01e11-e12"
    assert naming.episode_stem(t, "Burn Notice", 1, 11) == "burn_notice_s01e11"
    assert naming.range_template("{show} S{s:02}E{e:02} {title}") == \
        "{show} S{s:02}E{e:02}-E{e_end:02} {title}"
    assert naming.episode_stem("{show} - {s}x{e:02} - {title}", "Show", 1, 9, "Ten",
                               episode_end=10) == "Show - 1x09-10 - Ten"
    assert naming.episode_stem("{{e}} s{s}e{e}", "Show", 1, 1, episode_end=2) == "{e} s1e1-e2"
    with pytest.raises(ValueError, match="no \\{e\\}"):
        naming.episode_stem("{show} {title}", "Show", 1, 1, "x", episode_end=2)
    long = naming.episode_stem("{show} s{s:02}e{e:02} {title}", "Show", 1, 1, "y" * 400,
                               episode_end=2)
    assert long.startswith("Show s01e01-e02 y") and len(long.encode()) <= 255 - 12


# --- plan, apply, recovery, undo ---------------------------------------------------------------


def test_plan_names_the_range_and_its_copies(tmp_path):
    s = season(tmp_path)
    a = touch(s / "Disc 1" / "t00.mkv")
    copy = touch(s / "Disc 2" / "t05.mkv", a.read_bytes() + b"x")
    taken = touch(s / "synth_show_s01e02-e03.mkv")  # an existing file with the same name
    plan = plan_for(range_row(a, 2), row(copy, 2, dup_of=a, rank=0))
    assert plan.ok, plan.errors
    m = plan.moves[0]
    assert m.new == s / "synth_show_s01e02-e03_2.mkv" and m.collision_index == 2
    assert (m.season, m.episode, m.episode_end) == (1, 2, 3)
    assert "S01E02-E03 Episode Title 2" in m.reason
    d = plan.moves[1]
    assert d.kind == "duplicate" and d.new == copy.parent / "synth_show_s01e02-e03_dupe_1.mkv"
    assert d.episode_end == 3 and "(S01E02-E03)" in d.reason
    assert taken.exists()


def test_plan_refuses_ranges_it_cannot_name(tmp_path):
    s = season(tmp_path)
    a = touch(s / "t00.mkv")
    plan = plan_for(range_row(a, 2), opts=PlanOptions(template="{show} {title}"))
    assert [(x.reason, "no {e}" in x.detail) for x in plan.skipped] == [("cannot_name", True)]
    # not consecutive in the naming order: E2 and E3 swapped in 'dvd'
    b = make_bundle()
    ents = [OrderingEntry(season=1, episode={2: 3, 3: 2}.get(e.episode, e.episode),
                          episode_id=e.episode_id) for e in b.orderings["aired"].entries]
    b = b.model_copy(update={"orderings": {**b.orderings, "dvd": Ordering(
        source="tmdb", type="dvd", entries=ents)}})
    plan = plan_for(range_row(a, 2), bundle=b, opts=PlanOptions(naming_order="dvd"))
    assert [x.reason for x in plan.skipped] == ["cannot_name"]
    assert "not consecutive" in plan.skipped[0].detail and not plan.moves


def test_plan_notices_a_file_on_the_second_episode(tmp_path):
    s = season(tmp_path)
    a, b = touch(s / "t00.mkv"), touch(s / "t01.mkv")
    plan = plan_for(range_row(a, 2), row(b, 3))
    assert any("t01.mkv: S01E03 is also held by the two-episode file t00.mkv" in n
               for n in plan.notices)


def test_apply_recover_and_undo_keep_the_range(tmp_path):
    s = season(tmp_path)
    a = touch(s / "Disc 1" / "t00.mkv")
    db = StateDb(tmp_path / "state.db")
    plan = plan_for(range_row(a, 2, flags=["pinned"]))
    res = apply_plan(plan, db, command="c", args="", confirm=lambda p: True)
    assert not res.failed and (s / "synth_show_s01e02-e03.mkv").exists()
    (m,) = History(db).confirmed(SHOW)
    assert (m.episode_id, m.episode, m.episode_end, m.source) == ("tmdb:101", 2, 3, "manual")
    (op,) = done_ops(db, res.run_id)
    assert op.episode_end == 3 and "S01E02-E03" in format_ops([op])
    u = undo_run(db, None, confirm=lambda ops: True)
    assert len(u.undone) == 1 and a.exists()
    assert [x.episode_end for x in History(db).undone(SHOW)] == [3]

    # recovery confirms the range from the journal row
    b = touch(s / "Disc 1" / "t01.mkv")

    def crash(old, new):
        old.rename(new)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        apply_plan(plan_for(range_row(b, 2, flags=["pinned"])), db, command="c", args="",
                   confirm=lambda p: True, rename=crash)
    rep = recover(db)
    assert len(rep.completed) == 1 and not rep.unconfirmed
    (m,) = History(db).confirmed(SHOW)
    assert (m.episode, m.episode_end) == (2, 3)


# --- session ---------------------------------------------------------------------------------


@pytest.fixture
def bdir(tmp_path):
    d = tmp_path / SHOW
    write_refs(synth_bundle(), d)
    return d


def _session(bdir, tmp_path, rips, h):  # noqa: F811
    s = Session(write_refs(synth_bundle(), bdir), bdir, Config(), season=1,
                cache=AnalysisCache(tmp_path / "analysis.db"), history=h)
    s.analyze([rips[k] for k in ("e3", "e1", "nosubs")])
    return s


def _confirm_range(h, path, first, end, fp=None):
    fp = fp or fingerprint(path)
    h.touch_file(fp, size=1, path=str(path))
    h.confirm(fp, SHOW, episode_id=f"tmdb:1{first - 1:02d}", season=1, episode=first,
              title=None, source="manual", run_id=None, episode_end=end)


def test_session_holds_both_episodes_of_a_confirmed_range(bdir, tmp_path, rips):  # noqa: F811
    h = History(StateDb(":memory:"))
    _confirm_range(h, rips["e1"], 2, 3)  # the E1 rip confirmed as E2+E3: history wins
    s = _session(bdir, tmp_path, rips, h)
    rep = s.report()
    r1 = next(r for r in rep.files if r.path == str(rips["e1"]))
    ref = r1.decision.episode
    assert (ref.episode, ref.episode_end, ref.episode_end_id) == (2, 3, "tmdb:102")
    assert ref.code == "S01E02-E03" and r1.flags == [FLAG_CONFIRMED]
    r3 = next(r for r in rep.files if r.path == str(rips["e3"]))
    assert r3.decision.episode is None or r3.decision.episode.episode not in (2, 3)
    with pytest.raises(ValueError, match="S01E02-E03"):
        s.explain(rips["e1"])
    # the block solver's exclusion of a held column
    assert {(0, 2), (1, 2), (2, 2)} <= Session._pin_forbidden({}, 3, 4, {2})
    S = Session._pinned_S(np.zeros((2, 3)), {}, {1})
    assert (S[:, 1] < -1e6).all() and (S[:, [0, 2]] == 0).all()


def test_second_episode_of_another_files_range_is_a_history_duplicate(
        bdir, tmp_path, rips):  # noqa: F811
    h = History(StateDb(":memory:"))
    _confirm_range(h, rips["e1"], 2, 3, fp="9:other")  # another file holds E2+E3
    s = _session(bdir, tmp_path, rips, h)
    sol = s.solve()
    i = next(k for k, f in enumerate(sol.files) if f.path == rips["e3"])
    assert sol.fused.contributions["history"][i, 2] == pytest.approx(-np.log(10))
    rep = s.report()
    r3 = next(r for r in rep.files if r.path == str(rips["e3"]))
    assert r3.decision.episode.episode == 3 and FLAG_HISTORY_DUPLICATE in r3.flags


def test_tree_label_shows_the_range():
    from showvenist.tui.tree import _ep_label

    r = range_row(Path("/x/t.mkv"), 2, fp="f")
    assert _ep_label(r, 1) == "E02-E03 Episode Title 2"
    assert _ep_label(r, 2) == "S01E02-E03 Episode Title 2"
    assert EpisodeRef(episode_id="x", season=1, episode=4, title="t").code == "S01E04"


# --- eval from history -------------------------------------------------------------------------


def test_history_labels_turn_a_range_into_a_set_label(h):
    b = make_bundle()
    _range(h, "1:a", 2, 3)
    hl = history_labels(h, ["s"], bundle_of=lambda slug: b)
    assert [r.label for r in hl.rows + [p.row for p in hl.problems]] == ["@tmdb:101+@tmdb:102"]
    labels, problems = resolve_labels(
        [ManifestRow("/x.mkv", "s", "@tmdb:101+@tmdb:102")], {"s": b})
    assert not problems and labels[0].episode_ids == ("tmdb:101", "tmdb:102")
    _, problems = resolve_labels([ManifestRow("/x.mkv", "s", "@tmdb:101+S01E03")], {"s": b})
    assert problems and "mixes" in problems[0].reason


# --- CLI: assign, plan, apply, history, undo ---------------------------------------------------


def _assign(f, label, root, *extra):
    return runner.invoke(cli.app, ["assign", str(f), label, "--show", SHOW, "-o", str(root),
                                   "--yes", *extra])


def test_assign_refuses_labels_it_cannot_honor(tmp_path, monkeypatch):
    root, rips_dir = flat_setup(tmp_path, monkeypatch)
    f = rips_dir / "t01.mkv"
    cases = {
        "S01E01+S01E03": "not consecutive",
        "S01E02+S01E01": "does not come after",
        "S01E01+E02+E03": "at most two",
        "S01E01+S02E01": "S02E01 has no entry",
        "S01E01-E02": "S01E11+S01E12",
        "S01E01x": "unparseable",
        "S01E01+": "unparseable",
        "extra": "use --extra",
        "S01E01+S01E01": "same episode twice",
    }
    for label, why in cases.items():
        r = _assign(f, label, root)
        assert r.exit_code == 1, (label, r.output)
        assert why in r.output, (label, r.output)
    assert not (tmp_path / "XDG_DATA_HOME" / "showvenist" / "state.db").exists()


def test_assign_refuses_a_range_split_across_seasons(tmp_path, monkeypatch):
    root, rips_dir = flat_setup(tmp_path, monkeypatch)
    from showvenist.bundle.store import bundle_path, save_bundle

    save_bundle(write_refs(synth_bundle(seasons=2), bundle_path(SHOW, root)), root)
    r = _assign(rips_dir / "t01.mkv", "S01E04+S02E01", root)
    assert r.exit_code == 1 and "spans two seasons" in r.output


def test_assign_range_plan_apply_history_undo(tmp_path, monkeypatch):
    root, rips_dir = flat_setup(tmp_path, monkeypatch)  # rips/Season 1/t01.mkv (E1), t02 (E2)
    f = rips_dir / "t01.mkv"
    r = _assign(f, "S01E01+E02", root)
    assert r.exit_code == 0, r.output
    assert "S01E01-E02 'Story 1.1'" in r.stdout
    with StateDb.open_default() as db:
        (m,) = History(db).confirmed(SHOW)
        assert (m.episode_id, m.episode, m.episode_end, m.source) == ("tmdb:100", 1, 2, "manual")

    def identify(*a):
        return runner.invoke(cli.app, ["identify", str(rips_dir), "--show", SHOW, "-o",
                                       str(root), *a])

    r = identify("--plan")
    assert r.exit_code in (0, 2), r.output
    assert "synth_show_s01e01-e02.mkv" in r.stdout
    assert "S01E01-E02 Story 1.1" in r.stdout  # the report table and the plan's reason
    assert "synth_show_s01e02.mkv" not in r.stdout  # E2 is held by t01: nobody else gets it
    r = identify("--apply", "--confirm", "last")
    assert r.exit_code in (0, 2), r.output
    assert (rips_dir / "synth_show_s01e01-e02.mkv").exists() and not f.exists()
    with StateDb.open_default() as db:
        eps = [tuple(x) for x in db.conn.execute(
            "select episode, episode_end from rename_op where kind='episode'")]
        assert eps == [(1, 2)]
    # named now (§5.5): the name reads as E01, history gives the range back; E02 stays held
    r = identify("--json")
    rep = json.loads(r.stdout[r.stdout.index("{"):])
    got = {x["path"].rsplit("/", 1)[-1]: x for x in rep["files"]}
    named = got["synth_show_s01e01-e02.mkv"]
    assert "named" in named["flags"]
    assert named["decision"]["episode"]["episode_end"] == 2
    t02 = got["t02.mkv"]["decision"]["episode"]
    assert t02 is None or t02["episode"] not in (1, 2)

    r = runner.invoke(cli.app, ["history", "--show", SHOW, "-o", str(root)])
    assert r.exit_code == 0, r.output
    assert "S01E01-E02 Story 1.1" in r.stdout and "2 of 4 confirmed" in r.stdout
    assert "missing: E03, E04" in r.stdout

    r = runner.invoke(cli.app, ["undo", "--last", "--yes"])
    assert r.exit_code == 0, r.output
    assert "S01E01-E02" in r.stdout and "moved back 1" in r.stdout
    assert f.exists() and not (rips_dir / "synth_show_s01e01-e02.mkv").exists()
    with StateDb.open_default() as db:
        statuses = [tuple(x) for x in db.conn.execute(
            "select status, episode_end from match order by id")]
    # the assign's confirmed match stays (the apply linked it); the run itself wrote none
    assert statuses == [("confirmed", 2)]


def test_assign_rename_names_the_range(tmp_path, monkeypatch):
    root, rips_dir = flat_setup(tmp_path, monkeypatch)
    f = rips_dir / "t02.mkv"
    r = _assign(f, "S01E03+S01E04", root, "--rename")
    assert r.exit_code == 0, r.output
    assert (rips_dir / "synth_show_s01e03-e04.mkv").exists()
    with StateDb.open_default() as db:
        assert [tuple(x) for x in db.conn.execute(
            "select episode, episode_end, source from match")] == [(3, 4, "manual")]
    r = runner.invoke(cli.app, ["undo", "--yes"])
    assert r.exit_code == 0 and f.exists()
