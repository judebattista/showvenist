"""assign / history / forget / eval --from-history / identify with history, via the CLI."""

import json

import pytest
from typer.testing import CliRunner

from showvenist import cli
from showvenist.state.db import StateDb, state_db_path
from showvenist.state.fingerprint import fingerprint
from showvenist.state.history import History
from tests.test_cli_identify_eval import setup

runner = CliRunner()
SHOW = "synth-show-2000"


@pytest.fixture
def env(tmp_path, monkeypatch):
    root, rips = setup(tmp_path, monkeypatch)
    return root, rips


def run(root, *args, input=None):
    return runner.invoke(cli.app, [*args, "-o", str(root)], input=input)


def matches():
    with StateDb.open_default() as db:
        return [tuple(r) for r in db.conn.execute(
            "select status, episode_id, source from match order by id")]


def assign(root, rips, name, label, *extra):
    return run(root, "assign", str(rips / name), label, "--show", SHOW, "--yes", *extra)


def test_assign_asks_first_and_writes_nothing_on_no(env):
    root, rips = env
    r = run(root, "assign", str(rips / "t01.mkv"), "S01E01", "--show", SHOW, input="n\n")
    assert r.exit_code == 1 and "Nothing written" in r.output
    assert matches() == []
    assert (rips / "t01.mkv").exists()


def test_assign_writes_and_rejects_the_previous_match(env):
    root, rips = env
    r = run(root, "assign", str(rips / "t01.mkv"), "S01E01", "--show", SHOW, input="y\n")
    assert r.exit_code == 0, r.output
    assert matches() == [("confirmed", "tmdb:100", "manual")]
    r = assign(root, rips, "t01.mkv", "S01E03")
    assert r.exit_code == 0 and "will become rejected" in r.output
    assert matches() == [("rejected", "tmdb:100", "manual"), ("confirmed", "tmdb:102", "manual")]
    r = assign(root, rips, "t01.mkv", "--extra")
    assert r.exit_code == 0, r.output
    assert matches()[-1] == ("confirmed", None, "manual")
    assert sorted(p.name for p in rips.iterdir()) == ["t01.mkv", "t02.mkv"]  # nothing moved
    with StateDb.open_default() as db:
        assert db.conn.execute("select count(*) from rename_op").fetchone()[0] == 0
        assert db.conn.execute("select command from run").fetchall()[0][0] == "assign"


def test_assign_errors(env):
    root, rips = env
    f = str(rips / "t01.mkv")
    assert assign(root, rips, "t01.mkv", "S09E09").exit_code == 1  # no such episode
    assert run(root, "assign", f, "--show", SHOW, "--yes").exit_code == 1  # nothing to assign
    assert run(root, "assign", f, "S01E01", "--extra", "--show", SHOW).exit_code == 1
    assert assign(root, rips, "t01.mkv", "S01E01", "--order", "nope").exit_code == 1
    assert matches() == []


def test_history_lists_confirmed_missing_and_extras(env):
    root, rips = env
    assign(root, rips, "t01.mkv", "S01E02")
    assign(root, rips, "t02.mkv", "--extra")
    r = run(root, "history", "--show", SHOW)
    assert r.exit_code == 0, r.output
    assert "Season 1: 1 of 4 confirmed" in r.output
    assert f"S01E02 Story 1.2  {rips / 't01.mkv'}" in r.output
    assert "missing: E01, E03, E04" in r.output
    assert f"Confirmed extras (1)\n    {rips / 't02.mkv'}" in r.output
    assert "Season 1" not in run(root, "history", "--show", SHOW, "--season", "2").output
    assert SHOW in run(root, "history").output  # no --show: every show in history


def test_forget_asks_then_removes(env):
    root, rips = env
    assign(root, rips, "t01.mkv", "S01E01")
    f = str(rips / "t01.mkv")
    r = runner.invoke(cli.app, ["forget", f], input="n\n")
    assert r.exit_code == 1 and "Nothing removed" in r.output and len(matches()) == 1
    r = runner.invoke(cli.app, ["forget", fingerprint(rips / "t01.mkv"), "--yes"])
    assert r.exit_code == 0, r.output
    assert matches() == []
    assert runner.invoke(cli.app, ["forget", f, "--yes"]).exit_code == 1  # nothing known


def test_identify_reports_confirmed_files_from_history(env):
    root, rips = env
    base = ["identify", str(rips), "--show", SHOW, "-o", str(root), "--json"]
    assert not state_db_path().exists()
    r = runner.invoke(cli.app, base)
    assert r.exit_code in (0, 2) and not state_db_path().exists()  # report mode writes nothing
    assign(root, rips, "t01.mkv", "S01E04")  # a decision that contradicts the evidence
    r = runner.invoke(cli.app, base)
    rep = json.loads(r.stdout[r.stdout.index("{"):])
    by = {f["path"].rsplit("/", 1)[-1]: f for f in rep["files"]}
    assert by["t01.mkv"]["flags"] == ["confirmed"]
    assert by["t01.mkv"]["decision"]["episode"]["episode"] == 4
    assert "confirmed" not in by["t02.mkv"]["flags"]
    r = runner.invoke(cli.app, [*base, "--reanalyze"])
    rep = json.loads(r.stdout[r.stdout.index("{"):])
    assert all("confirmed" not in f["flags"] for f in rep["files"])
    assert len(matches()) == 1  # still nothing written by identify


def test_eval_from_history_end_to_end(env):
    root, rips = env
    assign(root, rips, "t01.mkv", "S01E01")
    assign(root, rips, "t02.mkv", "S01E02")
    with StateDb.open_default() as db:
        h = History(db)
        fp = fingerprint(rips / "t01.mkv")
        h.reject(fp, SHOW, episode_id="tmdb:103", run_id=None)  # a negative label
        h.touch_file("5:gone", size=5, path=str(rips / "gone.mkv"))
        h.confirm("5:gone", SHOW, episode_id="tmdb:102", season=1, episode=3, title=None,
                  source="manual", run_id=None)
    r = run(root, "eval", "--from-history", "--show", SHOW)
    assert r.exit_code == 0, r.output
    assert "OVERALL" in r.output
    assert "    2" in r.output.split("OVERALL")[1].split("\n")[0]  # two labeled files
    assert "decisions equal to a rejected match: 0 of 1" in r.output
    assert "gone.mkv: file no longer exists" in r.output
    assert sorted(p.name for p in rips.iterdir()) == ["t01.mkv", "t02.mkv"]


def test_eval_needs_exactly_one_label_source(env, tmp_path):
    root, _ = env
    assert run(root, "eval").exit_code == 1
    m = tmp_path / "m.csv"
    m.write_text("path,show,label\n")
    assert run(root, "eval", str(m), "--from-history").exit_code == 1
