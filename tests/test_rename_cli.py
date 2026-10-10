"""identify --plan / --apply, undo and assign --rename through the CLI, end to end with the real
Session on a tiny synthetic two-disc season (spec §4.3-4.4, §13, decision 58). tmp_path only;
XDG_* point into tmp_path, so the real state.db is never touched."""

from __future__ import annotations

import json
import shutil

import pytest
from typer.testing import CliRunner

from showvenist import cli
from showvenist.bundle.store import bundle_path, save_bundle
from showvenist.state.db import StateDb, state_db_path
from tests.test_cli_identify_eval import age
from tests.test_cli_identify_eval import setup as flat_setup
from tests.test_rename_plan import snapshot
from tests.test_session import dialogue_for, make_episode_mkv, synth_bundle, write_refs

runner = CliRunner()
SHOW = "synth-show-2000"


@pytest.fixture
def two_discs(tmp_path, monkeypatch):
    """Synth Show/Season 1/Disc 1 (E1 + a sidecar, E2), Disc 2 (E3, E4, two exact copies of
    E1)."""
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    monkeypatch.setattr(cli, "READINESS_SETTLE_SECONDS", 0.0)
    root = tmp_path / "bundles"
    d = bundle_path(SHOW, root)
    d.mkdir(parents=True)
    save_bundle(write_refs(synth_bundle(), d), root)
    show = tmp_path / "Synth Show"
    s = show / "Season 1"
    for disc, ks in ((1, (0, 1)), (2, (2, 3))):
        for j, k in enumerate(ks):
            age(make_episode_mkv(s / f"Disc {disc}" / f"t0{j}.mkv", [8, 11, 14, 17][k],
                                 dialogue_for(k)))
    for name in ("t02.mkv", "t03.mkv"):
        shutil.copyfile(s / "Disc 1" / "t00.mkv", s / "Disc 2" / name)
        age(s / "Disc 2" / name)
    (s / "Disc 1" / "t00.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n")
    age(s / "Disc 1" / "t00.en.srt")
    return root, show, s


def identify(root, show, *args, input=None):
    return runner.invoke(cli.app, ["identify", str(show), "--show", SHOW, "-o", str(root),
                                   *args], input=input)


def files_under(s):
    return sorted(p.relative_to(s).as_posix() for p in s.rglob("*") if p.is_file())


def rows(sql):
    with StateDb.open_default() as db:
        return [tuple(r) for r in db.conn.execute(sql)]


AFTER = [
    "Disc 2/synth_show_s01e01_dupe_1.mkv",
    "Disc 2/synth_show_s01e01_dupe_2.mkv",
    "synth_show_s01e01.en.srt",
    "synth_show_s01e01.mkv",
    "synth_show_s01e02.mkv",
    "synth_show_s01e03.mkv",
    "synth_show_s01e04.mkv",
]


def test_end_to_end_plan_apply_undo(two_discs, monkeypatch):
    root, show, s = two_discs
    before = snapshot(show)

    # the report carries the duplicate fields (decision 56)
    r = identify(root, show, "--json")
    rep = json.loads(r.stdout[r.stdout.index("{"):])
    dups = {f["path"].rsplit("/", 1)[-1]: (f["duplicate_of"], f["duplicate_rank"])
            for f in rep["files"] if f["duplicate_of"]}
    assert dups == {"t02.mkv": (str(s / "Disc 1" / "t00.mkv"), 0),
                    "t03.mkv": (str(s / "Disc 1" / "t00.mkv"), 1)}
    assert all(f["duplicate_of"] is None for f in rep["files"] if "duplicate" not in f["flags"])

    # --plan: the exact moves, nothing written
    r = identify(root, show, "--plan")
    assert r.exit_code == 0, r.output
    out = r.stdout
    assert "Moves (7)" in out and "Disc 2/synth_show_s01e01_dupe_2.mkv" in out
    assert "synth_show_s01e01.en.srt" in out and "[duplicate]" in out
    assert not state_db_path().exists()
    assert snapshot(show) == before

    # --apply: typed yes on a (pretend) TTY
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    r = identify(root, show, "--apply", input="yes\n")
    assert r.exit_code == 0, r.output
    assert "Moved 7 file(s), 0 failed (run 1)" in r.stdout
    assert "Undo with: showvenist undo 1" in r.stdout
    assert files_under(s) == AFTER
    assert sorted(snapshot(show)[0].values()) == sorted(before[0].values())  # same bytes
    assert rows("select status, count(*) from rename_op group by status") == [("done", 7)]
    # the copies here are byte-identical, so they share the primary's fingerprint and its match
    # (real MakeMKV copies differ in their headers and get their own)
    assert rows("select status, source, count(*) from match group by 1, 2") == [
        ("confirmed", "auto", 4)]

    # a second plan finds everything named: nothing to move
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    r = identify(root, show, "--plan")
    assert "No moves." in r.stdout and "Moves (" not in r.stdout
    assert files_under(s) == AFTER

    # undo: shows the moves, asks y/N
    r = runner.invoke(cli.app, ["undo"], input="n\n")
    assert r.exit_code == 1 and "Moves to reverse (7)" in r.stdout
    assert files_under(s) == AFTER
    r = runner.invoke(cli.app, ["undo", "--last"], input="y\n")
    assert r.exit_code == 0, r.output
    assert "moved back 7, skipped 0" in r.stdout
    after_undo = snapshot(show)
    assert after_undo[0] == before[0]  # the same files, byte for byte, at the same paths
    assert before[1] <= after_undo[1]  # no folder removed
    assert rows("select status, count(*) from match group by 1") == [("undone", 4)]
    r = runner.invoke(cli.app, ["undo"])
    assert r.exit_code == 1 and "nothing to undo" in r.output


def test_apply_refusals(two_discs, monkeypatch):
    root, show, s = two_discs
    before = snapshot(show)
    r = identify(root, show, "--apply")  # CliRunner: no TTY
    assert r.exit_code == 1 and "no TTY" in r.output
    assert not state_db_path().exists()
    for a, b in (("--plan", "--apply"), ("--plan", "--json"), ("--apply", "--report")):
        assert identify(root, show, a, b).exit_code == 1
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    for answer in ("no\n", "y\n", "YES please\n", "\n"):  # only a typed `yes` moves files
        r = identify(root, show, "--apply", input=answer)
        assert r.exit_code == 1, r.output
        assert "Nothing was moved." in r.stdout
    assert snapshot(show) == before
    assert rows("select count(*) from run") == [(0,)]
    assert rows("select count(*) from rename_op") == [(0,)]


def test_apply_with_confirm_runs_without_a_tty(two_discs):
    root, show, s = two_discs
    assert identify(root, show, "--plan").exit_code == 0
    r = identify(root, show, "--apply", "--confirm", "last")
    assert r.exit_code == 0, r.output
    assert files_under(s) == AFTER
    r = runner.invoke(cli.app, ["undo", "1", "--yes"])
    assert r.exit_code == 0 and "moved back 7" in r.stdout
    assert "synth_show_s01e01.mkv" not in files_under(s)


def test_assign_rename(tmp_path, monkeypatch):
    root, rips = flat_setup(tmp_path, monkeypatch)  # rips/Season 1/t01.mkv, t02.mkv
    f = rips / "t01.mkv"
    base = ["assign", str(f), "S01E02", "--show", SHOW, "-o", str(root), "--rename"]
    r = runner.invoke(cli.app, base, input="n\n")
    assert r.exit_code == 1 and "synth_show_s01e02.mkv" in r.stdout
    assert f.exists() and rows("select count(*) from match") == [(0,)]
    r = runner.invoke(cli.app, base, input="y\n")
    assert r.exit_code == 0, r.output
    assert not f.exists() and (rips / "synth_show_s01e02.mkv").exists()
    assert rows("select episode_id, status, source from match") == [
        ("tmdb:101", "confirmed", "manual")]
    assert rows("select kind, status from rename_op") == [("episode", "done")]
    # as an extra: to Season 1/extras/ (no disc folder: the name is kept)
    g = rips / "t02.mkv"
    r = runner.invoke(cli.app, ["assign", str(g), "--extra", "--show", SHOW, "-o", str(root),
                                "--rename", "--yes"])
    assert r.exit_code == 0, r.output
    assert (rips / "extras" / "t02.mkv").exists() and not g.exists()
    # undo the extra's run, then the episode's
    assert runner.invoke(cli.app, ["undo", "--yes"]).exit_code == 0
    assert runner.invoke(cli.app, ["undo", "--yes"]).exit_code == 0
    assert sorted(p.name for p in rips.iterdir() if p.is_file()) == ["t01.mkv", "t02.mkv"]
    assert (rips / "extras").is_dir()  # created folders stay


def test_duplicate_rank_follows_quality_then_path(tmp_path):
    from pathlib import Path

    from showvenist.analysis.duplicates import Quality
    from showvenist.config import Config
    from showvenist.session import Session

    s = Session(synth_bundle(), None, Config(), season=1)
    p, a, b, c, e = (Path(f"/r/{n}.mkv") for n in "pabce")
    s.duplicates = {a: p, b: p, c: p, e: p}
    s._exact = {e: b}  # an exact copy of b ranks with b's quality
    s._quality = {a: Quality(10, 1.0, 5), b: Quality(50, 0.5, 5), c: Quality(50, 0.5, 9)}
    got = {Path(k).name: v for k, v in s._duplicate_ranks().items()}
    assert got == {"c.mkv": (p, 0), "b.mkv": (p, 1), "e.mkv": (p, 2), "a.mkv": (p, 3)}
