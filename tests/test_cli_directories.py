"""identify on Show / Season / Disc folders (spec §4.3, §5) with tiny synthetic MKVs.

No real media, no network, no LLM.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from showvenist import cli
from showvenist import session as session_mod
from showvenist.bundle.schema import Ordering
from showvenist.bundle.store import bundle_path, save_bundle
from showvenist.hierarchy.marker import MARKER_NAME
from tests.test_session import dialogue_for, make_episode_mkv, synth_bundle, write_refs

runner = CliRunner()
SLUG = "synth-show-2000"
MINUTES = [8, 11, 14, 17]


def age(*paths: Path, seconds: float = 3600) -> None:
    then = time.time() - seconds
    for p in paths:
        os.utime(p, (then, then))


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """Bundle (two seasons, a second ordering 'dvd' equal to aired, a TMDB id) and a show
    folder: Season 1 with two discs of two episodes each, Season 2 with one disc."""
    base = tmp_path_factory.mktemp("dirs")
    root = base / "bundles"
    bdir = bundle_path(SLUG, root)
    bdir.mkdir(parents=True)
    b = write_refs(synth_bundle(seasons=2), bdir)
    aired = b.orderings["aired"]
    b = b.model_copy(update={
        "orderings": {"aired": aired,
                      "dvd": Ordering(source="tmdb", type="dvd", name="DVD",
                                      entries=list(aired.entries))},
        "show": b.show.model_copy(update={"ids": b.show.ids.model_copy(update={"tmdb": 4242})}),
    })
    save_bundle(b, root)
    show = base / "src" / "Synth Show"
    files = []
    for disc, ks in ((1, (0, 1)), (2, (2, 3))):
        for n, k in enumerate(ks):
            files.append(make_episode_mkv(show / "Season 1" / f"Disc {disc}" / f"t0{n}.mkv",
                                          MINUTES[k], dialogue_for(k)))
    for n, k in enumerate((0, 1)):
        files.append(make_episode_mkv(show / "Season 2" / "SYNTH_S2_D1" / f"t0{n}.mkv",
                                      MINUTES[k], dialogue_for(k, 2)))
    age(*files)
    return root, show


@pytest.fixture
def env(library, tmp_path, monkeypatch):
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    monkeypatch.setattr(cli, "READINESS_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    root, show = library
    copy = tmp_path / "Synth Show"
    shutil.copytree(show, copy)  # copytree keeps the back-dated mtimes
    return root, copy


@pytest.fixture
def sessions(monkeypatch):
    """Records every Session identify builds (to check use_blocks and the solve)."""
    made = []

    class Spy(session_mod.Session):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    monkeypatch.setattr(session_mod, "Session", Spy)
    return made


def ident(root, *args):
    return runner.invoke(cli.app, ["identify", *map(str, args), "-o", str(root)])


def report_of(r) -> dict | list:
    out = r.stdout
    start = min(i for i in (out.find("{"), out.find("[")) if i >= 0)
    return json.loads(out[start:])


def episodes(rep: dict) -> dict[str, tuple[str | None, int | None]]:
    return {
        "/".join(Path(f["path"]).parts[-2:]): (f["disc"], (f["decision"]["episode"] or {})
                                               .get("episode"))
        for f in rep["files"]
    }


def test_season_folder_tree_printed_first_and_show_from_folder_name(env, sessions):
    root, show = env
    r = ident(root, show / "Season 1", "--json")
    assert r.exit_code in (0, 2), r.output
    out = r.output
    # the tree is printed (to stderr) before any analysis progress
    assert out.index("Show: Synth Show") < out.index("[1/4]")
    assert "Season 1: 4 files" in r.stderr
    assert "disc 1   Disc 1: 2 files" in r.stderr and "disc 2   Disc 2: 2 files" in r.stderr
    assert "show resolved by: name" in r.stderr
    rep = report_of(r)
    assert rep["season"] == 1 and rep["show_slug"] == SLUG
    assert episodes(rep) == {
        "Disc 1/t00.mkv": ("Disc 1", 1), "Disc 1/t01.mkv": ("Disc 1", 2),
        "Disc 2/t00.mkv": ("Disc 2", 3), "Disc 2/t01.mkv": ("Disc 2", 4),
    }
    # blocks are used by default; the folder-name suggestion is a notice only
    assert len(sessions) == 1 and sessions[0].use_blocks
    assert sessions[0].solution.used_blocks
    assert set(sessions[0].solution.blocks) == {"Disc 1", "Disc 2"}
    assert any("Synth Show (2000) [tmdbid-4242]" in n and "not applied" in n
               for n in rep["notices"])
    assert any("no marker file was written" in n for n in rep["notices"])
    assert not (show / MARKER_NAME).exists()  # identify never writes a marker


def test_no_blocks(env, sessions):
    root, show = env
    r = ident(root, show / "Season 1", "--json", "--no-blocks")
    assert r.exit_code in (0, 2), r.output
    assert len(sessions) == 1 and not sessions[0].use_blocks
    assert not sessions[0].solution.used_blocks
    assert {v[1] for v in episodes(report_of(r)).values()} == {1, 2, 3, 4}


def test_show_folder_solves_each_season(env, sessions):
    root, show = env
    r = ident(root, show, "--json")
    assert r.exit_code in (0, 2), r.output
    reps = report_of(r)
    assert isinstance(reps, list) and [x["season"] for x in reps] == [1, 2]
    assert {v[1] for v in episodes(reps[1]).values()} == {1, 2}
    assert "Season 2: 2 files" in r.stderr and "SYNTH_S2_D1" in r.stderr
    assert [s.season for s in sessions] == [1, 2]
    # --season picks one season out of the tree
    r = ident(root, show, "--json", "--season", "2")
    assert report_of(r)["season"] == 2


def test_not_ready_file_is_skipped_and_exit_2(env):
    root, show = env
    good = show / "Season 1" / "Disc 2" / "t01.mkv"
    half = show / "Season 1" / "Disc 2" / "t02.mkv"
    data = good.read_bytes()
    half.write_bytes(data[: len(data) // 2])
    age(half)
    fresh = show / "Season 1" / "Disc 2" / "t03.mkv"
    shutil.copyfile(good, fresh)  # complete, but modified just now
    r = ident(root, show / "Season 1", "--json")
    assert r.exit_code == 2, r.output
    assert "Disc 2: 4 files, 2 not ready" in r.stderr
    assert "not ready: t02.mkv (truncated)" in r.stderr
    assert "not ready: t03.mkv (modified" in r.stderr
    rep = report_of(r)
    flags = {Path(f["path"]).name: f["flags"] for f in rep["files"] if "Disc 2" in f["path"]}
    assert "not_ready" in flags["t02.mkv"] and "not_ready" in flags["t03.mkv"]
    assert any("Disc 2: 2 files still being written" in n for n in rep["notices"])
    # ...and nothing is analyzed for them: only the 4 ready files are counted
    assert "[6/6]" in r.stderr


def test_unparseable_level_fails_non_interactive(env, monkeypatch):
    root, show = env
    odd = show / "Season 1" / "Bonus stuff"
    odd.mkdir()
    shutil.copyfile(show / "Season 1" / "Disc 1" / "t00.mkv", odd / "x.mkv")
    age(odd / "x.mkv")
    r = ident(root, show / "Season 1", "--json")
    assert r.exit_code == 1
    assert "cannot parse disc folder 'Bonus stuff'" in r.stderr and "never guess" in r.stderr
    assert "[1/" not in r.output  # stopped before analysis
    # interactive: report and ask; yes continues with the loose files
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    r = runner.invoke(cli.app, ["identify", str(show / "Season 1"), "-o", str(root),
                                "--json"], input="y\n")
    assert r.exit_code in (0, 2), r.output
    assert len(report_of(r)["files"]) == 5


def test_discs_without_numbers_fail_non_interactive(env):
    root, show = env
    s1 = show / "Season 1"
    (s1 / "Disc 1").rename(s1 / "Disc A")
    (s1 / "Disc 2").rename(s1 / "Disc B")
    r = ident(root, s1, "--json")
    assert r.exit_code == 1 and "creation time" in r.stderr and "never guess" in r.stderr


def test_marker_is_honored(env, sessions):
    root, show = env
    renamed = show.with_name("My Rips")  # folder name alone would not resolve
    show.rename(renamed)
    r = ident(root, renamed / "Season 1", "--json")
    assert r.exit_code == 1 and "no bundle matches" in r.stderr
    (renamed / MARKER_NAME).write_text(
        'show = "Synth Show"\nbundle = "synth-show-2000"\nlanguage = "en"\n'
        'naming_order = "dvd"\n'
    )
    before = (renamed / MARKER_NAME).read_text()
    r = ident(root, renamed / "Season 1", "--json")
    assert r.exit_code in (0, 2), r.output
    assert "show resolved by: marker" in r.stderr
    rep = report_of(r)
    assert rep["ordering"] == "dvd"  # --order defaults to the marker's naming_order
    assert sessions[-1].marker is not None and sessions[-1].starting_ordering == "dvd"
    assert not any("no marker file" in n for n in rep["notices"])
    r = ident(root, renamed / "Season 1", "--json", "--order", "aired")  # flags override
    assert report_of(r)["ordering"] == "aired"
    assert (renamed / MARKER_NAME).read_text() == before


def test_named_files_are_pins_not_analyzed(env):
    root, show = env
    d1 = show / "Season 1" / "Disc 1"
    (d1 / "t00.mkv").rename(d1 / "synth_s01e01.mkv")
    r = ident(root, show / "Season 1", "--json")
    assert r.exit_code in (0, 2), r.output
    assert "Disc 1: 2 files, 1 named" in r.stderr
    rep = report_of(r)
    named = next(f for f in rep["files"] if f["path"].endswith("synth_s01e01.mkv"))
    assert "named" in named["flags"] and named["decision"]["episode"]["episode"] == 1
    assert not any(s["available"] for s in named["signals"])  # never analyzed or scored


def test_explicit_files_and_allow_duplicates(env, sessions):
    root, show = env
    d2 = show / "Season 1" / "Disc 2"
    r = ident(root, d2 / "t00.mkv", d2 / "t01.mkv", "--json", "--allow-duplicates")
    assert r.exit_code in (0, 2), r.output
    rep = report_of(r)
    assert rep["season"] == 1 and {f["disc"] for f in rep["files"]} == {"Disc 2"}
    assert any("near-duplicate detection is not implemented" in n for n in rep["notices"])
