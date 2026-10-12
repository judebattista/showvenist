"""Re-runs over a season the tool already worked on: the extras folder (sv-g7y), confirmed
extras that were never moved (sv-dsk) and copies of a confirmed episode (sv-sq6). End to end
through the CLI with the real Session on the synthetic two-disc season of test_rename_cli;
XDG_* point into tmp_path."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from showvenist import cli
from showvenist.analysis.types import AnalysisCache
from showvenist.bundle.store import bundle_path, load_bundle
from showvenist.config import Config, HierarchyConfig
from showvenist.hierarchy.parse import build_tree, is_extras_folder
from showvenist.rename.plan import build_plan
from showvenist.rename.types import PlanOptions
from showvenist.session import FLAG_IN_EXTRAS, Session
from showvenist.state.db import StateDb
from showvenist.state.history import History
from tests.test_cli_identify_eval import age
from tests.test_rename_cli import SHOW, files_under, identify, rows, two_discs  # noqa: F401
from tests.test_session import dialogue_for, make_episode_mkv

runner = CliRunner()


def add_extra(s: Path) -> Path:
    """A short file with a little of E1's dialogue: identified as an extra."""
    p = make_episode_mkv(s / "Disc 2" / "t04.mkv", 2, dialogue_for(0)[:20])
    age(p)
    return p


def report_json(root, show) -> dict:
    r = identify(root, show, "--json")
    assert "cannot parse" not in r.output, r.output
    return json.loads(r.stdout[r.stdout.index("{"):])


def by_name(rep: dict) -> dict[str, dict]:
    return {f["path"].rsplit("/", 1)[-1]: f for f in rep["files"]}


def assign(root, *args):
    r = runner.invoke(cli.app, ["assign", *map(str, args), "--show", SHOW, "-o", str(root),
                                "--yes"])
    assert r.exit_code == 0, r.output
    return r


def test_extras_folder_names_are_case_insensitive():
    for name in ("extras", "Extras", "FEATURETTES", "Behind The Scenes", "deleted scenes",
                 "Interviews", "scenes", "Shorts", "trailers", "Other"):
        assert is_extras_folder(name), name
    for name in ("Disc 1", "Season 1", "extra", "bonus", "Specials"):
        assert not is_extras_folder(name), name


def test_tree_keeps_extras_folders_out_of_discs_and_problems(tmp_path):
    s = tmp_path / "Show" / "Season 1"
    for rel in ("Disc 1/t00.mkv", "extras/Disc 1__t09.mkv", "Featurettes/sub/f.mkv"):
        (s / rel).parent.mkdir(parents=True, exist_ok=True)
        (s / rel).write_bytes(b"x")
    (s / "Behind the Scenes").mkdir()  # empty: no problem either
    (tmp_path / "Show" / "Trailers").mkdir()
    (tmp_path / "Show" / "Trailers" / "t.mkv").write_bytes(b"x")
    show = build_tree(tmp_path / "Show", config=HierarchyConfig(), tag_of=lambda p: None)
    assert show.problems == []
    (season,) = show.seasons
    assert [d.key for d in season.discs] == ["Disc 1"] and season.loose == []
    assert sorted(f.path.name for f in season.extras) == ["Disc 1__t09.mkv", "f.mkv"]
    assert [f.path.name for f in show.extras] == ["t.mkv"]
    assert [f.path.name for f in show.all_files()] == ["t00.mkv"]  # extras are not solved
    # the season folder itself as the root
    alone = build_tree(s, config=HierarchyConfig(), tag_of=lambda p: None)
    assert alone.problems == [] and len(alone.seasons[0].extras) == 2


def test_rerun_after_apply_and_after_undo(two_discs, monkeypatch):  # noqa: F811
    root, show, s = two_discs
    add_extra(s)
    r = identify(root, show, "--plan")
    assert r.exit_code == 0, r.output
    assert "extras/Disc 2__t04.mkv" in r.stdout
    r = identify(root, show, "--apply", "--confirm", "last")
    assert r.exit_code == 0, r.output
    assert (s / "extras" / "Disc 2__t04.mkv").exists()

    # a re-run neither fails on extras/ nor solves or moves what is in it
    r = identify(root, show, "--report")
    assert r.exit_code == 0, r.output
    assert "cannot parse" not in r.output and "already in extras/" in r.stdout
    row = by_name(report_json(root, show))["Disc 2__t04.mkv"]
    assert row["flags"] == [FLAG_IN_EXTRAS] and row["decision"]["kind"] == "extra"
    r = identify(root, show, "--plan")
    assert r.exit_code == 0, r.output
    assert "Moves (" not in r.stdout and "already in extras/" in r.stdout

    # undo removes the extras/ folder it created, now empty; a re-run still works
    r = runner.invoke(cli.app, ["undo", "--last", "--yes"])
    assert r.exit_code == 0, r.output
    assert not (s / "extras").exists() and "removed the empty folder" in r.stdout
    assert (s / "Disc 2" / "t04.mkv").exists()
    # an empty extras folder the user made stays, and is no parse problem either
    (s / "extras").mkdir()
    (s / "Featurettes").mkdir()
    r = identify(root, show, "--report")
    assert "cannot parse" not in r.output and r.exit_code == 0, r.output


def test_undo_keeps_a_created_folder_that_is_not_empty(two_discs):  # noqa: F811
    root, show, s = two_discs
    add_extra(s)
    assert identify(root, show, "--plan").exit_code == 0
    assert identify(root, show, "--apply", "--confirm", "last").exit_code == 0
    (s / "extras" / "mine.txt").write_text("the user's")
    r = runner.invoke(cli.app, ["undo", "--last", "--yes"])
    assert r.exit_code == 0, r.output
    assert (s / "extras" / "mine.txt").exists() and "removed the empty folder" not in r.stdout


def test_file_in_extras_confirmed_as_an_episode_is_left_with_a_notice(two_discs):  # noqa: F811
    root, show, s = two_discs
    e2 = s / "Disc 1" / "t01.mkv"
    assign(root, e2, "S01E02")
    (s / "extras").mkdir()
    e2.rename(s / "extras" / "t01.mkv")  # moved there by hand
    rep = report_json(root, show)
    row = by_name(rep)["t01.mkv"]
    assert row["flags"] == [FLAG_IN_EXTRAS] and row["decision"]["kind"] == "extra"
    assert any("confirmed in history as S01E02" in n for n in rep["notices"])
    r = identify(root, show, "--plan")
    assert r.exit_code == 0, r.output
    assert "extras/t01.mkv" not in r.stdout.split("Left where")[0]  # not among the moves


def test_assign_extra_without_rename_is_moved_by_the_next_plan(two_discs):  # noqa: F811
    root, show, s = two_discs
    x = add_extra(s)
    assign(root, x, "--extra")
    assert x.exists()  # no --rename: nothing moved
    r = identify(root, show, "--plan")
    assert r.exit_code == 0, r.output
    assert "extras/Disc 2__t04.mkv" in r.stdout and "[confirmed]" in r.stdout
    r = identify(root, show, "--apply", "--confirm", "last")
    assert r.exit_code == 0, r.output
    assert (s / "extras" / "Disc 2__t04.mkv").exists() and not x.exists()
    # the earlier decision is linked, not written again
    assert rows("select count(*) from match where episode_id is null") == [(1,)]
    r = identify(root, show, "--plan")  # re-runs skip it
    assert "Moves (" not in r.stdout and "already in extras/" in r.stdout


def test_copies_of_a_confirmed_episode_are_dupes_not_extras(two_discs):  # noqa: F811
    root, show, s = two_discs
    assign(root, s / "Disc 1" / "t00.mkv", "S01E01")
    rep = report_json(root, show)
    got = by_name(rep)
    for name in ("t02.mkv", "t03.mkv"):  # both byte-identical copies of Disc 1/t00.mkv
        f = got[name] if got[name]["disc"] == "Disc 2" else None
        assert f is not None and f["duplicate_of"] == str(s / "Disc 1" / "t00.mkv")
        assert "confirmed" not in f["flags"] and f["decision"]["kind"] == "episode"
    r = identify(root, show, "--plan")
    assert r.exit_code == 0, r.output
    assert "Disc 2/synth_show_s01e01_dupe_1.mkv" in r.stdout
    assert "Disc 2/synth_show_s01e01_dupe_2.mkv" in r.stdout
    assert "extras/" not in r.stdout
    assert identify(root, show, "--apply", "--confirm", "last").exit_code == 0
    assert "Disc 2/synth_show_s01e01_dupe_1.mkv" in files_under(s)


def test_copy_with_its_own_fingerprint_of_a_confirmed_episode(two_discs, monkeypatch):  # noqa: F811
    """Real MakeMKV copies differ in their headers (Burn Notice S01: C1_t00 copies B1_t03):
    another fingerprint, the same content key. The copy is found against the confirmed file."""
    root, show, s = two_discs
    prim = s / "Disc 1" / "t00.mkv"
    import showvenist.session as session_mod

    real = session_mod.fingerprint
    monkeypatch.setattr(session_mod, "fingerprint", lambda p: f"{real(p)}:{Path(p).name}"
                        if Path(p).parent.name == "Disc 2" else real(p))
    h = History(StateDb(":memory:"))
    fp = real(prim)
    h.touch_file(fp, size=prim.stat().st_size, path=str(prim))
    h.confirm(fp, SHOW, episode_id="tmdb:100", season=1, episode=1, title=None,
              source="manual", run_id=None)
    bdir = bundle_path(SHOW, root)
    bundle = load_bundle(bdir)
    tree = build_tree(show, config=HierarchyConfig(), tag_of=lambda p: None)
    sess = Session(bundle, bdir, Config(), season=1, tree=tree, history=h,
                   cache=AnalysisCache(s.parent.parent / "analysis.db"))
    sess.analyze()
    rep = sess.report()
    copies = [f for f in rep.files if f.duplicate_of == str(prim)]
    assert sorted(Path(f.path).name for f in copies) == ["t02.mkv", "t03.mkv"]
    plan = build_plan(rep, bundle, PlanOptions(), history=h)
    kinds = {m.old.name: (m.kind, m.new.name) for m in plan.moves if m.old.parent.name == "Disc 2"
             and m.old.name in ("t02.mkv", "t03.mkv")}
    assert kinds == {"t02.mkv": ("duplicate", "synth_show_s01e01_dupe_1.mkv"),
                     "t03.mkv": ("duplicate", "synth_show_s01e01_dupe_2.mkv")}


def test_named_dupe_never_holds_the_episode_over_its_primary(two_discs):  # noqa: F811
    root, show, s = two_discs
    assert identify(root, show, "--plan").exit_code == 0
    assert identify(root, show, "--apply", "--confirm", "last").exit_code == 0
    got = by_name(report_json(root, show))
    assert got["synth_show_s01e01.mkv"]["flags"] == ["named"]  # Disc 2 is walked first
    assert "duplicate" in got["synth_show_s01e01_dupe_1.mkv"]["flags"]


def test_plan_leaves_an_extra_already_in_an_extras_folder(tmp_path):
    from tests.test_rename_plan import plan_for, row, season, touch

    s = season(tmp_path)
    a = touch(s / "extras" / "Disc 1__t09.mkv")
    b = touch(s / "Featurettes" / "making_of.mkv")
    c = touch(s / "Disc 1" / "t08.mkv")
    plan = plan_for(row(a, None, flags=["confirmed"]), row(b, None, flags=["in_extras"]),
                    row(c, None, flags=["confirmed"]))
    assert {k.path.name: (k.reason, k.detail) for k in plan.skipped} == {
        "Disc 1__t09.mkv": ("already_in_place", "already in extras/"),
        "making_of.mkv": ("already_in_place", "already in Featurettes/")}
    (m,) = plan.moves
    assert m.old == c and m.new == s / "extras" / "Disc 1__t08.mkv" and m.confirmed
