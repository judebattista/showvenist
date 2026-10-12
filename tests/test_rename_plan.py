"""Rename plans (spec §13.1-13.3, decisions 56-57) on tiny synthetic files (tmp_path only)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from showvenist.bundle.schema import Ordering, OrderingEntry
from showvenist.rename.plan import build_plan
from showvenist.rename.sidecars import find_sidecars
from showvenist.rename.types import PlanOptions
from showvenist.report import format_odds
from showvenist.report_schema import (
    Confidence,
    Decision,
    EpisodeRef,
    FileReport,
    Report,
    Summary,
)
from showvenist.state.db import StateDb
from showvenist.state.fingerprint import fingerprint
from showvenist.state.history import History
from tests.synth import make_bundle

SLUG = "synth-show-2000"


def touch(path: Path, data: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if data is not None else f"content of {path}".encode() * 20)
    return path


def snapshot(root: Path) -> tuple[dict[str, str], set[str]]:
    """Every file (relative path -> content hash) and every folder under root."""
    files, dirs = {}, set()
    for d, dn, fn in os.walk(root):
        for n in dn:
            dirs.add(os.path.relpath(os.path.join(d, n), root))
        for n in fn:
            p = os.path.join(d, n)
            files[os.path.relpath(p, root)] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return files, dirs


def row(path: Path, ep: int | None, *, level: str = "high", flags=(), disc: str | None = None,
        dup_of: Path | None = None, rank: int | None = None, fp: str | None = None,
        bundle=None) -> FileReport:
    """A report row; `ep` None = extra. Episode ids follow tests.synth.make_bundle."""
    ref = None
    if ep is not None:
        ref = EpisodeRef(episode_id=f"tmdb:{99 + ep}", season=1, episode=ep,
                         title=f"Episode Title {ep}")
    if fp is None:
        fp = fingerprint(path) if path.exists() else f"fp-{path.name}"
    delta = None if level == "high" else 1.0
    return FileReport(
        path=str(path), fingerprint=fp,
        disc=disc if disc is not None else (path.parent.name
                                            if path.parent.name.startswith("Disc") else None),
        duration_s=60.0, regime="base",
        decision=Decision(kind="episode" if ref else "extra", episode=ref),
        confidence=Confidence(level=level, delta=delta, odds=format_odds(delta)),
        flags=list(flags),
        duplicate_of=str(dup_of) if dup_of else None, duplicate_rank=rank,
    )


def report(*rows: FileReport, bundle=None) -> Report:
    b = bundle or make_bundle()
    return Report(tool_version="t", show_slug=SLUG, bundle_revision=b.bundle_revision,
                  ordering="aired", season=1, files=list(rows),
                  summary=Summary(files=len(rows), high=0, medium=0, low=0, extras=0))


def plan_for(*rows, bundle=None, **kw):
    b = bundle or make_bundle()
    opts = kw.pop("opts", None) or PlanOptions()
    return build_plan(report(*rows, bundle=b), b, opts, **kw)


def season(tmp_path: Path) -> Path:
    return tmp_path / "Synth Show" / "Season 1"


def moves(plan) -> list[tuple[str, str, str]]:
    return [(m.kind, m.old.name, str(m.new)) for m in plan.moves]


# ---------------------------------------------------------------------------------------------


def test_find_sidecars(tmp_path):
    v = touch(tmp_path / "title_t00.mkv")
    touch(tmp_path / "title_t00.en.srt")
    touch(tmp_path / "title_t00.nfo")
    touch(tmp_path / "title_t00.1.mkv")  # another video: never a sidecar
    touch(tmp_path / "title_t001.srt")  # different stem
    touch(tmp_path / "title_t01.srt")
    assert [p.name for p in find_sidecars(v)] == ["title_t00.en.srt", "title_t00.nfo"]
    assert find_sidecars(tmp_path / "missing" / "x.mkv") == []


def test_episodes_flatten_in_naming_order_with_sidecars(tmp_path):
    s = season(tmp_path)
    a = touch(s / "Disc 2" / "t00.mkv")
    b = touch(s / "Disc 1" / "t01.mkv")
    c = touch(s / "Disc 1" / "t00.mkv")
    touch(s / "Disc 1" / "t00.en.srt")
    before = snapshot(tmp_path)
    plan = plan_for(row(a, 3), row(b, 2), row(c, 1))
    assert plan.ok, plan.errors
    assert moves(plan) == [
        ("episode", "t00.mkv", str(s / "synth_show_s01e01.mkv")),
        ("sidecar", "t00.en.srt", str(s / "synth_show_s01e01.en.srt")),
        ("episode", "t01.mkv", str(s / "synth_show_s01e02.mkv")),
        ("episode", "t00.mkv", str(s / "synth_show_s01e03.mkv")),
    ]
    assert plan.moves[1].parent == 0 and plan.moves[1].fingerprint == plan.moves[0].fingerprint
    m = plan.moves[0]
    assert (m.season, m.episode, m.title, m.episode_id) == (1, 1, "Episode Title 1", "tmdb:100")
    assert m.disc == "Disc 1" and not m.pinned and m.confidence == "high"
    assert plan.cross_filesystem == [] and plan.skipped == []
    assert snapshot(tmp_path) == before  # planning writes nothing
    assert moves(plan_for(row(a, 3), row(b, 2), row(c, 1))) == moves(plan)  # deterministic


def test_naming_order_numbers_the_files(tmp_path):
    b = make_bundle()
    dvd = Ordering(source="tmdb", type="dvd", entries=[
        OrderingEntry(season=1, episode=k + 1, episode_id=f"tmdb:{100 + (k + 1) % 6}")
        for k in range(6)])
    b = b.model_copy(update={"orderings": {**b.orderings, "dvd": dvd}})
    f = touch(season(tmp_path) / "Disc 1" / "t00.mkv")
    plan = plan_for(row(f, 2), bundle=b, opts=PlanOptions(naming_order="dvd"))
    assert plan.naming_order == "dvd"
    assert plan.moves[0].new.name == "synth_show_s01e01.mkv"  # tmdb:101 is E01 in dvd order
    custom = PlanOptions(template="{show} - {s}x{e:02} - {title}")
    assert plan_for(row(f, 2), bundle=b, opts=custom).moves[0].new.name == (
        "Synth Show - 1x02 - Episode Title 2.mkv")


def test_extras_get_the_disc_prefix(tmp_path):
    s = season(tmp_path)
    x1 = touch(s / "Disc 1" / "Synth - Season 1_t05.mkv")
    x2 = touch(s / "Disc 2" / "Synth - Season 1_t05.mkv")  # same MakeMKV name on another disc
    x3 = touch(s / "loose.mkv")  # no disc folder: keeps its name
    x4 = touch(s / "extras" / "already.mkv")  # already in extras/: in place
    plan = plan_for(row(x1, None), row(x2, None), row(x3, None), row(x4, None))
    assert plan.ok
    assert sorted(str(m.new) for m in plan.moves) == sorted([
        str(s / "extras" / "Disc 1__Synth - Season 1_t05.mkv"),
        str(s / "extras" / "Disc 2__Synth - Season 1_t05.mkv"),
        str(s / "extras" / "loose.mkv"),
    ])
    assert all(m.kind == "extra" and m.episode_id is None for m in plan.moves)
    assert [(k.path.name, k.reason) for k in plan.skipped] == [
        ("already.mkv", "already_in_place")]
    off = plan_for(row(x1, None), opts=PlanOptions(move_extras=False))
    assert off.moves == [] and off.skipped[0].reason == "extras_disabled"


def test_duplicates_renamed_in_place_continuing_past_folder_and_history(tmp_path):
    s = season(tmp_path)
    p = touch(s / "Disc 1" / "t00.mkv")
    d1 = touch(s / "Disc 2" / "t03.mkv")
    d2 = touch(s / "Disc 2" / "t04.mkv")
    touch(s / "Disc 2" / "t04.en.srt")
    touch(s / "Disc 2" / "synth_show_s01e01_dupe_2.mkv")  # an earlier copy, already named
    touch(s / "Disc 2" / "synth_show_s01e05_dupe_9.mkv")  # another episode: not counted
    rows = [row(p, 1), row(d2, 1, dup_of=p, rank=0, flags=["duplicate"]),
            row(d1, 1, dup_of=p, rank=1, flags=["duplicate"])]
    plan = plan_for(*rows)
    assert plan.ok
    assert moves(plan) == [
        ("episode", "t00.mkv", str(s / "synth_show_s01e01.mkv")),
        ("duplicate", "t04.mkv", str(s / "Disc 2" / "synth_show_s01e01_dupe_3.mkv")),
        ("sidecar", "t04.en.srt", str(s / "Disc 2" / "synth_show_s01e01_dupe_3.en.srt")),
        ("duplicate", "t03.mkv", str(s / "Disc 2" / "synth_show_s01e01_dupe_4.mkv")),
    ]
    assert [m.dupe_index for m in plan.moves] == [None, 3, None, 4]
    assert plan.moves[1].episode_id == "tmdb:100" and plan.moves[2].parent == 1

    db = StateDb(":memory:")
    h = History(db)
    run = h.begin_run("apply", "")
    db.conn.execute(
        "insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status, show_slug,"
        " episode_id, dupe_index) values (?,?,?,?,?,?,?,?,?)",
        (run.id, "x", "duplicate", "/a", "/b", "undone", SLUG, "tmdb:100", 7))
    db.conn.commit()
    plan = plan_for(*rows, history=h)
    assert [m.dupe_index for m in plan.moves if m.kind == "duplicate"] == [8, 9]


def test_duplicates_whose_primary_is_not_planned_or_is_an_extra(tmp_path):
    s = season(tmp_path)
    p = touch(s / "Disc 1" / "t00.mkv")
    d = touch(s / "Disc 1" / "t01.mkv")
    plan = plan_for(row(p, 1, level="medium"), row(d, 1, level="medium", dup_of=p, rank=0))
    assert plan.moves == []
    assert {k.reason for k in plan.skipped} == {"below_high", "duplicate_primary_missing"}
    plan = plan_for(row(p, None), row(d, None, dup_of=p, rank=0))
    assert [m.kind for m in plan.moves] == ["extra"]
    assert plan.skipped[0].reason == "duplicate_of_extra"
    # a primary already in place still names its copies
    named = touch(s / "synth_show_s01e01.mkv")
    plan = plan_for(row(named, 1, flags=["confirmed"]), row(d, 1, dup_of=named, rank=0))
    assert [m.new.name for m in plan.moves] == ["synth_show_s01e01_dupe_1.mkv"]
    assert plan.skipped[0].reason == "already_in_place"


def test_collisions_with_existing_and_planned_targets(tmp_path):
    s = season(tmp_path)
    existing = touch(s / "synth_show_s01e02.mkv", b"someone else's file")
    a = touch(s / "Disc 1" / "t00.mkv")
    x1 = touch(s / "Disc 1" / "t09.mkv")
    x2 = touch(s / "Disc 1__t09.mkv")  # no disc folder: keeps its name, which is x1's target
    touch(s / "extras" / "Disc 1__t09.mkv_unrelated")
    plan = plan_for(row(a, 2, fp="fa"), row(x1, None), row(x2, None))
    assert plan.ok
    got = {m.old: m.new for m in plan.moves}
    assert got[a] == s / "synth_show_s01e02_2.mkv"
    assert {got[x1], got[x2]} == {s / "extras" / "Disc 1__t09.mkv",
                                  s / "extras" / "Disc 1__t09_2.mkv"}
    assert sorted(m.collision_index or 0 for m in plan.moves) == [0, 2, 2]
    assert any("suffixed _2" in n for n in plan.notices)
    assert existing.read_bytes() == b"someone else's file"


def test_skipped_files_and_their_reasons(tmp_path):
    s = season(tmp_path)
    f = {n: touch(s / "Disc 1" / f"{n}.mkv") for n in (
        "named", "notready", "unread", "medium", "pinned", "oos", "oospin", "excl", "cx",
        "playall")}
    ign = touch(s / "Ignored" / "x.mkv")
    touch(s / "Ignored" / ".showvenist-ignore")
    db = StateDb(":memory:")
    h = History(db)
    h.touch_file(fingerprint(f["cx"]), size=1, path=str(f["cx"]))
    h.confirm(fingerprint(f["cx"]), SLUG, episode_id=None, season=None, episode=None,
              title=None, source="manual", run_id=None)
    plan = plan_for(
        row(f["named"], 1, flags=["named"]),
        row(f["notready"], None, level="low", flags=["not_ready"]),
        row(f["unread"], None, level="low", flags=["unreadable"]),
        row(f["medium"], 2, level="medium"),
        row(f["pinned"], 3, level="low", flags=["pinned"]),
        row(f["oos"], 4, flags=["out_of_scope"]),
        row(f["oospin"], 5, level="medium", flags=["pinned", "out_of_scope"]),
        row(f["excl"], 6),
        row(f["cx"], None),
        row(f["playall"], 1, flags=["play_all"]),
        row(ign, None),
        history=h, opts=PlanOptions(exclude={f["excl"]}),
    )
    assert plan.ok
    # a confirmed extra outside extras/ is moved like any extra (sv-dsk), its match linked
    assert {m.old.name: m.new.name for m in plan.moves} == {
        "pinned.mkv": "synth_show_s01e03.mkv", "oospin.mkv": "synth_show_s01e05.mkv",
        "cx.mkv": "Disc 1__cx.mkv"}
    assert all(m.pinned for m in plan.moves if m.kind == "episode")
    cx = next(m for m in plan.moves if m.old.name == "cx.mkv")
    assert cx.kind == "extra" and cx.confirmed and "[confirmed]" in cx.reason
    assert {k.path.name: k.reason for k in plan.skipped} == {
        "named.mkv": "named", "notready.mkv": "not_ready", "unread.mkv": "unreadable",
        "medium.mkv": "below_high", "oos.mkv": "out_of_scope", "excl.mkv": "excluded",
        "playall.mkv": "below_high", "x.mkv": "ignored"}


def test_season_change_needs_a_pin_and_an_existing_folder(tmp_path):
    b = make_bundle()
    two = Ordering(source="tmdb", type="dvd", entries=[
        OrderingEntry(season=2, episode=1, episode_id="tmdb:100")])
    b = b.model_copy(update={"orderings": {**b.orderings, "dvd": two}})
    f = touch(season(tmp_path) / "Disc 1" / "t00.mkv")
    opts = PlanOptions(naming_order="dvd")
    plan = plan_for(row(f, 1), bundle=b, opts=opts)
    assert plan.moves == [] and plan.skipped[0].reason == "out_of_scope"
    plan = plan_for(row(f, 1, flags=["pinned"]), bundle=b, opts=opts)
    assert plan.moves == [] and "no folder for season 2" in plan.skipped[0].detail
    (tmp_path / "Synth Show" / "Season 02").mkdir()
    plan = plan_for(row(f, 1, flags=["pinned"]), bundle=b, opts=opts)
    assert plan.moves[0].new == tmp_path / "Synth Show" / "Season 02" / "synth_show_s02e01.mkv"
    assert any("another season folder" in n for n in plan.notices)


def test_preflight_permission_missing_source_and_cross_filesystem(tmp_path):
    s = season(tmp_path)
    a = touch(s / "Disc 1" / "t00.mkv")
    x = touch(s / "Disc 1" / "t01.mkv")
    touch(s / "Disc 1" / "t01.srt")
    plan = plan_for(row(a, 1), row(x, None), writable=lambda d: d != s / "Disc 1")
    assert not plan.ok and any("no write permission" in e and "Disc 1" in e
                               for e in plan.errors)
    # extras/ does not exist yet: its nearest existing folder (the season) must be writable
    plan = plan_for(row(a, 1), row(x, None), writable=lambda d: d != s)
    assert not plan.ok and all(str(s) in e for e in plan.errors)
    gone = s / "Disc 1" / "gone.mkv"
    plan = plan_for(row(gone, 2, fp="g"))
    assert any("the file is gone" in e for e in plan.errors)
    plan = plan_for(row(a, 1), row(x, None),
                    device=lambda p: 2 if "extras" in str(p) or p == s / "extras" else 1,
                    exists=lambda p: p != s / "extras" and os.path.exists(p))
    # extras/ does not exist: its nearest existing ancestor is the season folder (device 1)
    assert plan.ok and plan.cross_filesystem == []
    plan = plan_for(row(a, 1), row(x, None), device=lambda p: 2 if p == s else 1)
    assert plan.ok
    assert [m.kind for m in plan.cross_filesystem] == ["episode", "extra", "sidecar"]
    assert any("cross filesystems" in n for n in plan.notices)


def test_relative_paths_become_absolute(tmp_path, monkeypatch):
    f = touch(season(tmp_path) / "Disc 1" / "t00.mkv")
    monkeypatch.chdir(tmp_path)
    plan = plan_for(row(Path(os.path.relpath(f)), 1, fp=fingerprint(f)))
    assert plan.moves[0].old == f and plan.moves[0].new.is_absolute()
