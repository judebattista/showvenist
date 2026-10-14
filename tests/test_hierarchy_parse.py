import os
from pathlib import Path

import pytest

from showvenist.config import HierarchyConfig
from showvenist.hierarchy.parse import (
    build_tree,
    disc_side,
    is_duplicate_disc_problem,
    mark_duplicate_discs,
    parse_disc,
    parse_named,
    parse_season,
    parse_tag_season,
    sides_order,
    suggest_disc_order,
)
from showvenist.hierarchy.types import DiscNode, DiscRef, NamedEpisode, SeasonNode
from tests.fixtures import hierarchy_cases as C


@pytest.mark.parametrize(("name", "want"), C.SEASONS)
def test_parse_season(name, want):
    assert parse_season(name) == want


@pytest.mark.parametrize(("name", "num", "season"), C.DISCS)
def test_parse_disc(name, num, season):
    ref = parse_disc(name)
    assert ref is not None and ref.number == num and ref.season == season
    assert ref.source == "folder" and ref.label


@pytest.mark.parametrize("name", C.DISCS_NO_NUMBER)
def test_disc_without_number(name):
    ref = parse_disc(name)
    assert ref is not None and ref.number is None


@pytest.mark.parametrize("name", C.NOT_DISCS)
def test_not_a_disc(name):
    assert parse_disc(name) is None


def test_pure_season_tag_invents_no_disc():
    assert parse_disc("Babylon 5 - Season 2") is None
    assert parse_tag_season("Babylon 5 - Season 2") == 2
    assert parse_tag_season("THE WEST WING COMPLETE SERIES S3D2") == 3
    assert parse_tag_season("FIREFLY: DISC 1") is None


@pytest.mark.parametrize(("name", "s", "e", "suffix"), C.NAMED)
def test_parse_named(name, s, e, suffix):
    assert parse_named(name) == NamedEpisode(s, e, suffix)


@pytest.mark.parametrize("name", C.NOT_NAMED)
def test_not_named(name):
    assert parse_named(name) is None


def test_custom_patterns():
    assert parse_season("Staffel 3", [r"staffel (\d+)"]) == 3
    assert parse_disc("Platte 2", [r"platte (\d+)"]).number == 2
    assert parse_disc("X2 D4", [r"x(\d) d(\d)"]) == DiscRef(4, 2, "X2 D4", "folder")
    assert parse_named("ep 2x05", [r"(\d+)x(\d+)"]) == NamedEpisode(2, 5)


def _mk(root: Path, files):
    tags = {}
    for rel, tag in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        tags[p] = tag or None
    return tags


def test_build_tree_show(tmp_path):
    tags = _mk(tmp_path, C.SYNTH_FILES)
    show = build_tree(tmp_path / "Supernatural", config=HierarchyConfig(), tag_of=tags.get)
    assert show.name == "Supernatural"
    assert [s.number for s in show.seasons] == [10, 11]
    s10, s11 = show.seasons
    assert [d.number for d in s10.discs] == [1, 2, 3]
    assert [len(d.files) for d in s10.discs] == [2, 1, 1]
    assert all(d.order_source == "number" for d in s10.discs)
    assert s10.discs[0].files[0].tag_disc.source == "tag"
    # unparseable folder: a problem, files kept loose, never guessed
    assert any("temp" in p for p in show.problems)
    assert [f.path.parent.name for f in s10.loose] == ["temp"]
    # folder wins; disagreement recorded
    assert s10.discs[2].number == 3
    assert any("disc3" in d and "season 9" in d for d in show.disagreements)
    # flattened files with a tag disc go to a 'tag:' DiscNode, named is filled
    (d11,) = s11.discs
    assert d11.key.startswith("tag:") and d11.folder is None and d11.number == 1
    assert d11.order_source == "number"
    assert d11.files[0].named == NamedEpisode(11, 1)
    assert show.all_files()[0].ready


def test_build_tree_letter_discs_and_creation_order(tmp_path):
    tags = _mk(tmp_path, C.SYNTH_FILES)
    show = build_tree(tmp_path / "The_White_Shadow", config=HierarchyConfig(), tag_of=tags.get)
    (s1,) = show.seasons
    assert s1.number == 1 and [d.key for d in s1.discs] == ["disc1a", "disc1b"]
    assert all(d.number is None and d.order_source == "name" for d in s1.discs)
    a, b = s1.discs
    a.created, b.created = 200.0, 100.0
    out = suggest_disc_order([a, b])
    assert [d.key for d in out] == ["disc1b", "disc1a"]
    assert all(d.order_source == "creation-time" for d in out)


def test_build_tree_flat_files_season_from_tag(tmp_path):
    files = [
        ("Babylon5/Babylon 5 - Season 2_t00.mkv", "Babylon 5 - Season 2"),
        ("Babylon5/Babylon 5 - Season 4_t00.mkv", "Babylon 5 - Season 4"),
        ("Babylon5/the_wire_s01e01.mkv", ""),
    ]
    tags = _mk(tmp_path, files)
    show = build_tree(tmp_path / "Babylon5", config=HierarchyConfig(), tag_of=tags.get)
    assert [s.number for s in show.seasons] == [2, 4, None]
    assert all(not s.discs and len(s.loose) == 1 for s in show.seasons)
    assert show.seasons[2].loose[0].named == NamedEpisode(1, 1)


def test_build_tree_season_and_disc_roots(tmp_path):
    tags = _mk(tmp_path, C.SYNTH_FILES)
    s = build_tree(tmp_path / "Supernatural/Season10", config=HierarchyConfig(), tag_of=tags.get)
    assert s.name is None and [x.number for x in s.seasons] == [10]
    d = build_tree(
        tmp_path / "Supernatural/Season10/disc2", config=HierarchyConfig(), tag_of=tags.get
    )
    assert d.seasons[0].discs[0].number == 2 and d.seasons[0].number is None


def test_build_tree_unparseable_show_level(tmp_path):
    tags = _mk(
        tmp_path, [("Farscape/Peacekeeper_Wars/a.mkv", ""), ("Farscape/Season01/disc1/b.mkv", "")]
    )
    show = build_tree(tmp_path / "Farscape", config=HierarchyConfig(), tag_of=tags.get)
    assert not show.problems
    assert [p.name for p in show.unrecognized] == ["Peacekeeper_Wars"]
    assert [s.number for s in show.seasons] == [1]
    assert [f.path.name for f in show.all_files()] == ["b.mkv"]  # the unrecognized folder is out


def test_duplicate_disc_numbers_reported(tmp_path):
    tags = _mk(tmp_path, [("S/Season1/disc1/a.mkv", ""), ("S/Season1/D1/b.mkv", ""),
                          ("S/Season1/disc2/c.mkv", "")])
    show = build_tree(tmp_path / "S", config=HierarchyConfig(), tag_of=tags.get)
    (dup,) = show.problems
    assert "twice" in dup and is_duplicate_disc_problem(dup)
    assert not is_duplicate_disc_problem("cannot parse disc folder 'x' in 'y'")
    # §5.1 never guesses: two 'disc 1's lose their number (the unnumbered-disc path:
    # creation-time order, confirmed by the user); disc 2 keeps its own
    (s1,) = show.seasons
    nums = {d.key: d.number for d in s1.discs}
    assert nums == {"D1": None, "disc1": None, "disc2": 2}
    assert all(d.ref.label for d in s1.discs)


def test_mark_duplicate_discs_flat_tags():
    a = DiscNode("tag:x", DiscRef(1, 1, "SHOW S1 DISC 1", "tag"), None)
    b = DiscNode("tag:y", DiscRef(1, 1, "SHOW S01 DISC 1", "tag"), None)
    node = SeasonNode(1, None, discs=[a, b])
    (msg,) = mark_duplicate_discs(node, "Season 1")
    assert "'tag:x', 'tag:y'" in msg and a.number is None and b.number is None
    assert mark_duplicate_discs(node, "Season 1") == []  # nothing left to mark


def test_side_lettered_discs_sort_by_number_then_side(tmp_path):
    files = [(f"WS/Season 1/{k}/t.mkv", "") for k in ("disc2a", "disc1b", "disc10a", "disc1a")]
    tags = _mk(tmp_path, files)
    show = build_tree(tmp_path / "WS", config=HierarchyConfig(), tag_of=tags.get)
    (s1,) = show.seasons
    # DiscRef has no side field (contract): number stays None, the side is a secondary key
    assert all(d.number is None for d in s1.discs)
    assert [d.key for d in s1.discs] == ["disc1a", "disc1b", "disc2a", "disc10a"]
    assert disc_side(s1.discs[1].ref) == (1, "b") and disc_side(DiscRef(1)) is None
    assert [d.key for d in sides_order(list(reversed(s1.discs)))] == [
        "disc1a", "disc1b", "disc2a", "disc10a"]
    assert all(d.order_source == "name" for d in s1.discs)
    # a disc with no side, or two identical sides, is not ordered this way
    assert sides_order([*s1.discs, DiscNode("Disc A", DiscRef(None, None, "Disc A"), None)]) \
        is None
    assert sides_order([s1.discs[0], DiscNode("x", s1.discs[0].ref, None)]) is None


def test_suggest_disc_order_without_created_keeps_key_order():
    a, b = DiscNode("b", None, None), DiscNode("a", None, None)
    assert [d.key for d in suggest_disc_order([a, b])] == ["a", "b"]


def test_stat_is_injectable(tmp_path):
    tags = _mk(tmp_path, [("S/Season1/disc1/a.mkv", "")])
    calls = []

    def fake(p):
        calls.append(p)
        return os.stat(p)

    build_tree(tmp_path / "S", config=HierarchyConfig(), tag_of=tags.get, stat=fake)
    assert calls
