"""`.showvenist-ignore`: a folder holding that file is out of scope (never analyzed, matched,
renamed or moved, and never a parse problem)."""

from pathlib import Path

from showvenist.config import HierarchyConfig
from showvenist.hierarchy.parse import IGNORE_FILE, build_tree, is_ignored, walk_media


def make_show(tmp_path: Path, ignore: bool) -> Path:
    show = tmp_path / "Farscape"
    for d in ("Season01/disc1", "Peacekeeper_Wars"):
        (show / d).mkdir(parents=True)
    (show / "Season01/disc1/title_t00.mkv").write_bytes(b"x")
    (show / "Peacekeeper_Wars/Farscape- The Peacekeeper Wars_t00.mkv").write_bytes(b"x")
    if ignore:
        (show / "Peacekeeper_Wars" / IGNORE_FILE).write_text("")
    return show


def test_unparseable_folder_is_a_problem_until_it_is_ignored(tmp_path):
    show = make_show(tmp_path, ignore=False)
    t = build_tree(show, config=HierarchyConfig(), tag_of=lambda p: None)
    assert any("Peacekeeper_Wars" in p for p in t.problems)
    (show / "Peacekeeper_Wars" / IGNORE_FILE).write_text("")
    t = build_tree(show, config=HierarchyConfig(), tag_of=lambda p: None)
    assert not t.problems
    assert [p.name for p in t.ignored] == ["Peacekeeper_Wars"]
    assert [f.path.name for f in t.all_files()] == ["title_t00.mkv"]  # the miniseries is out


def test_walk_media_skips_ignored_folders_and_an_ignored_root(tmp_path):
    show = make_show(tmp_path, ignore=True)
    assert [p.name for p in walk_media(show)] == ["title_t00.mkv"]
    assert walk_media(show / "Peacekeeper_Wars") == []  # passed directly: still out of scope
    assert is_ignored(show / "Peacekeeper_Wars") and not is_ignored(show)
    assert walk_media(show / "Season01/disc1/title_t00.mkv") == [
        show / "Season01/disc1/title_t00.mkv"]  # an explicitly named file is the user's choice


def test_an_ignored_root_gives_an_empty_tree(tmp_path):
    show = make_show(tmp_path, ignore=True)
    t = build_tree(show / "Peacekeeper_Wars", config=HierarchyConfig(), tag_of=lambda p: None)
    assert t.ignored and not t.all_files() and not t.problems
