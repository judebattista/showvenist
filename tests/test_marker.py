from pathlib import Path

import pytest
from synth import make_bundle

from showvenist.bundle.schema import Ordering, OrderingEntry
from showvenist.bundle.store import save_bundle
from showvenist.hierarchy.marker import (
    ShowResolutionError,
    jellyfin_hint,
    named_episodes,
    read_marker,
    resolve_show,
    suggest_folder_name,
    write_marker,
)
from showvenist.hierarchy.types import FileNode, Marker, NamedEpisode


def test_marker_roundtrip_and_shape(tmp_path):
    m = Marker(show="Firefly", tmdb_id=1437, bundle="firefly-2002", naming_order="intended-order")
    p = write_marker(tmp_path, m)
    text = p.read_text()
    assert "# [defaults]" in text and "# include_specials = false" in text
    assert read_marker(tmp_path) == m
    assert not list(tmp_path.glob(".showvenist-*"))
    with pytest.raises(FileExistsError):
        write_marker(tmp_path, m)
    m2 = Marker(show='A "B"', include_specials=True)
    write_marker(tmp_path, m2, overwrite=True)
    assert read_marker(tmp_path) == m2


def test_marker_missing_and_bad(tmp_path):
    assert read_marker(tmp_path) is None
    f = tmp_path / ".showvenist.toml"
    f.write_text('show = "X"\nbogus = 1\n')
    with pytest.raises(ValueError, match=r"\.showvenist\.toml.*bogus"):
        read_marker(tmp_path)
    f.write_text('show = "X"\ntmdb_id = "12"\n')
    with pytest.raises(ValueError, match="tmdb_id"):
        read_marker(tmp_path)
    f.write_text('show = "X"\n[defaults]\ninclude_specials = "yes"\n')
    with pytest.raises(ValueError, match="include_specials"):
        read_marker(tmp_path)
    f.write_text("show = ")
    with pytest.raises(ValueError, match="invalid TOML"):
        read_marker(tmp_path)


def test_jellyfin():
    assert jellyfin_hint("Firefly (2002) [tmdbid-1437]") == 1437
    assert jellyfin_hint("Firefly [tmdbid=1437]") == 1437
    assert jellyfin_hint("Firefly (2002)") is None
    assert suggest_folder_name("Firefly", 2002, 1437) == "Firefly (2002) [tmdbid-1437]"


def _bundles(tmp_path, *specs):
    root = tmp_path / "bundles"
    for title, year, tid in specs:
        b = make_bundle()
        b.show.title, b.show.year, b.show.ids.tmdb = title, year, tid
        save_bundle(b, root)
    return root


def test_resolve_order(tmp_path):
    root = _bundles(tmp_path, ("The West Wing", 1999, 10), ("Firefly", 2002, 1437))
    d = tmp_path / "Whatever [tmdbid-1437]"
    d.mkdir()
    assert resolve_show(d, root, interactive=False).source == "tmdbid-hint"
    assert resolve_show(d, root, interactive=False).slug == "firefly-2002"
    c = resolve_show("The_West_Wing", root, interactive=False)
    assert (c.slug, c.source) == ("the-west-wing-1999", "name")
    assert resolve_show("Firefly (2002)", root, interactive=False).slug == "firefly-2002"
    write_marker(d, Marker(show="TWW", bundle="the-west-wing-1999"))
    assert resolve_show(d, root, interactive=False).source == "marker"
    assert resolve_show(d, root, interactive=False).slug == "the-west-wing-1999"
    assert resolve_show("x", root, interactive=False, tmdb_hint=10).slug == "the-west-wing-1999"


def test_resolve_ambiguous_and_none(tmp_path):
    root = _bundles(tmp_path, ("Battlestar Galactica", 1978, 1), ("Battlestar Galactica", 2004, 2))
    with pytest.raises(ShowResolutionError, match="--show"):
        resolve_show("Battlestar Galactica", root, interactive=False)
    c = resolve_show("Battlestar Galactica", root, interactive=True, pick=lambda c: c[1])
    assert (c.slug, c.source) == ("battlestar-galactica-2004", "pick")
    with pytest.raises(ShowResolutionError, match="ingest"):
        resolve_show("Nothing", root, interactive=True, pick=lambda c: c[0])


def _f(name, s, e, suffix=""):
    return FileNode(Path(name), 1, 0.0, named=NamedEpisode(s, e, suffix))


def test_named_episodes():
    b = make_bundle(n=11)
    files = [_f("a.mkv", 1, 2), _f("b.mkv", 1, 14), _f("c.mkv", 3, 1),
             _f("d.mkv", 1, 5), _f("e.mkv", 1, 5, "a"), FileNode(Path("z.mkv"), 1, 0.0)]
    r = named_episodes(files, b, "aired")
    assert r.resolved[Path("a.mkv")] == "tmdb:101"
    assert "11 episodes" in r.unresolved[Path("b.mkv")]
    assert "no season 3" in r.unresolved[Path("c.mkv")]
    assert r.duplicate_sets == [[Path("d.mkv"), Path("e.mkv")]]
    assert Path("z.mkv") not in r.resolved and Path("z.mkv") not in r.unresolved


def test_named_episodes_ordering_keyword():
    b = make_bundle(n=3)
    b.orderings["dvd-1"] = Ordering(source="tmdb", type="dvd", type_rank=1, entries=[
        OrderingEntry(season=1, episode=1, episode_id="tmdb:102")])
    b.orderings["dvd-2"] = Ordering(source="tmdb", type="dvd", type_rank=2, entries=[
        OrderingEntry(season=1, episode=1, episode_id="tmdb:100")])
    r = named_episodes([_f("a.mkv", 1, 1)], b, "dvd")
    assert r.resolved[Path("a.mkv")] == "tmdb:102"
    r = named_episodes([_f("a.mkv", 1, 1)], b, "nope")
    assert "unknown ordering" in r.unresolved[Path("a.mkv")]
