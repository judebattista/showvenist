import json

import pytest

from showvenist.bundle.schema import Bundle, Episode, ShowInfo
from showvenist.bundle.store import (
    BundleError,
    bundle_path,
    list_bundles,
    load_bundle,
    save_bundle,
    slugify,
)


def make() -> Bundle:
    return Bundle(
        show=ShowInfo(title="Fire Fly: Café!", year=2002),
        episodes=[Episode(id="tmdb:1", title="Serenity"), Episode(id="tmdb:2", title="Jaynestown")],
    )


def test_slugify():
    assert slugify("Fire Fly: Café!", 2002) == "fire-fly-cafe-2002"
    assert slugify("Show") == "show"
    assert slugify("!!!") == "show"


def test_roundtrip_and_revision(tmp_path):
    p = save_bundle(make(), tmp_path)
    assert p == bundle_path("fire-fly-cafe-2002", tmp_path) / "show.json"
    b = load_bundle("fire-fly-cafe-2002", tmp_path)
    assert b.bundle_revision.startswith("b3:")
    assert b.bundle_revision == b.compute_revision()
    assert b.episodes == make().episodes
    text = p.read_text()
    assert text.endswith("\n") and "\n  " in text
    assert list(tmp_path.glob("*/.*tmp")) == []
    save_bundle(b, tmp_path)
    assert p.read_text() == text


def test_list(tmp_path):
    assert list_bundles(tmp_path / "nope") == []
    save_bundle(make(), tmp_path)
    (tmp_path / "junk").mkdir()
    assert list_bundles(tmp_path) == ["fire-fly-cafe-2002"]


def test_errors(tmp_path):
    with pytest.raises(BundleError, match="not found"):
        load_bundle("x", tmp_path)
    d = tmp_path / "x"
    d.mkdir()
    (d / "show.json").write_text('{\n  "show": {\n')
    with pytest.raises(BundleError) as ei:
        load_bundle("x", tmp_path)
    assert str(ei.value).startswith(str(d / "show.json") + ":")
    assert ei.value.line is not None
    (d / "show.json").write_text(json.dumps({"show": {"title": 3}}))
    with pytest.raises(BundleError) as ei:
        load_bundle("x", tmp_path)
    assert "show.json" in str(ei.value) and "title" in str(ei.value)


def test_overrides_toggle(tmp_path):
    save_bundle(make(), tmp_path)
    d = bundle_path("fire-fly-cafe-2002", tmp_path)
    (d / "overrides.yaml").write_text("episodes:\n  tmdb:1:\n    title: Changed\n")
    assert load_bundle("fire-fly-cafe-2002", tmp_path).episodes[0].title == "Changed"
    raw = load_bundle("fire-fly-cafe-2002", tmp_path, apply_overrides=False)
    assert raw.episodes[0].title == "Serenity"
