"""Regression tests for the M1 review findings."""
import pytest
from typer.testing import CliRunner

from showvenist import cli
from showvenist.adapters.localsubs import scan_folder
from showvenist.bundle.overrides import apply_overrides
from showvenist.bundle.schema import (
    Bundle,
    Episode,
    Ordering,
    OrderingEntry,
    Provenance,
    ShowInfo,
    SubtitleRef,
    lang_codes,
    to_639_1,
)
from showvenist.bundle.store import bundle_path, load_bundle, save_bundle
from tests.test_tmdb import adapter

ENTRY = OrderingEntry(season=1, episode=1, episode_id="tmdb:1")


def _b():
    return Bundle(
        show=ShowInfo(title="S", year=2000),
        episodes=[Episode(id="tmdb:1", title="A")],
        orderings={"aired": Ordering(source="tmdb", type="aired", entries=[ENTRY])},
    )


def test_scalar_overrides_are_coerced(tmp_path):
    p = tmp_path / "overrides.yaml"
    p.write_text('episodes:\n  "tmdb:1":\n    excluded: "false"\n    runtime_min: "86"\n')
    ep = apply_overrides(_b(), p).episodes[0]
    assert ep.excluded is False and ep.runtime_min == 86


def test_subtitles_override_replaces_and_empty_removes(tmp_path):
    b = _b()
    b.episodes[0].subtitles = [SubtitleRef(lang="en", path="subs/x.srt", source="localsubs")]
    p = tmp_path / "overrides.yaml"
    p.write_text('episodes:\n  "tmdb:1":\n    subtitles: []\n')
    assert apply_overrides(b, p).episodes[0].subtitles == []


def test_override_cannot_replace_tmdb_ordering(tmp_path):
    p = tmp_path / "overrides.yaml"
    p.write_text("orderings:\n  aired:\n    - {season: 1, episode: 1, episode_id: 'tmdb:1'}\n")
    with pytest.raises(Exception, match="already exists"):
        apply_overrides(_b(), p)


def test_revision_changes_when_subtitle_content_changes(tmp_path):
    b = _b()
    b.episodes[0].subtitles = [SubtitleRef(lang="en", path="subs/a.srt", source="localsubs")]
    d = tmp_path / "s-2000"
    (d / "subs").mkdir(parents=True)
    (d / "subs" / "a.srt").write_text("one")
    r1 = b.compute_revision(d)
    (d / "subs" / "a.srt").write_text("two")
    assert b.compute_revision(d) != r1


def test_provenance_does_not_change_revision():
    a, b = _b(), _b()
    b.provenance = [Provenance(adapter="tmdb", fetched_at="2026-01-01")]
    assert a.compute_revision() == b.compute_revision()


def test_show_json_is_world_readable(tmp_path):
    assert save_bundle(_b(), tmp_path).stat().st_mode & 0o777 == 0o644


def test_language_codes():
    assert lang_codes("de") == {"de", "ger", "deu"}
    assert to_639_1("fre") == "fr" and to_639_1("fra") == "fr" and to_639_1("web") is None


def test_localsubs_language_suffix(tmp_path):
    (tmp_path / "Show.S01E01.WEB.srt").write_text("x")
    (tmp_path / "Show.S01E01.eng.srt").write_text("x")
    part, _ = scan_folder(tmp_path, _b(), lang="en")
    assert part.episodes[0].subtitles[0].lang == "en"


def test_ingest_refuses_overwrite_until_refresh(tmp_path, monkeypatch):
    runner = CliRunner()
    ad, _ = adapter()
    monkeypatch.setattr(cli, "_tmdb", lambda: ad)
    args = ["ingest", "Firefly", "--tmdb-id", "1437", "-o", str(tmp_path)]
    assert runner.invoke(cli.app, args).exit_code == 0
    again = runner.invoke(cli.app, args)
    assert again.exit_code == 1 and "--refresh" in again.output
    assert runner.invoke(cli.app, [*args, "--refresh"]).exit_code == 0
    assert load_bundle("firefly-2002", tmp_path) and bundle_path("firefly-2002", tmp_path).exists()
