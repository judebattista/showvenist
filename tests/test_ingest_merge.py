"""ingest with sources (tmdb primary, tvmaze secondary), refresh flow, config. No network."""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from showvenist import cli
from showvenist.adapters.tvmaze import TvmazeError
from showvenist.bundle import ingest as ing
from showvenist.bundle.quality import quality_report, render
from showvenist.bundle.schema import PartialBundle, ShowIds, ShowInfo
from showvenist.bundle.store import bundle_path, load_bundle
from showvenist.config import Config, load_config
from tests.test_merge import p_ep, primary, s_ep, secondary

runner = CliRunner()


class FakeTmdb:
    def __init__(self, eps):
        self.eps = eps

    def fetch(self, tid):
        b = primary(copy.deepcopy(self.eps))
        return PartialBundle(show=b.show, episodes=b.episodes, orderings=b.orderings,
                             provenance=b.provenance)


class FakeTvmaze:
    def __init__(self, eps, cands=None, search_error=None, fetch_error=None):
        self.eps, self.search_error, self.fetch_error = eps, search_error, fetch_error
        self.cands = cands if cands is not None else [
            {"tvmaze_id": 77, "title": "Testshow", "year": 2010, "status": "Ended"}]
        self.searched = self.fetched = 0

    def search(self, title, year=None):
        self.searched += 1
        if self.search_error:
            raise self.search_error
        return self.cands

    def fetch(self, tid):
        self.fetched += 1
        if self.fetch_error:
            raise self.fetch_error
        sec = secondary(copy.deepcopy(self.eps))
        sec.show = ShowInfo(title="Testshow", ids=ShowIds(tvmaze=tid))
        return sec


EPS = [(1, 1, p_ep(0)), (1, 2, p_ep(1)), (1, 3, p_ep(2))]
SEPS = [(1, 1, s_ep(0)), (1, 2, s_ep(1)), (1, 3, s_ep(2))]
KW = {"tmdb_id": 1}


def test_config_sources_default_env_and_validation(tmp_path):
    assert Config().ingest.sources == ["tmdb", "tvmaze"]
    toml = tmp_path / "c.toml"
    toml.write_text('[ingest]\nsources = ["tmdb"]\n')
    assert load_config(toml, env={}).ingest.sources == ["tmdb"]
    assert load_config(toml, env={"SHOWVENIST_INGEST_SOURCES": "tmdb, tvmaze"}
                       ).ingest.sources == ["tmdb", "tvmaze"]
    toml.write_text('[ingest]\nsources = ["tmdb", "imdb"]\n')
    with pytest.raises(ValidationError, match="unknown ingest source 'imdb'"):
        load_config(toml, env={})
    with pytest.raises(ValidationError, match="unknown ingest source"):
        load_config(tmp_path / "none.toml", env={"SHOWVENIST_INGEST_SOURCES": "tmdb,nope"})
    with pytest.raises(ValidationError, match="primary"):
        load_config(tmp_path / "none.toml", env={"SHOWVENIST_INGEST_SOURCES": "tvmaze"})


def test_merged_ingest_records_tvmaze_id(tmp_path):
    tv = FakeTvmaze(SEPS)
    b, d, scan, warns = ing.ingest(FakeTmdb(EPS), "Testshow", root=tmp_path,
                                   sources=("tmdb", "tvmaze"), tvmaze=tv, **KW)
    assert warns == [] and scan is None
    assert b.show.ids.tvmaze == 77 and b.show.ids.tmdb == 1
    assert {s.source for s in b.episodes[0].synopses} == {"tmdb", "tvmaze"}
    assert load_bundle("testshow-2010", tmp_path, apply_overrides=False).show.ids.tvmaze == 77
    assert tv.searched == 1


def test_tmdb_only_never_touches_tvmaze(tmp_path):
    tv = FakeTvmaze(SEPS)
    b, *_, warns = ing.ingest(FakeTmdb(EPS), "Testshow", root=tmp_path, sources=("tmdb",),
                               tvmaze=tv, **KW)
    assert tv.searched == tv.fetched == 0 and b.show.ids.tvmaze is None and warns == []


@pytest.mark.parametrize("tv", [
    FakeTvmaze(SEPS, cands=[]),
    FakeTvmaze(SEPS, cands=[{"tvmaze_id": 1, "title": "Testshow", "year": 2010},
                            {"tvmaze_id": 2, "title": "testshow", "year": 2010}]),
    FakeTvmaze(SEPS, cands=[{"tvmaze_id": 1, "title": "Testshow 2", "year": 2010}]),
    FakeTvmaze(SEPS, cands=[{"tvmaze_id": 1, "title": "Testshow", "year": 1999}]),
    FakeTvmaze(SEPS, search_error=TvmazeError("GET /search/shows failed: ConnectError")),
    FakeTvmaze(SEPS, fetch_error=TvmazeError("GET /shows/77 returned HTTP 500")),
], ids=["none", "two-exact", "other-title", "other-year", "search-fails", "fetch-fails"])
def test_secondary_problems_are_warnings(tmp_path, tv):
    b, d, _, warns = ing.ingest(FakeTmdb(EPS), "Testshow", root=tmp_path,
                                sources=("tmdb", "tvmaze"), tvmaze=tv, **KW)
    assert len(warns) == 1 and warns[0].startswith("tvmaze:")
    assert (d / "show.json").exists() and b.show.ids.tvmaze is None
    assert ("ambiguous or not found" in warns[0]) == (tv.search_error is None
                                                      and tv.fetch_error is None)


def test_ref_routes_to_tvmaze_and_skips_search(tmp_path):
    tv = FakeTvmaze(SEPS, cands=[])
    b, *_ = ing.ingest(FakeTmdb(EPS), "Testshow", root=tmp_path, sources=("tmdb", "tvmaze"),
                       tvmaze=tv, refs=("https://www.tvmaze.com/shows/77/testshow",), **KW)
    assert tv.searched == 0 and b.show.ids.tvmaze == 77
    assert ing.resolve_tvmaze_id(tv, "x", None, ("tvmaze:5",)) == 5


def test_tmdb_failure_is_an_error(tmp_path):
    class Boom(FakeTmdb):
        def fetch(self, tid):
            return PartialBundle()  # no show

    with pytest.raises(ing.IngestError):
        ing.ingest(Boom(EPS), "Testshow", root=tmp_path, sources=("tmdb", "tvmaze"),
                   tvmaze=FakeTvmaze(SEPS), **KW)


def test_refresh_reuses_known_tvmaze_id_and_carries_subtitles(tmp_path):
    ing.ingest(FakeTmdb(EPS), "Testshow", root=tmp_path, sources=("tmdb", "tvmaze"),
               tvmaze=FakeTvmaze(SEPS), **KW)
    p = bundle_path("testshow-2010", tmp_path) / "show.json"
    data = json.loads(p.read_text())
    data["episodes"][0]["subtitles"] = [{"lang": "en", "path": "subs/x.en.srt", "source": "m"}]
    p.write_text(json.dumps(data))
    tv = FakeTvmaze(SEPS, cands=[])
    built = ing.build(FakeTmdb(EPS), "Testshow", root=tmp_path, refresh=True,
                      sources=("tmdb", "tvmaze"), tvmaze=tv, **KW)
    assert tv.searched == 0 and built.warnings == [] and built.existing is not None
    assert built.bundle.episodes[0].subtitles[0].path == "subs/x.en.srt"


def test_existing_without_refresh_is_unchanged_error(tmp_path):
    ing.ingest(FakeTmdb(EPS), "Testshow", root=tmp_path, **KW)
    with pytest.raises(ing.IngestError, match="--refresh"):
        ing.ingest(FakeTmdb(EPS), "Testshow", root=tmp_path, **KW)


def test_subs_commit_installs_files(tmp_path):
    subs = tmp_path / "srt"
    subs.mkdir()
    (subs / "Testshow.S01E01.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
    root = tmp_path / "root"
    built = ing.build(FakeTmdb(EPS), "Testshow", root=root, subs_dir=subs, **KW)
    assert not bundle_path(built.slug, root).exists()  # build writes nothing
    assert built.bundle.episodes[0].subtitles
    d = ing.commit(built)
    assert (d / built.bundle.episodes[0].subtitles[0].path).is_file()


# --- CLI ------------------------------------------------------------------------------------


def _cli(monkeypatch, tmdb_eps, tv=None, interactive=False):
    monkeypatch.setattr(cli, "_tmdb", lambda: FakeTmdb(tmdb_eps))
    monkeypatch.setattr(cli, "_tvmaze", lambda: tv)
    monkeypatch.setattr(cli, "_interactive", lambda: interactive)


def run(tmp_path, *extra, **kw):
    return runner.invoke(cli.app, ["ingest", "Testshow", "--tmdb-id", "1", "-o", str(tmp_path),
                                   *extra], **kw)


def test_cli_first_ingest_prints_warning_and_conflicts(tmp_path, monkeypatch):
    split_eps = [(1, 1, p_ep(0)), (1, 2, p_ep(1, title="XXXVII"))]
    sec = [(1, 1, s_ep(0)),
           (1, 2, s_ep(1, title="The Glass Orchard (1)", runtime_min=22)),
           (1, 3, s_ep(8, title="The Glass Orchard (2)", runtime_min=22))]
    _cli(monkeypatch, split_eps, FakeTvmaze(sec))
    r = run(tmp_path, "--source", "tmdb,tvmaze")
    assert r.exit_code == 0, r.output
    assert "Unresolved conflicts" in r.output and "split" in r.output
    assert "bundle edit testshow-2010 --apply-fix 1" in r.output
    _cli(monkeypatch, EPS, FakeTvmaze(SEPS, cands=[]))
    r = run(tmp_path / "b", "--source", "tmdb,tvmaze")
    assert r.exit_code == 0 and "warning: tvmaze: ambiguous or not found" in r.output


def test_cli_unknown_source_is_error(tmp_path, monkeypatch):
    _cli(monkeypatch, EPS)
    r = run(tmp_path, "--source", "tmdb,imdb")
    assert r.exit_code == 1 and "unknown ingest source" in r.output
    r = run(tmp_path, "--source", "tmdb,wikipedia")
    assert r.exit_code == 1 and "not built yet" in r.output


NEW_EPS = [(1, 1, p_ep(0)), (1, 2, p_ep(1, title="Renamed Title")), (1, 3, p_ep(2)),
           (1, 4, p_ep(3))]


def test_cli_refresh_diff_writes_nothing(tmp_path, monkeypatch):
    _cli(monkeypatch, EPS)
    assert run(tmp_path).exit_code == 0
    p = bundle_path("testshow-2010", tmp_path)
    before = (p / "show.json").read_bytes()
    _cli(monkeypatch, NEW_EPS)
    r = run(tmp_path, "--refresh", "--diff")
    assert r.exit_code == 0, r.output
    assert "Episodes added (1)" in r.output and "Title changed (1)" in r.output
    assert (p / "show.json").read_bytes() == before and not (p / "show.json.prev").exists()


def test_cli_refresh_needs_yes_without_tty(tmp_path, monkeypatch):
    _cli(monkeypatch, EPS)
    assert run(tmp_path).exit_code == 0
    _cli(monkeypatch, NEW_EPS)
    r = run(tmp_path, "--refresh")
    assert r.exit_code == 1 and "--yes" in r.output
    assert len(load_bundle("testshow-2010", tmp_path).episodes) == 3


def test_cli_refresh_yes_writes_prev_keeps_overrides_warns_vanished(tmp_path, monkeypatch):
    _cli(monkeypatch, EPS)
    assert run(tmp_path).exit_code == 0
    p = bundle_path("testshow-2010", tmp_path)
    ov = p / "overrides.yaml"
    ov.write_text('episodes:\n  "tmdb:102":\n    runtime_min: 50\n')  # vanishes below
    old = (p / "show.json").read_bytes()
    _cli(monkeypatch, [e for e in EPS if e[2].id != "tmdb:102"])
    r = run(tmp_path, "--refresh", "--yes")
    assert r.exit_code == 0, r.output
    assert (p / "show.json.prev").read_bytes() == old
    assert ov.read_text().count("tmdb:102") == 1
    assert "unknown episode 'tmdb:102'" in r.output
    assert len(load_bundle("testshow-2010", tmp_path, apply_overrides=False).episodes) == 2
    # a second refresh replaces .prev (one generation)
    _cli(monkeypatch, EPS)
    new_old = (p / "show.json").read_bytes()
    assert run(tmp_path, "--refresh", "--yes").exit_code == 0
    assert (p / "show.json.prev").read_bytes() == new_old


def test_cli_refresh_tty_confirm(tmp_path, monkeypatch):
    _cli(monkeypatch, EPS)
    assert run(tmp_path).exit_code == 0
    p = bundle_path("testshow-2010", tmp_path)
    before = (p / "show.json").read_bytes()
    _cli(monkeypatch, NEW_EPS, interactive=True)
    r = run(tmp_path, "--refresh", input="\n")  # default is No
    assert r.exit_code == 0 and "Nothing written" in r.output
    assert (p / "show.json").read_bytes() == before
    r = run(tmp_path, "--refresh", input="y\n")
    assert r.exit_code == 0 and (p / "show.json.prev").exists()
    assert len(load_bundle("testshow-2010", tmp_path).episodes) == 4


def test_cli_diff_without_refresh_is_error(tmp_path, monkeypatch):
    _cli(monkeypatch, EPS)
    r = run(tmp_path, "--diff")
    assert r.exit_code == 1


def test_quality_lists_conflicts_per_season_with_numbers():
    from showvenist.bundle.schema import Provenance

    b = primary(EPS)
    b = b.model_copy(update={"provenance": [
        Provenance(adapter="tvmaze", conflict={"kind": "air_date", "season": "1",
                                               "primary": "tmdb:100", "secondary": "tvmaze:500",
                                               "detail": "dates differ"}),
        Provenance(adapter="tvmaze", conflict={"kind": "split", "season": "2", "detail": "two",
                                               "fix": "new_episodes: {}"}),
    ]})
    text = render(quality_report(b), "testshow-2010")
    assert "Season 1:" in text and "[1] air_date: dates differ" in text
    assert "Season 2:" in text and "[2] split: two" in text
    assert text.count("--apply-fix") == 1 and "--apply-fix 2" in text
