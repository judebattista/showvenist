"""Title-card wiring (sv-c4f): Session, `inspect --card` and the eval label-vs-card Problem,
on the tiny synthetic card video from test_cardframes (white 'EPISODE XXVII' on black)."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from showvenist import cli
from showvenist.analysis import cardframes
from showvenist.analysis.types import AnalysisCache
from showvenist.bundle.schema import (
    Bundle,
    Episode,
    Ordering,
    OrderingEntry,
    ShowInfo,
    TitleCardConfig,
)
from showvenist.bundle.store import bundle_path, save_bundle
from showvenist.config import load_config
from showvenist.eval import runner as eval_runner
from showvenist.eval.runner import run_eval
from showvenist.eval.types import ManifestRow
from showvenist.session import Session
from tests.test_cardframes import DUR, video  # noqa: F401  (module fixture)

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")

TITLES = ["XXV", "XXVI", "XXVII", "XXVIII"]
CARD = {"windows": [{"from": "end", "skip_seconds": 0, "length_seconds": 12}],
        "pattern": r"EPISODE\s+(?P<ref>[IVXLC]+)", "kind": "roman", "background": "dark"}
OVERRIDES = """title_card:
  windows:
    - { from: end, skip_seconds: 0, length_seconds: 12 }
  pattern: 'EPISODE\\s+(?P<ref>[IVXLC]+)'
  kind: roman
  background: dark
"""
runner = CliRunner()


def make_bundle(card=True) -> Bundle:
    eps = [Episode(id=f"tmdb:{k}", title=t, runtime_min=1) for k, t in enumerate(TITLES)]
    entries = [OrderingEntry(season=1, episode=k + 1, episode_id=e.id) for k, e in enumerate(eps)]
    b = Bundle(show=ShowInfo(title="Synth Show", year=2000), episodes=eps,
               orderings={"aired": Ordering(source="tmdb", type="aired", entries=entries)})
    if card:
        b = b.model_copy(update={"title_card": TitleCardConfig.model_validate(CARD)})
    return b.with_revision()


def count_grabs(monkeypatch):
    calls = []
    real = cardframes._grab

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(cardframes, "_grab", spy)
    return calls


# ---- Session ------------------------------------------------------------------------------------


def session(bundle, cache, path):
    s = Session(bundle, None, load_config(None, env={}), season=1, cache=cache)
    s.analyze([path])
    return s


def test_session_reads_cards_and_scores(video, tmp_path):  # noqa: F811
    s = session(make_bundle(), AnalysisCache(tmp_path / "a.db"), video)
    f = s.files[0]
    assert f.card_reads and any("XXVII" in r.text for r in f.card_reads)
    res = s.score()["titlecard"]
    assert res.available[0] and res.llr[0, 2] == 8.0 and res.llr[0].sum() == 8.0
    rep = s.report()
    sig = {x.name: x for x in rep.files[0].signals}
    assert sig["titlecard"].winner == 8.0 or rep.files[0].decision.episode.episode != 3


def test_session_without_config_never_reads_cards(video, tmp_path, monkeypatch):  # noqa: F811
    def boom(*a, **k):
        raise AssertionError("read_cards called")

    monkeypatch.setattr("showvenist.session.read_cards", boom)
    s = session(make_bundle(card=False), AnalysisCache(tmp_path / "a.db"), video)
    assert s.files[0].card_reads is None
    assert not s.score()["titlecard"].available.any()


def test_session_card_failure_does_not_fail_analysis(video, tmp_path, monkeypatch):  # noqa: F811
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr("showvenist.session.read_cards", boom)
    s = session(make_bundle(), AnalysisCache(tmp_path / "a.db"), video)
    assert s.files[0].card_reads == [] and not s.errors


# ---- inspect --card -----------------------------------------------------------------------------


def cli_setup(tmp_path, monkeypatch, config=True):
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    root = tmp_path / "bundles"
    d = bundle_path("synth-show-2000", root)
    save_bundle(make_bundle(card=False), root)
    if config:
        (d / "overrides.yaml").write_text(OVERRIDES)
    return root


def test_inspect_card_prints_text_then_cache(video, tmp_path, monkeypatch):  # noqa: F811
    root = cli_setup(tmp_path, monkeypatch)
    calls = count_grabs(monkeypatch)
    args = ["inspect", str(video), "--card", "--show", "synth-show-2000", "-o", str(root)]
    r = runner.invoke(cli.app, args)
    assert r.exit_code == 0, r.output
    assert "EPISODE" in r.output and "XXVII" in r.output
    r2 = runner.invoke(cli.app, args)
    assert r2.exit_code == 0 and "XXVII" in r2.output and len(calls) == 1


def test_inspect_card_adhoc_windows(video, tmp_path, monkeypatch):  # noqa: F811
    cli_setup(tmp_path, monkeypatch, config=False)
    calls = count_grabs(monkeypatch)
    args = ["inspect", str(video), "--card", "--from", "end", "--skip", "0", "--length", "12"]
    r = runner.invoke(cli.app, args)
    assert r.exit_code == 0, r.output
    assert "XXVII" in r.output and "background:   any" in r.output
    runner.invoke(cli.app, args)
    assert len(calls) == 1  # ad-hoc windows cache under their own key
    r = runner.invoke(cli.app, [*args, "--dark"])
    assert "background:   dark" in r.output and "XXVII" in r.output


def test_inspect_card_errors(video, tmp_path, monkeypatch):  # noqa: F811
    root = cli_setup(tmp_path, monkeypatch, config=False)
    r = runner.invoke(cli.app, ["inspect", str(video), "--card"])
    assert r.exit_code == 1 and "no title_card configured" in r.output
    r = runner.invoke(cli.app, ["inspect", str(video), "--card", "--show", "synth-show-2000",
                                "-o", str(root)])
    assert r.exit_code == 1 and "no title_card configured" in r.output
    r = runner.invoke(cli.app, ["inspect", str(video), "--card", "--from", "end"])
    assert r.exit_code == 1 and "go together" in r.output


# ---- eval ---------------------------------------------------------------------------------------


def eval_run(rips: Path, label: str, bundle, cache):
    rows = [ManifestRow(str(rips / "x.mkv"), "synth-show-2000", label)]
    return run_eval(rows, {"synth-show-2000": bundle}, load_config(None, env={}),
                    {"synth-show-2000": None}, cache)


def test_eval_card_problem_and_metrics_unchanged(video, tmp_path, monkeypatch):  # noqa: F811
    rips = tmp_path / "Season 1"
    rips.mkdir()
    shutil.copy(video, rips / "x.mkv")
    cache = AnalysisCache(tmp_path / "a.db")
    _, res = eval_run(rips, "S01E01", make_bundle(), cache)
    msgs = [p.reason for p in res.problems]
    assert msgs == ["card says XXVII but the label is S01E01 (XXV)"]
    # the audit adds a Problem and nothing else: same metrics without it
    monkeypatch.setattr(eval_runner, "_card_problems", lambda *a: [])
    _, plain = eval_run(rips, "S01E01", make_bundle(), cache)
    assert plain.problems == [] and res.overall == plain.overall
    assert res.per_season == plain.per_season
    monkeypatch.undo()
    # matching label, extras and `episode?` give no Problem
    for label in ("S01E03", "extra", "episode?"):
        _, ok = eval_run(rips, label, make_bundle(), cache)
        assert ok.problems == [], label
