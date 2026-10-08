"""Title-card contract (spec §6.6, §8.4, §9.2, decision 54; tasks sv-bug, sv-8ue, sv-2k9)."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from showvenist.analysis.cardframes import card_params, read_cards
from showvenist.analysis.types import EXTRACTOR_VERSIONS, CardRead
from showvenist.bundle.overrides import BundleError, apply_overrides
from showvenist.bundle.schema import Bundle, Episode, ShowInfo, TitleCardConfig
from showvenist.bundle.store import save_bundle
from showvenist.config import WeightsConfig, load_config
from showvenist.fusion import FOR_ONLY
from showvenist.scoring.titlecard import MATCH_LLR, TitleCardScorer
from showvenist.scoring.types import (
    DEFAULT_CLAMPS,
    DEFAULT_WEIGHTS,
    REGIME_SCORERS,
    Candidate,
    FileInfo,
    ScoreContext,
)

GOOD = """
title_card:
  windows:
    - { from: end, skip_seconds: 0, length_seconds: 180 }
    - { from: start, skip_seconds: 120, length_seconds: 90 }
  pattern: 'EPISODE\\s+(?P<ref>[IVXLC]+)'
  kind: roman
"""


def bundle() -> Bundle:
    return Bundle(
        show=ShowInfo(title="S"), episodes=[Episode(id="tmdb:1", title="X")]
    ).with_revision()


def load(tmp_path, text, b=None):
    p = tmp_path / "overrides.yaml"
    p.write_text(text)
    return apply_overrides(b or bundle(), p)


def test_config_parses_from_overrides(tmp_path):
    cfg = load(tmp_path, GOOD).title_card
    assert cfg is not None and cfg.kind == "roman" and cfg.background == "dark"
    assert [(w.edge, w.skip_seconds, w.length_seconds) for w in cfg.windows] == [
        ("end", 0, 180), ("start", 120, 90)]


def test_absent_means_no_card(tmp_path):
    assert load(tmp_path, "episodes: {}\n").title_card is None


@pytest.mark.parametrize("bad", [
    "windows: []\n  pattern: 'a(?P<ref>x)'\n  kind: roman",                       # no windows
    "windows: [{from: end, length_seconds: 60}]\n  pattern: 'no group'\n  kind: roman",
    "windows: [{from: end, length_seconds: 60}]\n  pattern: '(unclosed'\n  kind: roman",
    "windows: [{from: middle, length_seconds: 60}]\n  pattern: '(?P<ref>x)'\n  kind: roman",
    "windows: [{from: end, length_seconds: 0}]\n  pattern: '(?P<ref>x)'\n  kind: roman",
    "windows: [{from: end, length_seconds: 60}]\n  pattern: '(?P<ref>x)'\n  kind: arabic",
])
def test_bad_config_names_file_and_line(tmp_path, bad):
    with pytest.raises(BundleError) as e:
        load(tmp_path, "title_card:\n  " + bad + "\n")
    assert "overrides.yaml" in str(e.value) and "title_card" in str(e.value)
    assert e.value.line == 2 or "line" in str(e.value)


def test_revision_unchanged_without_card_and_changes_with_one(tmp_path):
    b = bundle()
    assert load(tmp_path, "episodes: {}\n", b).bundle_revision == b.bundle_revision
    assert load(tmp_path, GOOD, b).bundle_revision != b.bundle_revision


def test_card_never_written_to_show_json(tmp_path):
    b = load(tmp_path, GOOD)
    saved = json.loads(save_bundle(b, tmp_path / "a").read_text())
    plain = json.loads(save_bundle(bundle(), tmp_path / "b").read_text())
    assert "title_card" not in saved
    assert saved["bundle_revision"] == plain["bundle_revision"]


def test_scorer_registered_in_both_regimes():
    assert "titlecard" in REGIME_SCORERS["base"] and "titlecard" in REGIME_SCORERS["llm"]
    assert DEFAULT_CLAMPS["titlecard"] == (0.0, 8.0) and DEFAULT_WEIGHTS["titlecard"] == 1.0
    assert "titlecard" in FOR_ONLY and TitleCardScorer.can_be_negative is False
    assert MATCH_LLR == 8.0


def test_weights_in_config():
    cfg = load_config(None, env={})
    assert cfg.weights.titlecard == 1.0 and cfg.weights.llm.titlecard == 1.0
    assert "titlecard" in WeightsConfig.model_fields


def test_card_reads_default_is_not_read():
    f = FileInfo(path="a", fingerprint="f", duration_s=1.0)
    assert f.card_reads is None
    assert CardRead(12.0, "EPISODE X").offset_s == 12.0


def test_stub_scorer_is_unavailable_everywhere():
    b = bundle()
    ctx = ScoreContext(b, [Candidate(0, b.episodes[0], 1, 1)], load_config(None, env={}))
    res = TitleCardScorer().score([FileInfo(path="a", fingerprint="f", duration_s=1.0)], ctx)
    assert not res.available.any() and np.count_nonzero(res.llr) == 0


def test_cache_params_depend_on_windows_and_background_not_on_pattern():
    base = TitleCardConfig.model_validate({
        "windows": [{"from": "end", "length_seconds": 180}],
        "pattern": "(?P<ref>x)", "kind": "roman"})
    other_pattern = base.model_copy(update={"pattern": "EP (?P<ref>y)"})
    other_bg = base.model_copy(update={"background": "any"})
    other_win = TitleCardConfig.model_validate({
        "windows": [{"from": "end", "length_seconds": 90}],
        "pattern": "(?P<ref>x)", "kind": "roman"})
    assert card_params(base) == card_params(other_pattern)
    assert len({card_params(base), card_params(other_bg), card_params(other_win)}) == 3
    assert EXTRACTOR_VERSIONS["cardframes"] == 1


def test_read_cards_is_a_stub_until_sv_8ue():
    with pytest.raises(NotImplementedError):
        read_cards(None, "f", 1.0, TitleCardConfig.model_validate({  # type: ignore[arg-type]
            "windows": [{"from": "end", "length_seconds": 60}],
            "pattern": "(?P<ref>x)", "kind": "roman"}))


def test_card_config_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        TitleCardConfig.model_validate({
            "windows": [{"from": "end", "length_seconds": 60}], "pattern": "(?P<ref>x)",
            "kind": "roman", "extra": 1})
