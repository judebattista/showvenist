"""Weight fitting (spec §16.3): synthetic items from known weights, no media."""

from __future__ import annotations

import tomllib

import numpy as np
import pytest
from scipy.special import logsumexp

from showvenist.config import WeightsConfig, load_config
from showvenist.eval import fit
from showvenist.eval.fit import (
    FitItem,
    _build,
    _objective,
    fit_loso,
    fit_weights,
    format_fit_report,
    mean_log_loss,
    render_weights,
    to_weights_config,
    write_weights,
)
from showvenist.fusion import log_prior
from showvenist.scoring.types import DEFAULT_CLAMPS, PI_X_DEFAULT, REGIME_SCORERS

BASE = REGIME_SCORERS["base"]
TRUE_W = {
    "duration": 0.6,
    "title": 1.5,
    "refsubs": 0.5,
    "synopsis": 2.0,
    "guests": 1.2,
    "cues": 0.3,
}


def _raw(rng, E, signal_col):
    """Raw LLRs within the clamp ranges, with a boost on `signal_col` (or none if None)."""
    llr = {
        "duration": rng.normal(-1.0, 2.0, E),
        "title": np.where(rng.random(E) < 0.05, 6.0, 0.0),
        "refsubs": rng.normal(-1.0, 3.0, E),
        "synopsis": np.abs(rng.normal(0, 1.0, E)),
        "guests": np.abs(rng.normal(0, 1.5, E)) * (rng.random(E) < 0.3),
        "cues": np.abs(rng.normal(0, 1.5, E)),
    }
    if signal_col is not None:
        llr["refsubs"][signal_col] += 3.0
        llr["synopsis"][signal_col] += 1.0
    return llr


def _score(llr, w, E):
    S = log_prior(E)
    for k in BASE:
        lo, hi = DEFAULT_CLAMPS[k]
        S = S + w[k] * np.clip(llr[k], lo, hi)
    return S


def synth_items(n, w=TRUE_W, seed=0, seasons=4, E=20):
    """Items whose labels are sampled from the conditional-logit model with weights `w`."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        sig = int(rng.integers(E)) if rng.random() < 0.85 else None
        llr = _raw(rng, E, sig)
        S = _score(llr, w, E)
        full = np.append(S, 0.0)
        p = np.exp(full - logsumexp(full))
        y = int(rng.choice(E + 1, p=p))
        target = "extra" if y == E else "episode"
        items.append(
            FitItem(f"show-S{i % seasons + 1:02d}", "base", llr, E, target, None if y == E else y)
        )
    return items


def test_recovers_true_weights():
    items = synth_items(4000)
    w = fit_weights(items, "base", lam=0.01)
    for k in ("duration", "refsubs", "synopsis", "title"):
        assert w[k] == pytest.approx(TRUE_W[k], abs=0.2), (k, w)
    assert set(w) == set(BASE)


def test_ridge_pulls_toward_defaults():
    items = synth_items(400, seed=1)
    dist = []
    for lam in (0.0, 10.0, 100.0, 1e5):
        w = fit_weights(items, "base", lam=lam)
        dist.append(sum((v - 1.0) ** 2 for v in w.values()))
    assert dist == sorted(dist, reverse=True)
    assert dist[-1] < 1e-3
    # a custom default is the ridge target too
    w = fit_weights(items, "base", lam=1e6, defaults={"synopsis": 3.0})
    assert w["synopsis"] == pytest.approx(3.0, abs=1e-2)


def test_weights_bounded_nonnegative():
    # extras with strong positive cues evidence: the unconstrained fit would want w < 0
    rng = np.random.default_rng(2)
    items = [
        FitItem("a", "base", {"cues": np.full(5, 4.0) * rng.random(5)}, 5, "extra")
        for _ in range(50)
    ]
    w = fit_weights(items, "base", lam=0.0)
    assert w["cues"] >= 0.0 and w["cues"] < 1e-3


def test_gradient_matches_finite_differences():
    items = synth_items(60, seed=3)
    rng = np.random.default_rng(4)
    for _ in range(10):  # mix in set labels, extras and varying E
        E = int(rng.integers(3, 25))
        items.append(FitItem("x", "base", _raw(rng, E, 0), E, "any_episode"))
        items.append(FitItem("x", "base", _raw(rng, E, None), E, "extra"))
    d = _build(items, "base", DEFAULT_CLAMPS, PI_X_DEFAULT, 0.3)
    w0 = np.ones(len(BASE))
    w = rng.uniform(0.2, 2.0, len(BASE))
    f, g = _objective(w, d, 0.7, w0)
    eps = 1e-6
    num = np.array(
        [
            (_objective(w + eps * e, d, 0.7, w0)[0] - _objective(w - eps * e, d, 0.7, w0)[0])
            / (2 * eps)
            for e in np.eye(len(w))
        ]
    )
    np.testing.assert_allclose(g, num, rtol=1e-5, atol=1e-5)


def _episode_mass(items, w):
    out = []
    for it in items:
        S = log_prior(it.n_candidates) + w["synopsis"] * np.clip(it.llr["synopsis"], 0, 4)
        out.append(np.exp(logsumexp(S) - np.logaddexp(logsumexp(S), 0.0)))
    return float(np.mean(out))


def test_set_label_raises_episode_mass():
    rng = np.random.default_rng(5)
    # episode? files: diffuse synopsis evidence over two plausible episodes, no number known
    unknown = []
    for _ in range(80):
        s = np.zeros(20)
        s[rng.choice(20, 2, replace=False)] = 1.0
        unknown.append(FitItem("a", "base", {"synopsis": s}, 20, "any_episode"))
    w = fit_weights(unknown, "base", lam=0.1)
    d = {"synopsis": 1.0}
    assert w["synopsis"] > 1.0
    assert _episode_mass(unknown, w) > _episode_mass(unknown, d)
    # objective with an any_episode target is log Σ_j softmax mass of the episodes
    ll = mean_log_loss(unknown[:1], "base", d)
    assert ll == pytest.approx(-np.log(_episode_mass(unknown[:1], d)))


def test_extras_share_lowers_weight_of_misleading_signal():
    rng = np.random.default_rng(6)
    items = []
    for _ in range(200):  # episodes: synopsis points to the right one
        t = int(rng.integers(20))
        s = np.zeros(20)
        s[t] = 2.0
        items.append(FitItem("a", "base", {"synopsis": s}, 20, "episode", t))
    for _ in range(20):  # extras: synopsis fires on some random candidate
        s = np.zeros(20)
        s[int(rng.integers(20))] = 2.0
        items.append(FitItem("a", "base", {"synopsis": s}, 20, "extra"))
    lo = fit_weights(items, "base", lam=0.1, extras_share=0.05)["synopsis"]
    none = fit_weights(items, "base", lam=0.1)["synopsis"]
    hi = fit_weights(items, "base", lam=0.1, extras_share=0.5)["synopsis"]
    assert lo > none > hi
    d = _build(items, "base", DEFAULT_CLAMPS, PI_X_DEFAULT, 0.5)
    assert d.c.sum() == pytest.approx(len(items))
    assert d.c[d.kind == 1].sum() == pytest.approx(0.5 * len(items))
    with pytest.raises(ValueError):
        fit_weights(items[:200], "base", extras_share=0.2)  # no extras to weight up


def test_only_regime_scorers_and_items():
    rng = np.random.default_rng(7)
    items = [
        FitItem("a", "llm", {"llm": rng.normal(0, 2, 8), "synopsis": np.ones(8)}, 8, "episode", 0)
        for _ in range(20)
    ]
    items.append(FitItem("a", "base", {"synopsis": np.ones(8)}, 8, "episode", 0))
    w = fit_weights(items, "llm")
    assert set(w) == set(REGIME_SCORERS["llm"])
    with pytest.raises(ValueError):
        fit_weights(items[:20], "base")
    with pytest.raises(ValueError):
        fit_weights([FitItem("a", "base", {"title": np.ones(3)}, 4, "episode", 0)], "base")
    with pytest.raises(ValueError):
        fit_weights([FitItem("a", "base", {}, 4, "episode", 4)], "base")


def test_loso_never_uses_held_out_season(monkeypatch):
    items = synth_items(400, seed=8, seasons=4)
    calls = []
    real = fit.fit_weights

    def spy(its, regime, **kw):
        calls.append({it.season_key for it in its})
        return real(its, regime, **kw)

    monkeypatch.setattr(fit, "fit_weights", spy)
    res = fit_loso(items, "base", lam=1.0)
    seasons = {it.season_key for it in items}
    assert len(res.folds) == 4 and len(calls) == 5
    for fold, used in zip(res.folds, calls[:4], strict=True):
        assert fold.season_key not in used
        assert used == seasons - {fold.season_key}
    assert calls[4] == seasons  # in-sample fit on everything
    assert res.pooled_fitted < res.pooled_default  # true weights differ from 1.0
    assert sum(f.n_items for f in res.folds) == len(items)
    text = format_fit_report(res)
    assert "pooled" in text and "synopsis" in text and "show-S01" in text
    with pytest.raises(ValueError):
        fit_loso([it for it in items if it.season_key == "show-S01"], "base")


def test_loso_extras_share_tolerates_fold_without_extras():
    items = synth_items(200, seed=9, seasons=2)
    items = [it for it in items if not (it.season_key == "show-S01" and it.target == "extra")]
    res = fit_loso(items, "base", extras_share=0.2)
    assert len(res.folds) == 2


CONFIG = """\
# my config
[tmdb]
api_key = "abc"   # keep me

[weights]                # written by `eval --fit`; defaults built in
# duration = 1.0 …
duration = 0.5  # old
cues = 9.0

[weights.llm]            # files the LLM has scored (§11.1)
llm = 2.0

[llm]
mode = "off"
"""

BASE_W = {
    "duration": 0.81234,
    "title": 1.0,
    "refsubs": 1.4,
    "synopsis": 2.5,
    "guests": 1.3,
    "cues": 0.0,
}
LLM_W = {"duration": 0.7, "title": 1.1, "refsubs": 1.2, "llm": 1.45}


def test_write_weights_preserves_other_content(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(CONFIG)
    write_weights(p, BASE_W, LLM_W)
    text = p.read_text()
    for keep in (
        "# my config",
        'api_key = "abc"   # keep me',
        "# duration = 1.0 …",
        "# written by `eval --fit`; defaults built in",
        '[llm]\nmode = "off"',
        "# files the LLM has scored (§11.1)",
    ):
        assert keep in text
    assert "duration = 0.8123  # old" in text
    assert text.count("[weights]") == 1 and text.count("[weights.llm]") == 1
    raw = tomllib.loads(text)
    assert raw["tmdb"] == {"api_key": "abc"} and raw["llm"] == {"mode": "off"}
    assert raw["weights"]["cues"] == 0.0 and raw["weights"]["llm"]["llm"] == 1.45
    cfg = load_config(p, env={})
    assert cfg.weights == to_weights_config(BASE_W, LLM_W).model_copy(update={"duration": 0.8123})
    assert cfg.tmdb.api_key == "abc"
    # idempotent
    write_weights(p, BASE_W, LLM_W)
    assert p.read_text() == text


def test_write_weights_creates_file_and_sections(tmp_path):
    p = tmp_path / "sub" / "config.toml"
    write_weights(p, BASE_W, LLM_W)
    cfg = load_config(p, env={})
    assert cfg.weights.synopsis == 2.5 and cfg.weights.llm.llm == 1.45
    q = tmp_path / "c.toml"
    q.write_text('[naming]\nmove_extras = false\n\n[llm]\nmode = "auto"\n')
    write_weights(q, BASE_W, LLM_W)
    cfg = load_config(q, env={})
    assert cfg.naming.move_extras is False and cfg.llm.mode == "auto"
    assert cfg.weights.llm.refsubs == 1.2 and cfg.weights.guests == 1.3
    # only [weights] present: [weights.llm] is added right after it
    text = render_weights("[weights]\ntitle = 3.0\n\n[naming]\nmove_extras = true\n", BASE_W, LLM_W)
    assert text.index("[weights.llm]") < text.index("[naming]")
    assert tomllib.loads(text)["naming"] == {"move_extras": True}


def test_write_weights_drops_conflicting_llm_keys():
    text = render_weights("[weights]\nllm.llm = 3.0\nduration = 2.0\n", BASE_W, LLM_W)
    assert tomllib.loads(text)["weights"]["llm"]["llm"] == 1.45


def test_weights_config_round_trip():
    cfg = to_weights_config(BASE_W, LLM_W)
    assert isinstance(cfg, WeightsConfig)
    assert cfg.synopsis == 2.5 and cfg.llm.llm == 1.45 and cfg.llm.duration == 0.7
    assert WeightsConfig.model_validate(cfg.model_dump()) == cfg
    assert to_weights_config({}, {}) == WeightsConfig()
    with pytest.raises(ValueError):
        to_weights_config({"bogus": 1.0}, {})
