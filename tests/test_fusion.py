"""Fusion (spec §11.1, §11.2, §11.4): prior arithmetic, clamps, weights, regimes, history."""

from __future__ import annotations

import math

import numpy as np
import pytest

from showvenist.config import LlmWeights, WeightsConfig
from showvenist.fusion import fuse, log_prior
from showvenist.scoring.types import DEFAULT_CLAMPS, ScoreResult


def _res(llr, available=None) -> ScoreResult:
    llr = np.asarray(llr, dtype=float)
    avail = np.ones(llr.shape[0], bool) if available is None else np.asarray(available, bool)
    return ScoreResult(llr, avail)


# (a) prior arithmetic ---------------------------------------------------------------------------


def test_prior_e22_needs_about_1_4_nats():
    lp = log_prior(22, 0.15)
    assert lp == pytest.approx(math.log(0.85 / 22) - math.log(0.15))
    assert -lp == pytest.approx(1.4, abs=0.05)


def test_prior_e200_needs_about_3_5_nats():
    # spec §11.2 says "about 3.5"; exactly log(200 · 0.15 / 0.85) ≈ 3.56
    assert -log_prior(200, 0.15) == pytest.approx(math.log(200 * 0.15 / 0.85))
    assert -log_prior(200, 0.15) == pytest.approx(3.5, abs=0.1)


def test_prior_more_candidates_demand_more_evidence():
    assert log_prior(200) < log_prior(22) < log_prior(5)


def test_prior_rejects_bad_inputs():
    with pytest.raises(ValueError):
        log_prior(0)
    with pytest.raises(ValueError):
        log_prior(10, 0.0)


# (b) clamp + weight arithmetic and regimes -----------------------------------------------------


def test_clamp_and_weight_base_regime():
    w = WeightsConfig(duration=2.0, refsubs=0.5)
    res = {
        "duration": _res([[-20.0, 1.0, 5.0]]),  # clamp [-8, 2]
        "refsubs": _res([[30.0, -1.0, 3.0]]),  # clamp [-4, 12]
    }
    f = fuse(res, w)
    lp = log_prior(3)
    np.testing.assert_allclose(f.contributions["duration"], [[-16.0, 2.0, 4.0]])
    np.testing.assert_allclose(f.contributions["refsubs"], [[6.0, -0.5, 1.5]])
    np.testing.assert_allclose(f.S, [[lp - 10.0, lp + 1.5, lp + 5.5]])
    assert f.log_prior == pytest.approx(lp)
    assert f.regimes == ["base"]


def test_llm_regime_replaces_synopsis_guests_cues_only_for_llm_files():
    F, E = 3, 2
    ones = np.ones((F, E))
    w = WeightsConfig(
        duration=1.0, synopsis=1.0, guests=1.0, cues=1.0,
        llm=LlmWeights(duration=3.0, llm=2.0),
    )
    res = {
        "duration": _res(ones),
        "synopsis": _res(ones),
        "guests": _res(ones),
        "cues": _res(ones),
        # file 1 is LLM-scored; files 0 and 2 are not (refusal / not run)
        "llm": _res(np.full((F, E), 20.0), available=[False, True, False]),
    }
    f = fuse(res, w)
    lp = log_prior(E)
    assert f.regimes == ["base", "llm", "base"]
    # base files: duration + synopsis + guests + cues at base weights
    np.testing.assert_allclose(f.S[[0, 2]], lp + 4.0)
    # llm file: duration at llm weight 3 + llm clamped to 12 at weight 2 = 27
    np.testing.assert_allclose(f.S[1], lp + 3.0 + 24.0)
    for name in ("synopsis", "guests", "cues"):
        np.testing.assert_array_equal(f.contributions[name][1], 0.0)
        np.testing.assert_array_equal(f.contributions[name][[0, 2]], 1.0)
    np.testing.assert_array_equal(f.contributions["llm"][[0, 2]], 0.0)


def test_unavailable_and_absent_scorers_contribute_zero():
    res = {"duration": _res([[5.0, -5.0], [-3.0, 1.0]], available=[True, False])}
    f = fuse(res, WeightsConfig())
    lp = log_prior(2)
    np.testing.assert_allclose(f.S, [[lp + 2.0, lp - 5.0], [lp, lp]])
    assert set(f.contributions) == {"duration"}


def test_contributions_sum_to_S():
    rng = np.random.default_rng(1)
    res = {n: _res(rng.normal(0, 5, (4, 6))) for n in DEFAULT_CLAMPS}
    res["llm"].available[:] = [True, False, True, False]
    hp = np.zeros((4, 6))
    hp[2, 3] = math.log(10)
    f = fuse(res, WeightsConfig(), history_penalty=hp)
    total = f.log_prior + sum(f.contributions.values())
    np.testing.assert_allclose(f.S, total)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        fuse({"duration": _res(np.zeros((2, 3))), "title": _res(np.zeros((2, 4)))},
             WeightsConfig())


# (g) history penalty -----------------------------------------------------------------------------


def test_history_penalty_lowers_taken_episode():
    res = {"duration": _res(np.zeros((2, 3)))}
    hp = np.zeros((2, 3))
    hp[:, 1] = math.log(10)
    f0 = fuse(res, WeightsConfig())
    f = fuse(res, WeightsConfig(), history_penalty=hp)
    np.testing.assert_allclose(f.S[:, 1], f0.S[:, 1] - math.log(10))
    np.testing.assert_allclose(f.S[:, [0, 2]], f0.S[:, [0, 2]])
    np.testing.assert_allclose(f.contributions["history"], -hp)


# (h) for-only scorers never lower S --------------------------------------------------------------


@pytest.mark.parametrize("name", ["title", "synopsis", "guests", "cues"])
def test_for_only_scorer_negative_raw_never_lowers_S(name):
    raw = np.array([[-50.0, -0.1, 0.0, 2.0]])
    f = fuse({name: _res(raw)}, WeightsConfig())
    assert (f.contributions[name] >= 0).all()
    assert (f.S >= f.log_prior).all()
    np.testing.assert_allclose(f.contributions[name], [[0.0, 0.0, 0.0, 2.0]])


def test_for_only_guard_holds_even_with_negative_clamp_or_weight():
    raw = np.array([[-3.0, 1.0]])
    bad_clamps = dict(DEFAULT_CLAMPS, synopsis=(-4.0, 4.0))
    f = fuse({"synopsis": _res(raw)}, WeightsConfig(synopsis=-1.0), clamps=bad_clamps)
    assert (f.contributions["synopsis"] >= 0).all()
