# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "wordfreq", "pytest"]
# ///
"""Run: uv run spike/test_score.py   (synthetic text only)"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("score", Path(__file__).parent / "score.py")
sc = importlib.util.module_from_spec(spec)
sys.modules["score"] = sc
spec.loader.exec_module(sc)


def test_prior():
    assert sc.log_prior(22) == pytest.approx(np.log(0.85 / 22) - np.log(0.15))
    assert -1.5 < sc.log_prior(22) < -1.3  # ~1.4 nats needed


def test_assignment_known_answer():
    E = 4
    S = np.full((4, E), -3.0)
    for i in range(4):
        S[i, i] = 5.0
    a = sc.assign(S)
    assert a["sigma"].tolist() == [0, 1, 2, 3]
    # forbidding (i,i): best alternative is unmatched (0) -> delta = 5
    assert a["delta"].tolist() == pytest.approx([5.0] * 4)
    assert all(r == -1 for r in a["runner_up"])
    assert sc.level(5.0) == "high" and sc.level(3) == "medium" and sc.level(1) == "low"


def test_assignment_conflict_and_unmatched():
    # files 0,1 both want episode 0 (6 vs 5); file 1 prefers it over episode 1 (1); file 2 is an extra
    S = np.array([[6.0, -2, -2], [5.0, 1.0, -2], [-3, -3, -3]])
    a = sc.assign(S)
    assert a["sigma"].tolist() == [0, 1, -1]
    # file 0: total 7; forbid -> f1 takes ep0 (5), f0 unmatched: 5 -> delta 2
    assert a["delta"][0] == pytest.approx(7 - 5)
    assert a["delta"][2] > 0  # extra: forbidding unmatched forces -3


def test_bm25_synopsis_and_names():
    tm = {"series_cast": [{"name": "Zed Quark", "character": "Captain Bloop"}],
          "episodes": [
              {"season": 1, "episode": i + 1, "title": t, "overview": o,
               "guest_stars": [{"actor": "Pat Fenwick", "character": g}]}
              for i, (t, o, g) in enumerate([
                  ("Volcano Rescue", "Captain Bloop rescues villagers beneath a volcano eruption lava", "Krellix"),
                  ("Quiet Harbor", "A fisherman mends nets near the harbor lighthouse storm", "Morvane"),
                  ("Desert Heist", "Thieves plan a heist across the desert dunes camel vault", "Tolliver"),
                  ("Orchard", "Apples ripen in the orchard while farmers argue about frost", "Wimbrel"),
                  ("Clockwork", "An inventor builds clockwork gears for a mechanical bird", "Draxon"),
                  ("Glacier", "Explorers cross a glacier searching for a frozen cavern", "Pelham"),
              ])]}
    s = sc.Season(tm)
    assert "bloop" not in {t for d in s.docs for t in d}  # series cast removed
    lines = ["the lava from the volcano is rising", "eruption eruption near the villagers", "Captain Bloop help"]
    f = s.features(lines)
    assert f.shape == (6, 2)
    assert f[0, 0] > 0 and f[1:, 0].max() == 0 and f[0, 0] <= sc.CLAMP_SYN
    f2 = s.features(["Krellix is here", "ask Morvane and Krellix"])
    assert f2[0, 1] > 0
    assert f2[1, 1] > 0 and f2[2, 1] == 0
    # nothing heard -> zeros
    assert np.all(s.features(["hello there friend"]) == 0)


def _synthetic(rng, F=22, E=22, signal=(3.0, 1.5), noise=1.0):
    """Features where guest is informative, synopsis weakly; truth = identity."""
    feat = np.zeros((F, E, 2))
    for i in range(F):
        feat[i, :, 0] = np.clip(rng.normal(0, noise, E), 0, 4)
        feat[i, :, 1] = np.clip(rng.normal(0, noise * .3, E), 0, 6)
        feat[i, i, 0] = np.clip(feat[i, i, 0] + signal[1], 0, 4)
        feat[i, i, 1] = np.clip(signal[0] + feat[i, i, 1], 0, 6)
    return feat


def test_fit_sanity():
    rng = np.random.default_rng(0)
    feat = _synthetic(rng)
    truth = list(range(22))
    w = sc.fit_weights([feat], [truth], [22], lam=0.5)
    assert w.shape == (2,) and np.all(w >= 0)
    # fitted weights should not do worse in NLL than defaults
    n0 = sc.nll(np.ones(2), [feat], [truth], [22], 0.5, np.ones(2))
    assert sc.nll(w, [feat], [truth], [22], 0.5, np.ones(2)) <= n0 + 1e-9
    # strong regularization keeps w near 1
    w_big = sc.fit_weights([feat], [truth], [22], lam=1e6)
    assert np.allclose(w_big, 1.0, atol=1e-2)
    # an uninformative scorer gets down-weighted vs. an informative one
    feat2 = feat.copy()
    feat2[:, :, 0] = np.clip(rng.normal(0, 1, (22, 22)), 0, 4)
    w2 = sc.fit_weights([feat2], [truth], [22], lam=0.1)
    assert w2[0] < w2[1]


def test_run_modes_and_metrics():
    rng = np.random.default_rng(1)
    data = {f"s{k}": {"feat": _synthetic(rng), "truth": list(range(22))} for k in range(3)}
    r = sc.run_modes(data, 1.0)
    for mode in ("default", "loso", "insample"):
        m = r["modes"][mode]["s0"]
        assert m["top1_assigned"] >= 0.9
        assert m["per_scorer"]["guest"]["rank1_rate"] > 0.9
        assert m["gap_true_vs_best_other"]["median"] > 0
    pooled = sc.pool(r["modes"]["loso"])
    assert pooled["n_files"] == 66 and pooled["errors_at_high"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
