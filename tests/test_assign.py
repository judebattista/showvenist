"""Basic assignment (spec §11.5) and confidence by re-solve (§11.7), checked by brute force."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from showvenist.assign.basic import assign_basic
from showvenist.assign.confidence import confidence
from showvenist.config import WeightsConfig
from showvenist.fusion import fuse, log_prior
from showvenist.scoring.types import HIGH_DELTA, ScoreResult


def _all_assignments(F: int, E: int):
    """Every map file -> candidate or -1, with candidates used at most once."""
    for cols in itertools.product(range(-1, E), repeat=F):
        used = [c for c in cols if c >= 0]
        if len(used) == len(set(used)):
            yield cols


def _value(S, cols) -> float:
    return float(sum(S[i, c] for i, c in enumerate(cols) if c >= 0))


def _brute(S, forbidden=frozenset()):
    best, arg = -math.inf, None
    for cols in _all_assignments(*S.shape):
        if any((i, c) in forbidden for i, c in enumerate(cols)):
            continue
        v = _value(S, cols)
        if v > best:
            best, arg = v, cols
    return best, arg


# (c) brute force -------------------------------------------------------------------------------


SHAPES = [(1, 1), (2, 2), (3, 2), (2, 3), (4, 4), (5, 3), (3, 5), (5, 5), (4, 1)]


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("shape", SHAPES)
def test_assign_and_delta_match_brute_force(shape, seed):
    rng = np.random.default_rng(1000 * seed + 10 * shape[0] + shape[1])
    S = rng.normal(0.0, 3.0, shape)
    a = assign_basic(S)
    best, _ = _brute(S)
    assert a.total == pytest.approx(best)
    assert _value(S, a.column) == pytest.approx(best)
    used = [c for c in a.column if c >= 0]
    assert len(used) == len(set(used))

    for i, fc in enumerate(confidence(S, a)):
        alt_best, _ = _brute(S, frozenset({(i, a.column[i])}))
        if alt_best == -math.inf:
            assert fc.delta == math.inf and fc.runner_up is None
            continue
        assert fc.delta == pytest.approx(best - alt_best, abs=1e-9)
        assert fc.runner_up_delta == pytest.approx(fc.delta)
        assert fc.runner_up != a.column[i]
        # the runner-up is a choice for file i that some constrained optimum makes
        alts = [
            cols for cols in _all_assignments(*S.shape)
            if cols[i] == fc.runner_up and cols[i] != a.column[i]
        ]
        assert max(_value(S, c) for c in alts) == pytest.approx(alt_best)


def test_forbidden_pairs_respected():
    S = np.array([[5.0, 1.0], [1.0, 5.0]])
    a = assign_basic(S, {(0, 0)})
    assert a.column[0] != 0
    assert a.total == pytest.approx(_brute(S, frozenset({(0, 0)}))[0])
    # forbidding the unmatched column forces an episode
    b = assign_basic(np.array([[-3.0, -1.0]]), {(0, -1)})
    assert b.column == [1] and b.total == pytest.approx(-1.0)


def test_degenerate_shapes():
    assert assign_basic(np.zeros((0, 4))).column == []
    assert confidence(np.zeros((0, 4)), assign_basic(np.zeros((0, 4)))) == []
    a = assign_basic(np.zeros((3, 0)))
    assert a.column == [-1, -1, -1] and a.total == 0.0
    # no episodes: forbidding unmatched is infeasible -> Δ = inf, no runner-up
    for fc in confidence(np.zeros((3, 0)), a):
        assert fc.delta == math.inf and fc.runner_up is None and fc.level == "high"


# (d) two-parter swap ---------------------------------------------------------------------------


def test_two_parter_swap_gives_both_small_delta():
    lp = log_prior(10)
    S = np.full((2, 10), lp - 5.0)
    S[:, 0] = S[:, 1] = lp + 8.0  # both files look equally like E1 and E2
    S[0, 0] += 0.3  # a whisker of preference
    a = assign_basic(S)
    assert sorted(a.column) == [0, 1]
    conf = confidence(S, a)
    for fc in conf:
        assert fc.delta == pytest.approx(0.3)
        assert fc.level == "low"
    assert conf[0].runner_up == a.column[1] and conf[1].runner_up == a.column[0]


# (e) identified extra --------------------------------------------------------------------------


def test_extra_with_strong_negative_duration_is_identified():
    F, E = 3, 22
    dur = np.zeros((F, E))
    dur[2] = -30.0  # clamped to -8: far from every runtime
    syn = np.zeros((F, E))
    syn[0, 3] = syn[1, 7] = 4.0
    title = np.zeros((F, E))
    title[0, 3] = title[1, 7] = 6.0
    res = {
        "duration": ScoreResult(dur, np.ones(F, bool)),
        "synopsis": ScoreResult(syn, np.ones(F, bool)),
        "title": ScoreResult(title, np.ones(F, bool)),
    }
    S = fuse(res, WeightsConfig()).S
    a = assign_basic(S)
    assert a.column == [3, 7, -1]
    conf = confidence(S, a)
    assert conf[2].delta >= HIGH_DELTA and conf[2].level == "high"
    assert conf[2].runner_up not in (None, -1, 3, 7)
    assert conf[2].delta == pytest.approx(8.0 - log_prior(E))


# (f) more files than episodes ------------------------------------------------------------------


def test_more_files_than_episodes_leaves_weakest_unmatched():
    S = np.array([[6.0, 1.0], [1.0, 5.0], [2.0, 2.5], [0.5, 0.5]])
    a = assign_basic(S)
    assert a.column == [0, 1, -1, -1]
    assert a.total == pytest.approx(11.0)
    conf = confidence(S, a)
    # forcing file 2 onto an episode costs it its place: best is file 2 -> E1, file 0 -> E0
    assert conf[2].runner_up == 1
    assert conf[2].delta == pytest.approx(11.0 - (6.0 + 2.5))


# (g) history penalty changes the assignment ----------------------------------------------------


def test_history_penalty_moves_file_off_taken_episode():
    E = 4
    dur = np.zeros((1, E))
    dur[0, 0] = 2.0
    dur[0, 1] = 1.5
    title = np.full((1, E), 3.0)
    res = {
        "duration": ScoreResult(dur, np.ones(1, bool)),
        "title": ScoreResult(title, np.ones(1, bool)),
    }
    S0 = fuse(res, WeightsConfig()).S
    assert assign_basic(S0).column == [0]
    hp = np.zeros((1, E))
    hp[0, 0] = math.log(10)  # E0 already confirmed for a different fingerprint
    S1 = fuse(res, WeightsConfig(), history_penalty=hp).S
    assert S1[0, 0] < S0[0, 0]
    assert assign_basic(S1).column == [1]
