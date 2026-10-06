"""Ordering detection (spec §11.6)."""

from __future__ import annotations

import numpy as np

from showvenist.assign.basic import assign_basic
from showvenist.assign.ordering import detect_ordering
from showvenist.assign.types import BlockParams, DiscSpec

E = 14
# Firefly-like: the release follows the intended order; aired order moves episodes around.
INTENDED = list(range(E))  # episode id at intended position p
AIRED = [1, 2, 6, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 0]  # episode id at aired position p
PER_DISC = [4, 4, 3, 3]

# mechanism tests use small synthetic matrices: pin the 4-nat threshold they were written for
P = BlockParams(ordering_threshold=4.0)


def _files(seed=0):
    """Files in release (intended) order: one per episode plus an extra per disc."""
    rng = np.random.default_rng(seed)
    episode_of, disc_rows, row = [], [], 0
    ep = 0
    for n in PER_DISC:
        rows = []
        for _ in range(n):
            episode_of.append(ep)
            rows.append(row)
            ep, row = ep + 1, row + 1
        episode_of.append(None)  # an extra
        rows.append(row)
        row += 1
        disc_rows.append(rows)
    return episode_of, disc_rows, rng


def _builder(episode_of, disc_rows, rng, scramble=False):
    F = len(episode_of)
    base = {}  # S over episode ids; columns are permuted per ordering
    M = rng.normal(0.0, 0.3, (F, E))
    for i, e in enumerate(episode_of):
        if e is None:
            M[i] = -5.0
        else:
            M[i, e] = 4.0
    if scramble:  # discs hold episodes from all over the season, in every ordering
        rows = [r for rows in disc_rows for r in rows]
        perm = rng.permutation(len(rows))
        disc_rows = [list(np.array(rows)[perm[k : k + 5]]) for k in range(0, len(rows), 5)]
    discs = [
        DiscSpec(f"disc{k + 1}", tuple(int(r) for r in rows), k + 1)
        for k, rows in enumerate(disc_rows)
    ]
    base["intended"] = M[:, INTENDED]
    base["aired"] = M[:, AIRED]
    calls = []

    def build(name):
        calls.append(name)
        return base[name], discs, ()

    return build, calls


def test_switches_to_intended_when_aired_fits_badly():
    build, calls = _builder(*_files())
    choice = detect_ordering(build, ["aired", "intended"], "aired", P)
    assert choice.ordering == "intended" and choice.switched and not choice.fits_nowhere
    assert choice.solution.used_blocks
    assert set(choice.tried) == {"aired", "intended"}
    assert choice.tried["intended"] > choice.tried["aired"] + P.ordering_threshold
    S = build("intended")[0]
    assert choice.solution.total >= assign_basic(S).total - P.ordering_threshold


def test_keeps_starting_order_when_it_fits():
    build, calls = _builder(*_files())
    choice = detect_ordering(build, ["aired", "intended"], "intended", P)
    assert choice.ordering == "intended" and not choice.switched and not choice.fits_nowhere
    # every ordering is solved (decision 46), but the start is kept: no other beats it
    assert set(calls) == {"intended", "aired"}
    assert choice.tried["aired"] <= choice.tried["intended"] + P.ordering_threshold


def test_fits_nowhere_falls_back_to_basic():
    episode_of, disc_rows, rng = _files(seed=3)
    build, _ = _builder(episode_of, disc_rows, rng, scramble=True)
    choice = detect_ordering(build, ["aired", "intended"], "aired", P)
    assert choice.fits_nowhere and not choice.switched and choice.ordering == "aired"
    assert not choice.solution.used_blocks
    S = build("aired")[0]
    basic = assign_basic(S)
    assert choice.solution.column == basic.column
    assert choice.solution.total == basic.total


def test_orderings_with_other_files_are_not_compared():
    episode_of, disc_rows, rng = _files()
    build, _ = _builder(episode_of, disc_rows, rng)

    def build2(name):
        if name == "intended":
            S, discs, loose = build(name)
            return (
                S[:-1],
                [
                    DiscSpec(d.key, tuple(f for f in d.files if f < len(S) - 1), d.number)
                    for d in discs
                ],
                loose,
            )
        return build(name)

    choice = detect_ordering(build2, ["aired", "intended"], "aired", P)
    assert "intended" not in choice.tried and choice.fits_nowhere


def test_hysteresis_keeps_start_when_the_alternative_is_only_slightly_better():
    from showvenist.assign.types import DiscSpec

    S0 = np.zeros((3, 3))
    S0[0, 0] = S0[1, 1] = S0[2, 2] = 5.0
    S1 = S0.copy()
    S1[0, 0] = 6.0  # one nat better: below the 4-nat threshold
    discs = [DiscSpec("d1", (0, 1, 2), 1)]

    def build(name):
        return (S0 if name == "a" else S1), discs, ()

    choice = detect_ordering(build, ["a", "b"], "a", P)
    assert choice.ordering == "a" and not choice.switched


def test_wrong_season_filter_is_detected_even_when_blocks_and_basic_fail_together():
    """Firefly: aired has fewer candidates than files, so blocks and basic are both poor and
    close to each other; another ordering is much better."""
    from showvenist.assign.types import DiscSpec

    rng = np.random.default_rng(3)
    truth = 6
    good = np.full((truth, truth), -3.0)
    np.fill_diagonal(good, 6.0)
    bad = np.full((truth, 4), -3.0)  # only 4 candidates for 6 files: two files cannot match
    bad[0, 0] = bad[1, 1] = bad[2, 2] = bad[3, 3] = 6.0
    bad += rng.normal(0, 0.05, bad.shape)
    discs = [DiscSpec("d1", (0, 1, 2), 1), DiscSpec("d2", (3, 4, 5), 2)]

    def build(name):
        return (bad if name == "aired" else good), discs, ()

    choice = detect_ordering(build, ["aired", "intended"], "aired", P)
    assert choice.ordering == "intended" and choice.switched


def test_default_threshold_is_the_tuned_value():
    assert BlockParams().ordering_threshold == 15.0  # decision 47
