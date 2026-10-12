"""Disc block solver (spec §11.5) and block confidence (§11.7), checked by brute force."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

import showvenist.assign.blocks as blocks_mod
from showvenist.assign.blocks import block_confidence, solve_blocks
from showvenist.assign.types import BlockParams, DiscSpec

P = BlockParams()  # u = 1, g = 2, slack 2


# ---- independent reference: the model of spec §11.5 written out by enumeration -------------------
# Nothing here is shared with the solver: placements, the length bound, gap and unclaimed
# accounting are re-derived from the spec text. Skipped disc numbers are read from the numbers
# themselves (the flag is only used for unnumbered discs, whose skips the solver can't see).


def _ref_order(discs):
    """§11.5: discs sorted by parsed number. §14: unnumbered discs keep the confirmed order."""
    if any(d.number is None for d in discs):
        return list(discs)
    return sorted(discs, key=lambda d: d.number)


def _ref_matchings(files, a, b):
    """Every injective map of `files` into positions [a, b) ∪ {unmatched}."""
    if not files:
        yield ()
        return
    for rest in _ref_matchings(files[1:], a, b):
        taken = {c for c in rest if c != -1}
        yield (-1, *rest)
        for c in range(a, b):
            if c not in taken:
                yield (c, *rest)


def _ref_block(S, d, a, b, params, forbidden):
    """block_score = best matching − u · (positions in [a, b) no file claims); u = 0 if
    incomplete."""
    u = 0.0 if d.incomplete else params.unclaimed_penalty
    best, arg = -math.inf, None
    for cols in _ref_matchings(d.files, a, b):
        if any((f, c) in forbidden for f, c in zip(d.files, cols, strict=True)):
            continue
        claimed = sum(1 for c in cols if c != -1)
        v = sum(S[f, c] for f, c in zip(d.files, cols, strict=True) if c != -1)
        v -= u * ((b - a) - claimed)
        if v > best:
            best, arg = v, cols
    return best, arg


def _ref_placements(order, E, slack, k=0, prev_b=0):
    """Every in-order, non-overlapping choice of one block per disc. A fixed block is forced; a
    complete disc spans at most files + slack positions; an incomplete one is open-ended."""
    if k == len(order):
        yield ()
        return
    d = order[k]
    if d.fixed is not None:
        a, b = d.fixed
        options = [(a, b)] if prev_b <= a <= b <= E else []
    else:
        options = [
            (a, b)
            for a in range(prev_b, E + 1)
            for b in range(a, E + 1)
            if d.incomplete or b - a <= len(d.files) + slack
        ]
    for a, b in options:
        for rest in _ref_placements(order, E, slack, k + 1, b):
            yield ((a, b), *rest)


def _ref_gap_cost(order, k, prev_b, a, g):
    """g per skipped position before disc k, free where numbers skip and before a first disc
    numbered > 1. Nothing is ever charged after the last disc (placements may end anywhere)."""
    d = order[k]
    if k == 0:
        if d.number is not None and d.number > 1:
            return 0.0
        return g * a
    p = order[k - 1]
    if d.number is not None and p.number is not None:
        skip = d.number > p.number + 1
    else:
        skip = d.free_gap_before
    return 0.0 if skip else g * (a - prev_b)


def _brute(S, discs, params=P, forbidden=frozenset()):
    """(best total, column) of the block model over every placement and every matching."""
    F, E = S.shape
    order = _ref_order(discs)
    memo = {}

    def block(k, a, b):
        if (k, a, b) not in memo:
            memo[(k, a, b)] = _ref_block(S, order[k], a, b, params, forbidden)
        return memo[(k, a, b)]

    best, arg = -math.inf, None
    for placement in _ref_placements(order, E, params.block_slack):
        v, prev_b = 0.0, 0
        for k, (a, b) in enumerate(placement):
            v -= _ref_gap_cost(order, k, prev_b, a, params.gap_penalty)
            v += block(k, a, b)[0]
            prev_b = b
        if v > best:
            best, arg = v, placement
    if arg is None:
        return -math.inf, None
    column = [-1] * F
    for k, (a, b) in enumerate(arg):
        for f, c in zip(order[k].files, block(k, a, b)[1], strict=True):
            column[f] = c
    return best, column


def _value(S, discs, sol, params=P):
    """Recompute a solution's objective from its blocks and columns (no loose files)."""
    order = _ref_order(discs)
    v, prev_b = 0.0, 0
    for k, d in enumerate(order):
        a, b = sol.blocks[d.key]
        assert a >= prev_b
        v -= _ref_gap_cost(order, k, prev_b, a, params.gap_penalty)
        cols = [sol.column[f] for f in d.files]
        assert all(c == -1 or a <= c < b for c in cols)
        u = 0.0 if d.incomplete else params.unclaimed_penalty
        v += sum(S[f, c] for f, c in zip(d.files, cols, strict=True) if c >= 0)
        v -= u * ((b - a) - sum(c >= 0 for c in cols))
        prev_b = b
    return v


def _random_case(rng):
    """A season with structure: consecutive true blocks, skipped titles (fewer files than the
    block's episodes), extras and duplicates (more files), skipped disc numbers, unnumbered,
    incomplete and fixed discs. One case in four is pure noise. Returns S, discs, features."""
    E = int(rng.integers(2, 7))
    D = int(rng.integers(1, 4))
    unnumbered = rng.random() < 0.25
    numbers = sorted(rng.choice(np.arange(1, 6), D, replace=False).tolist())
    cuts = sorted(rng.choice(np.arange(0, E + 1), D + 1, replace=True).tolist())
    rows: list[np.ndarray] = []
    discs, feats = [], set()
    for k in range(D):
        a, b = cuts[k], cuts[k + 1]
        files = []
        for e in range(a, b):
            if rng.random() < 0.3:
                feats.add("skipped_title")
                continue
            if len(files) >= 3:
                break
            r = rng.normal(0.0, 1.5, E)
            r[e] = rng.uniform(2.0, 5.0)
            files.append(len(rows))
            rows.append(r)
            if rng.random() < 0.2 and len(files) < 3:  # a near-duplicate rip
                files.append(len(rows))
                rows.append(r - 0.3)
                feats.add("duplicate")
        while len(files) < 3 and rng.random() < 0.3:  # an extra
            files.append(len(rows))
            rows.append(rng.normal(-3.0, 1.0, E))
            feats.add("extra")
        if len(files) < b - a:
            feats.add("fewer_files_than_episodes")
        if len(files) > b - a:
            feats.add("more_files_than_episodes")
        number = None if unnumbered else numbers[k]
        skip = k > 0 and numbers[k] > numbers[k - 1] + 1
        incomplete = bool(rng.random() < 0.25)
        fixed = None
        if rng.random() < 0.2:
            fixed = (a, b) if rng.random() < 0.7 else (int(rng.integers(0, E)), E)
            feats.add("fixed")
        feats |= {"unnumbered"} if unnumbered else set()
        feats |= {"skipped_number"} if skip else set()
        feats |= {"incomplete"} if incomplete else set()
        discs.append(
            DiscSpec(f"d{k}", tuple(files), number, incomplete, fixed, free_gap_before=skip)
        )
    S = np.array(rows).reshape(len(rows), E)
    if rng.random() < 0.25:
        S = rng.normal(0.0, 3.0, S.shape)
        feats.add("noise")
    if not unnumbered:
        rng.shuffle(discs)  # the solver sorts by number
    return S, discs, feats


SEEDS = range(80)


def test_random_cases_cover_every_feature():
    seen = set()
    for seed in SEEDS:
        seen |= _random_case(np.random.default_rng(seed))[2]
    assert seen >= {
        "fixed",
        "unnumbered",
        "incomplete",
        "skipped_number",
        "skipped_title",
        "fewer_files_than_episodes",
        "more_files_than_episodes",
        "extra",
        "duplicate",
        "noise",
    }


# ---- (a) brute force ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_solve_blocks_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    S, discs, _ = _random_case(rng)
    sol = solve_blocks(S, discs)
    best, _ = _brute(S, discs)
    if best == -math.inf:
        assert sol.total == -math.inf
        return
    assert sol.total == pytest.approx(best)
    assert _value(S, discs, sol) == pytest.approx(best)
    used = [c for c in sol.column if c >= 0]
    assert len(used) == len(set(used))


@pytest.mark.parametrize("slack", [0, 1, 3])
def test_block_slack_is_a_parameter(slack):
    params = BlockParams(block_slack=slack)
    for seed in range(15):
        S, discs, _ = _random_case(np.random.default_rng(900 + seed))
        best, _ = _brute(S, discs, params=params)
        assert solve_blocks(S, discs, params=params).total == pytest.approx(best)


@pytest.mark.parametrize("seed", range(20))
def test_forbidden_pairs_match_brute_force(seed):
    rng = np.random.default_rng(500 + seed)
    S, discs, _ = _random_case(rng)
    F, E = S.shape
    if F == 0:
        return
    forbidden = {(int(rng.integers(0, F)), int(rng.integers(-1, E))) for _ in range(2)}
    sol = solve_blocks(S, discs, forbidden=forbidden)
    best, _ = _brute(S, discs, forbidden=frozenset(forbidden))
    assert sol.total == pytest.approx(best)
    if math.isfinite(best):
        assert not any((i, c) in forbidden for i, c in enumerate(sol.column))


def test_skipped_title_keeps_the_last_episode_matched():
    """Disc 1 holds episodes 1-4 but MakeMKV skipped episode 3's title: 3 files for a 4-wide
    block. The episode-4 file must stay matched (one unclaimed position costs u)."""
    E = 8
    truth = {0: 0, 1: 1, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}  # row -> episode column
    S = np.full((7, E), -2.0)
    for r, e in truth.items():
        S[r, e] = 4.0
    discs = [DiscSpec("disc1", (2, 0, 1), 1), DiscSpec("disc2", (3, 4, 5, 6), 2)]
    sol = solve_blocks(S, discs)
    assert sol.column == [truth[r] for r in range(7)]
    assert sol.blocks == {"disc1": (0, 4), "disc2": (4, 8)}
    assert sol.total == pytest.approx(7 * 4.0 - P.unclaimed_penalty)
    # without slack the ep-4 file is forced out of its block and loses its evidence
    tight = solve_blocks(S, discs, params=BlockParams(block_slack=0))
    assert tight.column[2] != 3 and tight.total < sol.total - 4.0


# ---- (b) content, not counts --------------------------------------------------------------------


def _season(truth_per_disc, E, extras_per_disc, dups=(), strong=4.0, seed=0):
    """Files by disc: each true episode once, then duplicates (copies of an episode's row minus
    a little) and extras (rejected everywhere). Returns S, discs, truth column per row."""
    rng = np.random.default_rng(seed)
    rows, discs, truth = [], [], []
    for k, (eps, n_extra) in enumerate(zip(truth_per_disc, extras_per_disc, strict=True)):
        files = []
        for e in eps:
            r = rng.normal(0.0, 0.3, E)
            r[e] = strong
            files.append(len(rows))
            rows.append(r)
            truth.append(e)
        for e in (x for x in dups if x in eps):
            files.append(len(rows))
            rows.append(rows[files[eps.index(e)]] - 0.2)
            truth.append(-1)
        for _ in range(n_extra):
            files.append(len(rows))
            rows.append(rng.normal(-5.0, 0.5, E))
            truth.append(-1)
        order = rng.permutation(len(files))  # titles within a disc are in random order
        discs.append(DiscSpec(f"disc{k + 1}", tuple(files[i] for i in order), k + 1))
    return np.array(rows), discs, truth


def test_blocks_found_from_content_not_counts():
    eps = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12)), list(range(12, 16))]
    S, discs, truth = _season(eps, 16, extras_per_disc=[2, 1, 3, 0], dups=(5,))
    # a misleading file: disc 1's episode-2 file scores better on episode 14 than on 2, and
    # episode 14's own file is weak; the basic assignment takes the bait, the blocks don't
    f2 = next(f for f in discs[0].files if truth[f] == 1)
    f14 = next(f for f in discs[3].files if truth[f] == 13)
    S[f2, 1], S[f2, 13] = 1.0, 2.0
    S[f14, 13] = 0.5
    sol = solve_blocks(S, discs)
    assert sol.column == truth
    assert sol.blocks == {"disc1": (0, 4), "disc2": (4, 8), "disc3": (8, 12), "disc4": (12, 16)}
    from showvenist.assign.basic import assign_basic

    assert assign_basic(S).column[f2] == 13  # the plain solve is wrong here


# ---- (c) missing disc, (d) incomplete disc, (e) fixed block ------------------------------------


def test_missing_disc_leaves_free_gap():
    eps = [list(range(0, 4)), list(range(4, 8)), list(range(12, 16))]
    S, discs, truth = _season(eps, 16, extras_per_disc=[1, 1, 1])
    discs[2] = DiscSpec("disc4", discs[2].files, 4, free_gap_before=True)
    sol = solve_blocks(S, discs)
    assert sol.column == truth
    assert sol.blocks["disc4"] == (12, 16)
    matched = sum(S[f, c] for f, c in enumerate(truth) if c >= 0)
    assert sol.total == pytest.approx(matched)  # no gap charge, no unclaimed charge
    # without the skip, the 4-position gap is charged g each (or the evidence is given up)
    discs[2] = DiscSpec("disc4", discs[2].files, 3)
    assert solve_blocks(S, discs).total < matched - 1.0


def test_leading_gap_free_when_first_disc_is_not_one():
    S, discs, truth = _season([list(range(4, 8))], 8, extras_per_disc=[0])
    matched = sum(S[f, c] for f, c in enumerate(truth) if c >= 0)
    sol = solve_blocks(S, [DiscSpec("d2", discs[0].files, 2)])
    assert sol.total == pytest.approx(matched) and sol.blocks["d2"] == (4, 8)
    sol1 = solve_blocks(S, [DiscSpec("d1", discs[0].files, 1)])
    assert sol1.total < matched  # disc 1 starting at 4: the leading gap is charged


def test_incomplete_last_disc():
    # disc 2 is still being copied: only its episode-5 and episode-8 files are ready
    eps = [list(range(0, 4)), [4, 7]]
    S, discs, truth = _season(eps, 8, extras_per_disc=[0, 0])
    inc = DiscSpec("disc2", discs[1].files, 2, incomplete=True)
    sol = solve_blocks(S, [discs[0], inc])
    assert sol.column == truth
    assert sol.blocks["disc2"] == (4, 8)  # open upper boundary, unclaimed positions free
    matched = sum(S[f, c] for f, c in enumerate(truth) if c >= 0)
    assert sol.total == pytest.approx(matched)
    # marked complete, 2 files span 4 positions only through a slack of 2, paying u twice
    complete = [discs[0], DiscSpec("disc2", discs[1].files, 2)]
    slack2 = BlockParams(unclaimed_penalty=1.0, block_slack=2)
    sol_c = solve_blocks(S, complete, params=slack2)
    assert sol_c.column == truth
    assert sol_c.total == pytest.approx(matched - 2 * slack2.unclaimed_penalty)
    # without slack, 2 files can't span 4 positions: one of the two is given up
    sol_t = solve_blocks(S, complete, params=BlockParams(block_slack=0))
    assert sol_t.column != truth and sol_t.total < matched


def test_fixed_block_is_forced():
    eps = [list(range(0, 4)), list(range(4, 8))]
    S, discs, truth = _season(eps, 8, extras_per_disc=[0, 0])
    # history confirms disc 2 at [4, 8); its own file evidence is wiped out
    for f in discs[1].files:
        S[f] = 0.0
    S[discs[1].files[0], 0] = 3.0  # and one file even points elsewhere
    fixed = DiscSpec("disc2", discs[1].files, 2, fixed=(4, 8))
    sol = solve_blocks(S, [discs[0], fixed])
    assert sol.blocks["disc2"] == (4, 8)
    assert all(sol.column[f] == -1 or 4 <= sol.column[f] < 8 for f in discs[1].files)
    best, _ = _brute(S, [discs[0], fixed])
    assert sol.total == pytest.approx(best)


def test_disc_without_ready_files_gets_empty_block():
    S, discs, truth = _season([list(range(0, 4))], 8, extras_per_disc=[1])
    empty = DiscSpec("disc2", (), 2)
    sol = solve_blocks(S, [discs[0], empty])
    a, b = sol.blocks["disc2"]
    assert a == b and sol.column == truth


# ---- (f) loose files ----------------------------------------------------------------------------


def test_loose_files_take_unclaimed_positions():
    S, discs, truth = _season([list(range(0, 4))], 8, extras_per_disc=[0])
    F = S.shape[0]
    loose_rows = np.full((3, 8), -1.0)
    loose_rows[0, 6] = 3.0  # a loose episode 7
    loose_rows[1, 2] = 9.0  # points at episode 3, which disc 1 claims: next best is unmatched
    loose_rows[2, 2], loose_rows[2, 5] = 9.0, 2.0  # claimed best, then episode 6
    S = np.vstack([S, loose_rows])
    sol = solve_blocks(S, discs, loose=(F, F + 1, F + 2))
    assert sol.column[:F] == truth
    assert sol.column[F:] == [6, -1, 5]
    dp = solve_blocks(S[:F], discs).total
    assert sol.total == pytest.approx(dp + 3.0 + 2.0)
    assert "loose" not in sol.blocks


def test_infeasible_forbidden_gives_minus_inf():
    S, discs, _ = _season([[0]], 1, extras_per_disc=[0])
    sol = solve_blocks(S, discs, forbidden={(0, 0), (0, -1)})
    assert sol.total == -math.inf


def test_loose_file_holding_an_episode_a_disc_takes():
    """Sequential limitation (module docstring): disc 1's weak episode-3 file claims episode 3
    in the DP, so the loose file that holds episode 3 strongly is left unmatched. Its
    confidence must not call it an identified extra: the pinned alternative (loose file on
    episode 3, disc files forbidden from it, blocks re-solved) beats the total, so Δ = 0, level
    low, runner-up episode 3."""
    E = 6
    S = np.full((5, E), -3.0)
    for r in range(4):
        S[r, r] = 4.0
    S[2, 2] = 1.0  # disc 1's episode-3 file is weak
    S[4] = -6.0
    S[4, 2] = 9.0  # the loose file holds episode 3 strongly
    discs = [DiscSpec("disc1", (0, 1, 2, 3), 1)]
    sol = solve_blocks(S, discs, loose=(4,))
    assert sol.column == [0, 1, 2, 3, -1]  # sequential: the disc took episode 3
    conf = block_confidence(S, discs, sol, loose=(4,))
    fc = conf[4]
    assert fc.delta == 0.0 and fc.level == "low" and fc.runner_up == 2
    # the joint value the pinned alternative found: disc block without episode 3 for file 2
    pinned = solve_blocks(S, discs, forbidden={(2, 2)}).total + 9.0
    assert pinned > sol.total
    # disc files are unaffected by the loose-file rule
    assert conf[0].level == "high"


def test_loose_file_confidence_is_exact_when_nothing_competes():
    S, discs, truth = _season([list(range(0, 4))], 8, extras_per_disc=[0])
    F = S.shape[0]
    extra = np.full((1, 8), -5.0)
    extra[0, 6] = 1.5  # a loose file weakly on episode 7, which no disc wants
    S = np.vstack([S, extra])
    sol = solve_blocks(S, discs, loose=(F,))
    assert sol.column[F] == 6
    fc = block_confidence(S, discs, sol, loose=(F,))[F]
    assert fc.delta == pytest.approx(1.5) and fc.runner_up == -1  # next best: unmatched


def test_infeasible_alternative_is_low_not_high():
    """File 0 sits in an empty fixed block: forbidding its unmatched column is infeasible.
    That must read as no information (Δ 0, low), never as certainty."""
    S = np.array([[3.0, 3.0]])
    discs = [DiscSpec("d1", (0,), 1, fixed=(0, 0))]
    sol = solve_blocks(S, discs)
    assert sol.column == [-1]
    (fc,) = block_confidence(S, discs, sol)
    assert fc.delta == 0.0 and fc.level == "low" and fc.runner_up is None
    assert fc.runner_up_delta is None


# ---- (g) confidence -----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(25))
def test_block_confidence_matches_brute_force(seed):
    rng = np.random.default_rng(2000 + seed)
    S, discs, _ = _random_case(rng)
    sol = solve_blocks(S, discs)
    best = sol.total
    for i, fc in enumerate(block_confidence(S, discs, sol)):
        alt, alt_cols = _brute(S, discs, forbidden=frozenset({(i, sol.column[i])}))
        if alt == -math.inf or best == -math.inf:
            assert fc.delta == 0.0 and fc.runner_up is None and fc.level == "low"
            continue
        assert fc.delta == pytest.approx(best - alt, abs=1e-9)
        assert fc.runner_up_delta == pytest.approx(fc.delta)
        assert fc.runner_up != sol.column[i]
        # the runner-up is file i's choice in SOME constrained optimum
        alt_fixed, _ = _brute(
            S,
            discs,
            forbidden=frozenset({(i, c) for c in range(-1, S.shape[1]) if c != fc.runner_up}),
        )
        assert alt_fixed == pytest.approx(alt)


def test_block_confidence_reuses_untouched_discs(monkeypatch):
    eps = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12))]
    S, discs, truth = _season(eps, 12, extras_per_disc=[1, 1, 1])
    calls: list[tuple] = []
    real = blocks_mod._block_score_uncached

    def counting(S_, files, a, b, u, forbidden, *rest):
        calls.append((files, a, b, forbidden))
        return real(S_, files, a, b, u, forbidden, *rest)

    monkeypatch.setattr(blocks_mod, "_block_score_uncached", counting)
    sol = solve_blocks(S, discs)
    n_solve = len(calls)
    calls.clear()
    conf = block_confidence(S, discs, sol)
    assert len(conf) == S.shape[0]
    # the first re-solve fills the cache with every unconstrained block (one cold DP's worth);
    # every other evaluation is a block of the disc holding file i that the pair can change
    cold = [c for c in calls if not c[3]]
    assert len(cold) == n_solve
    by_file: dict[int, int] = {}
    for files, a, b, forbidden in calls:
        if not forbidden:
            continue
        ((i, j),) = forbidden
        assert i in files and (j == -1 or a <= j < b)
        by_file[i] = by_file.get(i, 0) + 1
    # at most one disc's worth of spans (forbidding unmatched touches all of them), never more
    assert by_file and max(by_file.values()) <= n_solve / len(discs)
    print(f"\ncold DP: {n_solve} block evaluations; per re-solve at most {max(by_file.values())}")


# ---- (i) performance ----------------------------------------------------------------------------


def test_performance_six_discs_24_episodes():
    eps = [list(range(4 * k, 4 * k + 4)) for k in range(6)]
    S, discs, truth = _season(eps, 24, extras_per_disc=[2, 1, 1, 2, 0, 3], dups=(9,), strong=3.0)
    t0 = time.perf_counter()
    sol = solve_blocks(S, discs)
    t_solve = time.perf_counter() - t0
    assert sol.column == truth
    t0 = time.perf_counter()
    conf = block_confidence(S, discs, sol)
    t_conf = time.perf_counter() - t0
    print(f"\nsolve_blocks {S.shape}: {t_solve * 1000:.1f} ms; block_confidence: {t_conf:.2f} s")
    assert len(conf) == S.shape[0]
    assert t_solve < 1.0
    assert t_conf < 5.0
