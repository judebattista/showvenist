"""Disc block assignment (spec §11.5) and its confidence by exact re-solve (§11.7).

Model (positions are the columns of S, in matching order; discs sorted by number):

    total = Σ_d block_score(d, a_d, b_d) − Σ_d gap(d, b_{d−1}, a_d) + loose total

    block_score(d, a, b) = max over one-to-one matchings of disc d's files to positions [a, b)
                           (any file may stay unmatched at 0) of Σ S − u · (unclaimed positions)

The penalty is optimized jointly with the matching: − u · (L − matched) = − u·L + Σ_matched u,
so the block's basic assignment runs on S + u and subtracts u·L. A disc marked incomplete has
u = 0 and an open upper boundary.

Block length bound: L ≤ len(files) + params.block_slack (spec: "bounded by the number of ready
files", plus slack). The slack lets a block cover an episode that has no file of its own: a
title MakeMKV skipped (minimum length) or a two-part episode muxed as one file (§14). Each such
position costs u when no file claims it, so the slack is a soft allowance, not a free one. A disc
can claim FEWER positions than it has files (extras, duplicates stay unmatched). An incomplete
disc may extend to the end of the season. Fixed blocks (confirmed history) are forced to their
[a, b) and ignore the bound.

Gaps: g per skipped position between consecutive blocks (and before the first block), free
where `free_gap_before` is set, free before the first disc if its number is > 1, always free
after the last disc (the DP's answer may end anywhere).

Disc order: by `number` when every disc has one; if any disc is unnumbered, the given order is
kept for all of them (the caller has ordered them, e.g. by folder creation time, §14).

Loose files (no disc) are assigned AFTER the DP, by the basic assignment over the positions no
disc file claimed (unclaimed positions inside blocks included; their u is still charged). Their
total is added.

Limitation (sequential, not joint): a disc block may take an episode that a loose file holds
more strongly, and the loose file then stays unmatched. The exact joint optimum couples every
block through the loose files' choices (maximize over the loose files' injective choices λ of
DP(disc files forbidden from λ's columns) + Σ S(loose, λ)), which is exponential in the number
of loose files, so it is not solved. Confidence guards against it instead: a loose file's
alternative is the best re-solve with that file PINNED to each other column c, disc files
forbidden from c (blocks re-solved), so the column a disc took is a real alternative. When
such an alternative beats the sequential total, Δ = 0 and the file goes to review with that
column as runner-up; it is never reported as a confident extra.

Rows of S in neither a disc nor `loose` are outside the solve: column −1, low confidence.

Title-order tie-breaker (§11.5, decision 63), for a disc given `titles` and a weight w > 0.
It works in two stages inside the block score, so it can never change which positions a block
claims, nor which files claim them:
1. the block's basic assignment (above) fixes the matched files and the matched columns;
2. the matched files that have a title index are re-matched among the columns they hold, with
   S − w · min(d², 4), where d = (rank of the file's title among them) − (rank of the column
   among them). Ranks are taken within that matched set, so extras ripped first (play-all,
   t00) shift nothing, and the quadratic makes every adjacent crossing cost exactly 2w whatever
   else is out of place (|d| ties as soon as one file is out of order). The cap bounds what
   one file can pay (4w), so a far crossing never costs more than 8w.
A disc already in title order pays nothing, so its score is unchanged. Confidence re-solves the
same objective: with evidence tied, the best alternative is an adjacent swap, so the bonus on
its own gives Δ ≤ 2w (0.6 nats at the default), far below high.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from showvenist.assign.basic import NEG, assign_basic
from showvenist.assign.types import BlockParams, BlockSolution, DiscSpec
from showvenist.scoring.types import FileConfidence, level_of

NEG_INF = float("-inf")
Forbidden = set[tuple[int, int]] | frozenset[tuple[int, int]]


def _ordered(discs: Sequence[DiscSpec]) -> list[DiscSpec]:
    if all(d.number is not None for d in discs):
        return sorted(discs, key=lambda d: d.number)  # stable: equal numbers keep given order
    return list(discs)


TITLE_ORDER_CAP = 4  # d² is capped here: one file pays at most 4w


def _title_order(
    sub: np.ndarray,
    titles: tuple[int | None, ...],
    cols: list[int],
    forb: set[tuple[int, int]],
    w: float,
) -> tuple[float, list[int]]:
    """Stage 2 of the tie-breaker (module docstring): (change in the block total, columns).
    `cols` are local block columns per disc file (−1 = unmatched)."""
    ks = [k for k, c in enumerate(cols) if c >= 0 and titles[k] is not None]
    if len(ks) < 2:
        return 0.0, cols
    rank = {k: r for r, k in enumerate(sorted(ks, key=lambda k: titles[k]))}
    held = sorted(cols[k] for k in ks)
    M = np.empty((len(ks), len(held)))
    for p, k in enumerate(ks):
        for q, c in enumerate(held):
            d = rank[k] - q
            M[p, q] = NEG if (k, c) in forb else sub[k, c] - w * min(d * d, TITLE_ORDER_CAP)
    rows, picks = linear_sum_assignment(M, maximize=True)
    before = float(sum(sub[k, cols[k]] for k in ks))  # stage 1's own matching is feasible
    out = list(cols)
    for p, q in zip(rows, picks, strict=True):
        out[ks[p]] = held[q]
    return float(M[rows, picks].sum()) - before, out


def _block_score_uncached(
    S: np.ndarray,
    files: tuple[int, ...],
    a: int,
    b: int,
    u: float,
    forbidden: frozenset,
    titles: tuple[int | None, ...] | None = None,
    w: float = 0.0,
) -> tuple[float, list[int] | None]:
    """One block evaluation: (score, column per disc file) or (−inf, None) if infeasible.
    `forbidden` holds global (row, column) pairs already filtered to this disc and block.
    With `titles` and w > 0, the matched files are re-matched in title order (stage 2)."""
    L = b - a
    if not files:
        return -u * L, []
    local = {f: k for k, f in enumerate(files)}
    sub = S[np.ix_(list(files), range(a, b))] + u
    forb = {(local[i], -1 if j == -1 else j - a) for i, j in forbidden}
    res = assign_basic(sub, forb)
    if not math.isfinite(res.total):
        return NEG_INF, None
    total, cols = res.total, list(res.column)
    if titles is not None and w > 0:
        change, cols = _title_order(sub, titles, cols, forb, w)
        total += change
    return total - u * L, [a + c if c >= 0 else -1 for c in cols]


class _BlockCache:
    """Block scores keyed by (disc key, title indices, a, b, forbidden pairs that can affect
    that block). The titles are part of the key, so one cache serves a disc solved with and
    without the title-order tie-breaker."""

    def __init__(self, S: np.ndarray, params: BlockParams):
        self.S = S
        self.params = params
        self.store: dict[tuple, tuple[float, list[int] | None]] = {}

    def score(
        self, disc: DiscSpec, a: int, b: int, touching: Sequence[tuple[int, int]]
    ) -> tuple[float, list[int] | None]:
        rel = frozenset((i, j) for i, j in touching if j == -1 or a <= j < b)
        w = self.params.title_order_weight
        titles = disc.titles if w > 0 else None
        key = (disc.key, titles, a, b, rel)
        hit = self.store.get(key)
        if hit is None:
            u = 0.0 if disc.incomplete else self.params.unclaimed_penalty
            hit = _block_score_uncached(self.S, disc.files, a, b, u, rel, titles, w)
            self.store[key] = hit
        return hit


def _infeasible(F: int) -> BlockSolution:
    return BlockSolution(column=[-1] * F, total=NEG_INF, blocks={})


def _solve(
    S: np.ndarray,
    discs: Sequence[DiscSpec],
    loose: Sequence[int],
    params: BlockParams,
    forbidden: Forbidden,
    cache: _BlockCache,
) -> BlockSolution:
    F, E = S.shape
    order = _ordered(discs)
    g = params.gap_penalty
    pos = np.arange(E + 1, dtype=float)

    prev = np.full(E + 1, NEG_INF)  # best total with the previous block ending at b'
    prev[0] = 0.0  # virtual start: the leading gap is measured from position 0
    back: list[tuple[dict[int, int], np.ndarray]] = []  # per disc: b -> a, and a -> best b'
    for k, d in enumerate(order):
        rows = set(d.files)
        touching = [(i, j) for i, j in forbidden if i in rows]
        free = d.free_gap_before or (k == 0 and d.number is not None and d.number > 1)
        # M[a] = max over b' ≤ a of prev[b'] − gap(b', a), with its argmax
        vals = prev if free else prev + g * pos
        arg = np.zeros(E + 1, dtype=int)
        M = np.empty(E + 1)
        best_v, best_i = NEG_INF, 0
        for a in range(E + 1):
            if vals[a] > best_v:
                best_v, best_i = vals[a], a
            M[a] = best_v if free else best_v - g * a
            arg[a] = best_i
        cur = np.full(E + 1, NEG_INF)
        choice: dict[int, int] = {}
        if d.fixed is not None:
            spans: Iterable[tuple[int, int]] = (
                [d.fixed] if 0 <= d.fixed[0] <= d.fixed[1] <= E else []
            )
        else:
            max_len = E if d.incomplete else len(d.files) + params.block_slack
            spans = ((a, b) for a in range(E + 1) for b in range(a, min(E, a + max_len) + 1))
        for a, b in spans:
            if M[a] == NEG_INF:
                continue
            s, _ = cache.score(d, a, b, touching)
            v = M[a] + s
            if v > cur[b]:
                cur[b], choice[b] = v, a
        back.append((choice, arg))
        prev = cur

    b_end = int(np.argmax(prev))
    dp_total = float(prev[b_end])
    if dp_total == NEG_INF:
        return _infeasible(F)

    column = [-1] * F
    blocks: dict[str, tuple[int, int]] = {}
    b = b_end
    for d, (choice, arg) in zip(reversed(order), reversed(back), strict=True):
        a = choice[b]
        rows = set(d.files)
        touching = [(i, j) for i, j in forbidden if i in rows]
        _, cols = cache.score(d, a, b, touching)
        assert cols is not None
        for f, c in zip(d.files, cols, strict=True):
            column[f] = c
        blocks[d.key] = (a, b)
        b = int(arg[a])

    total = dp_total
    loose = list(loose)
    if loose:
        claimed = {c for c in column if c >= 0}
        free_cols = [j for j in range(E) if j not in claimed]
        lidx = {f: k for k, f in enumerate(loose)}
        cidx = {j: k for k, j in enumerate(free_cols)}
        lforb = {
            (lidx[i], -1 if j == -1 else cidx[j])
            for i, j in forbidden
            if i in lidx and (j == -1 or j in cidx)
        }
        res = assign_basic(S[np.ix_(loose, free_cols)], lforb)
        if not math.isfinite(res.total):
            return _infeasible(F)
        for f, c in zip(loose, res.column, strict=True):
            column[f] = free_cols[c] if c >= 0 else -1
        total += res.total
    return BlockSolution(column=column, total=float(total), blocks=blocks)


def solve_blocks(
    S: np.ndarray,
    discs: Sequence[DiscSpec],
    loose: Sequence[int] = (),
    params: BlockParams = BlockParams(),  # noqa: B008 (frozen dataclass)
    forbidden: Forbidden | None = None,
) -> BlockSolution:
    """Best disc block assignment (module docstring). total is −inf when infeasible."""
    S = np.asarray(S, dtype=float)
    return _solve(S, discs, loose, params, frozenset(forbidden or ()), _BlockCache(S, params))


def _loose_alternative(
    S: np.ndarray,
    discs: Sequence[DiscSpec],
    loose: Sequence[int],
    params: BlockParams,
    i: int,
    sigma: int,
    cache: _BlockCache,
) -> tuple[float, int | None]:
    """Best total with loose file i pinned to some column c ≠ σ(i) (−1 = unmatched) and every
    disc file forbidden from c, blocks re-solved: (total, c), or (−inf, None) if none is
    feasible. Columns are tried in order of an upper bound, which prunes exactly."""
    E = S.shape[1]
    disc_rows = [f for d in discs for f in d.files]
    others = [f for f in loose if f != i]
    dp_bound = _solve(S, discs, (), params, frozenset(), cache).total
    others_bound = assign_basic(S[others]).total if others else 0.0
    if not math.isfinite(dp_bound):
        return NEG_INF, None
    cands = sorted(
        (c for c in range(-1, E) if c != sigma),
        key=lambda c: -(S[i, c] if c >= 0 else 0.0),
    )
    best, runner = NEG_INF, None
    for c in cands:
        bound = dp_bound + (S[i, c] if c >= 0 else 0.0) + others_bound
        if bound <= best:
            break  # sorted by S[i, c]: every later bound is lower
        forb = {(i, k) for k in range(-1, E) if k != c}
        if c >= 0:
            forb |= {(f, c) for f in disc_rows}
        alt = _solve(S, discs, loose, params, frozenset(forb), cache)
        if alt.total > best:
            best, runner = alt.total, c
    return best, runner


def block_confidence(
    S: np.ndarray,
    discs: Sequence[DiscSpec],
    solution: BlockSolution,
    loose: Sequence[int] = (),
    params: BlockParams = BlockParams(),  # noqa: B008
) -> list[FileConfidence]:
    """Δ_i = total − total with (i, σ(i)) forbidden, re-solving the block DP (§11.7). Block
    scores are shared across the re-solves, so only the disc holding file i is re-evaluated.

    Loose files use the pinned joint alternative (module docstring). When the constrained
    problem is infeasible (or the solution itself is), Δ = 0, level low, no runner-up: an
    impossible alternative must never read as confidence."""
    S = np.asarray(S, dtype=float)
    F = S.shape[0]
    cache = _BlockCache(S, params)
    loose_set = set(loose)
    in_solve = {f for d in discs for f in d.files} | loose_set
    out: list[FileConfidence] = []
    for i in range(F):
        if i not in in_solve:
            out.append(FileConfidence(0.0, "low", None, None))
            continue
        sigma = solution.column[i]
        if i in loose_set:
            alt_total, runner = _loose_alternative(S, discs, loose, params, i, sigma, cache)
        else:
            alt = _solve(S, discs, loose, params, frozenset({(i, sigma)}), cache)
            alt_total, runner = alt.total, alt.column[i]
        if not math.isfinite(alt_total) or not math.isfinite(solution.total):
            out.append(FileConfidence(0.0, "low", None, None))
            continue
        delta = max(0.0, solution.total - alt_total)
        out.append(FileConfidence(delta, level_of(delta), runner, delta))
    return out
