"""Disc block assignment (spec §11.5) and its confidence by exact re-solve (§11.7).

Model (positions are the columns of S, in matching order; discs sorted by number):

    total = Σ_d block_score(d, a_d, b_d) − Σ_d gap(d, b_{d−1}, a_d) + loose total

    block_score(d, a, b) = max over one-to-one matchings of disc d's files to positions [a, b)
                           (any file may stay unmatched at 0) of Σ S − u · (unclaimed positions)

The penalty is optimized jointly with the matching: − u · (L − matched) = − u·L + Σ_matched u,
so the block's basic assignment runs on S + u and subtracts u·L. A disc marked incomplete has
u = 0 and an open upper boundary.

Block length bound: L ≤ len(files) + BLOCK_SLACK (spec: "bounded by the number of ready files").
A disc can claim FEWER positions than it has files (extras, duplicates stay unmatched), never
more than that bound; an incomplete disc may extend to the end of the season. Fixed blocks
(confirmed history) are forced to their [a, b) and ignore the bound.

Gaps: g per skipped position between consecutive blocks (and before the first block), free
where `free_gap_before` is set, free before the first disc if its number is > 1, always free
after the last disc (the DP's answer may end anywhere).

Disc order: by `number` when every disc has one; if any disc is unnumbered, the given order is
kept for all of them (the caller has ordered them, e.g. by folder creation time, §14).

Loose files (no disc) are assigned AFTER the DP, by the basic assignment over the positions no
disc file claimed (unclaimed positions inside blocks included). Their total is added.

Rows of S in neither a disc nor `loose` are outside the solve: column −1, low confidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

from showvenist.assign.basic import assign_basic
from showvenist.assign.types import BlockParams, BlockSolution, DiscSpec
from showvenist.scoring.types import FileConfidence, level_of

BLOCK_SLACK = 0  # extra positions a complete disc's block may span beyond its file count

NEG_INF = float("-inf")
Forbidden = set[tuple[int, int]] | frozenset[tuple[int, int]]


def _ordered(discs: Sequence[DiscSpec]) -> list[DiscSpec]:
    if all(d.number is not None for d in discs):
        return sorted(discs, key=lambda d: d.number)  # stable: equal numbers keep given order
    return list(discs)


def _block_score_uncached(
    S: np.ndarray, files: tuple[int, ...], a: int, b: int, u: float, forbidden: frozenset
) -> tuple[float, list[int] | None]:
    """One block evaluation: (score, column per disc file) or (−inf, None) if infeasible.
    `forbidden` holds global (row, column) pairs already filtered to this disc and block."""
    L = b - a
    if not files:
        return -u * L, []
    local = {f: k for k, f in enumerate(files)}
    sub = S[np.ix_(list(files), range(a, b))] + u
    forb = {(local[i], -1 if j == -1 else j - a) for i, j in forbidden}
    res = assign_basic(sub, forb)
    if not math.isfinite(res.total):
        return NEG_INF, None
    return res.total - u * L, [a + c if c >= 0 else -1 for c in res.column]


class _BlockCache:
    """Block scores keyed by (disc key, a, b, forbidden pairs that can affect that block)."""

    def __init__(self, S: np.ndarray, params: BlockParams):
        self.S = S
        self.params = params
        self.store: dict[tuple, tuple[float, list[int] | None]] = {}

    def score(
        self, disc: DiscSpec, a: int, b: int, touching: Sequence[tuple[int, int]]
    ) -> tuple[float, list[int] | None]:
        rel = frozenset((i, j) for i, j in touching if j == -1 or a <= j < b)
        key = (disc.key, a, b, rel)
        hit = self.store.get(key)
        if hit is None:
            u = 0.0 if disc.incomplete else self.params.unclaimed_penalty
            hit = _block_score_uncached(self.S, disc.files, a, b, u, rel)
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
            max_len = E if d.incomplete else len(d.files) + BLOCK_SLACK
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


def block_confidence(
    S: np.ndarray,
    discs: Sequence[DiscSpec],
    solution: BlockSolution,
    loose: Sequence[int] = (),
    params: BlockParams = BlockParams(),  # noqa: B008
) -> list[FileConfidence]:
    """Δ_i = total − total with (i, σ(i)) forbidden, re-solving the block DP (§11.7). Block
    scores are shared across the re-solves, so only the disc holding file i is re-evaluated."""
    S = np.asarray(S, dtype=float)
    F = S.shape[0]
    cache = _BlockCache(S, params)
    in_solve = {f for d in discs for f in d.files} | set(loose)
    out: list[FileConfidence] = []
    for i in range(F):
        if i not in in_solve:
            out.append(FileConfidence(0.0, "low", None, None))
            continue
        alt = _solve(S, discs, loose, params, frozenset({(i, solution.column[i])}), cache)
        if not math.isfinite(alt.total) or not math.isfinite(solution.total):
            delta, runner, rdelta = math.inf, None, None
        else:
            delta = max(0.0, solution.total - alt.total)
            runner, rdelta = alt.column[i], delta
        out.append(FileConfidence(delta, level_of(delta), runner, rdelta))
    return out
