"""Ordering detection (spec §11.6, decision 46).

Every ordering of the bundle is solved with blocks (it costs milliseconds). An ordering FITS
when its block total is within `ordering_threshold` of its own basic total. Totals include the
prior, so they are on one scale, but only orderings whose S has the same rows as the starting
one are comparable.

Rule:
1. Switch to the best-scoring other ordering that fits AND whose block total beats the starting
   ordering's block total by MORE than the threshold (hysteresis: a better fit has to be
   clearly better). A non-fitting ordering never wins, however high its total.
2. Otherwise keep the starting ordering if it fits.
3. Otherwise the result is the starting ordering's basic assignment (used_blocks=False,
   fits_nowhere=True) and the caller warns that the discs don't look sequential (§11.6 step 4).
   This also covers a start that doesn't fit while some ordering fits without beating the start
   by the threshold: such an ordering's block total is below the start's basic total anyway.

The earlier rule looked at other orderings only when the starting one fitted worse than its own
basic assignment. That misses a wrong season filter: Firefly's aired order has 11 candidates in
season 1 for 14 files, so blocks and basic fail together and stay close.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from showvenist.assign.basic import assign_basic
from showvenist.assign.blocks import solve_blocks
from showvenist.assign.types import BlockParams, BlockSolution, DiscSpec, OrderingChoice

Built = tuple[np.ndarray, Sequence[DiscSpec], Sequence[int]]


def _basic(S: np.ndarray, discs: Sequence[DiscSpec], loose: Sequence[int]) -> BlockSolution:
    """Basic assignment over the rows that take part in the solve (disc files and loose)."""
    rows = sorted({f for d in discs for f in d.files} | set(loose))
    a = assign_basic(S[rows]) if rows else None
    column = [-1] * S.shape[0]
    if a is not None:
        for r, c in zip(rows, a.column, strict=True):
            column[r] = c
    return BlockSolution(column, a.total if a else 0.0, {}, used_blocks=False)


def detect_ordering(
    build: Callable[[str], Built],
    orderings: Iterable[str],
    starting: str,
    params: BlockParams = BlockParams(),  # noqa: B008 (frozen dataclass)
) -> OrderingChoice:
    S0, discs0, loose0 = build(starting)
    S0 = np.asarray(S0, dtype=float)
    sol0 = solve_blocks(S0, discs0, loose0, params)
    basic0 = _basic(S0, discs0, loose0)
    tried = {starting: sol0.total}

    def fits(sol: BlockSolution, basic: BlockSolution) -> bool:
        return sol.total >= basic.total - params.ordering_threshold

    best: tuple[str, BlockSolution] | None = None  # best FITTING other ordering
    for name in orderings:
        if name == starting or name in tried:
            continue
        S, discs, loose = build(name)
        S = np.asarray(S, dtype=float)
        if S.shape[0] != S0.shape[0]:
            continue  # different files: totals are not comparable
        sol = solve_blocks(S, discs, loose, params)
        tried[name] = sol.total
        if fits(sol, _basic(S, discs, loose)) and (best is None or sol.total > best[1].total):
            best = (name, sol)

    if best is not None and best[1].total > sol0.total + params.ordering_margin:
        return OrderingChoice(best[0], best[1], tried, switched=True)
    if fits(sol0, basic0):
        return OrderingChoice(starting, sol0, tried)
    return OrderingChoice(starting, basic0, tried, switched=False, fits_nowhere=True)
