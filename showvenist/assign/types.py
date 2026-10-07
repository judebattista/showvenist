"""M4 contract: the disc block solver's inputs and outputs (spec §11.5-11.6).

assign/blocks.py (sv-o2p.6):
    solve_blocks(S, discs, loose=(), params=BlockParams(), forbidden=None) -> BlockSolution
    block_confidence(S, discs, solution, loose=(), params=BlockParams()) -> list[FileConfidence]

assign/ordering.py (sv-o2p.6):
    detect_ordering(build, orderings, starting, params) -> OrderingChoice
        `build(ordering) -> (S, discs, loose)` is supplied by the Session: the fused score matrix
        for the candidates in THAT ordering (season filter recomputed, §11.3).

S is the fused (F, E) matrix of §11.1; columns are the candidates of the ordering being solved,
sorted by (season, number): column index = position in the matching order. `unmatched` has score
0 for every file (as in assign_basic). Files are rows of S.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from showvenist.scoring.types import FileConfidence

# Placeholders until tuned by eval in M4 (spec §18): nats.
UNCLAIMED_PENALTY_DEFAULT = 2.0  # u: a position inside a block that no file claims
GAP_PENALTY_DEFAULT = 2.0  # g: a position skipped between consecutive blocks
ORDERING_THRESHOLD_DEFAULT = 15.0  # block total short of basic total by more than this: try others
# Extra positions a complete disc's block may span beyond its ready-file count: a title MakeMKV
# skipped (minimum length) or a two-part episode in one file leaves an episode with no file of
# its own. Each extra position still costs u when unclaimed (§14 "Skipped titles").
BLOCK_SLACK_DEFAULT = 1
ORDERING_MARGIN_DEFAULT = 4.0


@dataclass(frozen=True)
class BlockParams:
    unclaimed_penalty: float = UNCLAIMED_PENALTY_DEFAULT
    gap_penalty: float = GAP_PENALTY_DEFAULT
    ordering_threshold: float = ORDERING_THRESHOLD_DEFAULT
    block_slack: int = BLOCK_SLACK_DEFAULT  # complete disc: block length ≤ files + block_slack
    # another ordering must beat the starting one by more than this to be chosen (hysteresis);
    # `ordering_threshold` is the separate bar for discarding blocks that fit badly
    ordering_margin: float = ORDERING_MARGIN_DEFAULT


@dataclass(frozen=True)
class DiscSpec:
    """One disc as the solver sees it. `files` are row indexes into S (ready files only)."""

    key: str
    files: tuple[int, ...]
    number: int | None  # discs are sorted by number; None-numbered discs keep the given order
    incomplete: bool = False  # u = 0 and the upper boundary is left open (§11.5)
    fixed: tuple[int, int] | None = None  # block [a, b) confirmed in history (M5)
    free_gap_before: bool = False  # disc numbers skip here (folders 1, 2, 4): the gap is free


@dataclass
class BlockSolution:
    column: list[int]  # per file (row of S): episode column, or -1 = unmatched
    total: float
    blocks: dict[str, tuple[int, int]] = field(default_factory=dict)  # disc key -> [a, b)
    used_blocks: bool = True  # False when it fell back to the basic assignment


@dataclass
class OrderingChoice:
    ordering: str
    solution: BlockSolution
    tried: dict[str, float] = field(default_factory=dict)  # ordering -> total
    switched: bool = False  # another ordering fits and beats the start by > the threshold
    fits_nowhere: bool = False  # nothing fits well enough: warn and use the start's basic


__all__ = [
    "BlockParams",
    "BlockSolution",
    "DiscSpec",
    "FileConfidence",
    "OrderingChoice",
]
