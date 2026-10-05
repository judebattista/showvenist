"""Candidate set for a solve (spec §11.3): season and --episodes are hard filters."""

from __future__ import annotations

from collections.abc import Collection

from showvenist.bundle.schema import Bundle
from showvenist.scoring.types import Candidate


def build_candidates(
    bundle: Bundle,
    ordering: str = "aired",
    season: int | None = None,
    episodes: Collection[tuple[int, int]] | None = None,
    include_specials: bool = False,
) -> list[Candidate]:
    """Episodes of `ordering`, numbered in that ordering, sorted by (season, number).

    - `season` keeps one season (as numbered in `ordering`); None keeps all.
    - `episodes` is a set of (season, number) pairs that overrides the season filter.
    - Season 0 (specials) only with include_specials or when `season == 0`.
    - Episodes flagged `excluded` (overrides) are never candidates.
    """
    order = bundle.orderings[ordering]
    by_id = bundle.episode_by_id()
    entries = sorted(order.entries, key=lambda e: (e.season, e.episode))
    out: list[Candidate] = []
    for ent in entries:
        ep = by_id.get(ent.episode_id)
        if ep is None or ep.excluded:
            continue
        if episodes is not None:
            if (ent.season, ent.episode) not in episodes:
                continue
        else:
            if season is not None and ent.season != season:
                continue
            if ent.season == 0 and not include_specials and season != 0:
                continue
        out.append(Candidate(len(out), ep, ent.season, ent.episode))
    return out
