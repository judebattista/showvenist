"""Title-order evidence (spec §11.5 "Title-order tie-breaker", decision 63).

Whether MakeMKV's title order (`_tNN`) can break ties inside disc blocks for this season. The
verdict is STRICT: every pair of confidently placed files in one disc (or one disc folder of an
earlier run, from the move journal) is concordant when title order and episode order agree and
discordant when they disagree. Any discordant pair anywhere turns the tie-breaker off for the
whole season; all concordant, or no pairs at all, turns it on. Pairs never span two groups, and
two files of one episode (or one title) make no pair.

The caller collects the groups:
- this season: each disc's high- or medium-confidence files (and pinned or confirmed rows) of
  a solve WITHOUT the tie-breaker, keyed by their episode position in the matching order;
- history: `History.confirmed_moves`, grouped by the parent folder of `old_path`
  (`history_groups`), keyed by (season, episode) as recorded.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from showvenist.hierarchy.parse import parse_title_index

Item = tuple[int, Any]  # (title index, episode order key: anything totally ordered)


@dataclass(frozen=True)
class TitleOrderVerdict:
    use: bool
    agree: int = 0  # groups with at least one pair, every pair concordant
    against: str | None = None  # the first group holding a discordant pair
    reverse: bool = False  # that group's pairs are ALL discordant (it runs in reverse)
    disagree: int = 0  # groups holding a discordant pair

    def line(self) -> str:
        """The evidence-pane line (§11.5 item 4)."""
        if not self.use:
            how = "runs in reverse" if self.reverse else "is out of title order"
            return f"title order not used; {self.against} {how}"
        if self.agree == 0 and not self.disagree:
            return "title order used as a tie-breaker; no evidence yet"
        if self.disagree:
            return (f"title order used as a tie-breaker; {self.agree} of "
                    f"{self.agree + self.disagree} discs agree ({self.against} does not)")
        discs = "1 disc agrees" if self.agree == 1 else f"{self.agree} discs agree"
        return f"title order used as a tie-breaker; {discs}"


def _pairs(items: Sequence[Item]) -> tuple[int, int]:
    """(concordant, discordant) pairs among one group's items."""
    con = dis = 0
    for k, (t1, e1) in enumerate(items):
        for t2, e2 in items[k + 1:]:
            if t1 == t2 or e1 == e2:
                continue
            if (t1 < t2) == (e1 < e2):
                con += 1
            else:
                dis += 1
    return con, dis


MAJORITY = 2 / 3  # share of groups with pairs that must be fully concordant (decision 65)


def judge(groups: Mapping[str, Iterable[Item]]) -> TitleOrderVerdict:
    """The verdict over `groups` (label -> items), in the mapping's order: on when at least
    MAJORITY of the groups that hold any pair are fully concordant (a group with one discordant
    pair is discordant), and when there are no pairs at all."""
    agree = disagree = 0
    against: str | None = None
    reverse = False
    for label, items in groups.items():
        con, dis = _pairs(list(items))
        if dis:
            disagree += 1
            if against is None:
                against, reverse = label, con == 0
        elif con:
            agree += 1
    total = agree + disagree
    use = total == 0 or agree / total >= MAJORITY
    return TitleOrderVerdict(use, agree, against, reverse, disagree)


def _folder_label(folder: Path) -> str:
    return f"{folder.parent.name}/{folder.name}" if folder.parent.name else folder.name


def history_groups(moves: Iterable[tuple[str, int, int]]) -> dict[str, list[Item]]:
    """Moves (old_path, season, episode) grouped by the old path's folder, labelled
    "Season 1/Disc 2"; old paths without a title index take no part."""
    by_folder: dict[Path, list[Item]] = {}
    for old, season, episode in moves:
        p = Path(old)
        t = parse_title_index(p.name)
        if t is None:
            continue
        by_folder.setdefault(p.parent, []).append((t, (season, episode)))
    return {_folder_label(f): items for f, items in sorted(by_folder.items())}


__all__ = ["TitleOrderVerdict", "history_groups", "judge"]
