"""Hand-labeling sheet for the library's unnamed rips (spec §16.2, decision 41).

Walks a library read-only, probes every `*_tNN.mkv` rip, and writes a CSV to be labeled by hand:
an episode (`S01E05`), `extra`, or `episode?` (a regular episode, number unknown). Rows are
ordered per season so files near the show's episode length come first.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from showvenist.analysis.probe import probe
from showvenist.hierarchy.parse import walk_media
from showvenist.state.fingerprint import content_key

_RIP = re.compile(r"_t(\d\d)\.mkv$", re.I)
_NAMED = re.compile(r"[sS]\d{1,2}[eE]\d{1,3}")
_DISC = re.compile(r"disc[ _-]?(\d+)", re.I)
_SEASON = re.compile(r"season[ _-]?(\d+)", re.I)
NEAR = 0.20  # within 20% of the show's median named-episode length

COLUMNS = ["path", "show", "season", "disc", "title_no", "duration_min", "size_gb",
           "ref_min", "vs_episode", "hint", "title_tag", "same_as", "label"]
# Rows with `same_as` are byte-identical (same content key) to an earlier row: leave their label
# empty; the conversion to a manifest copies the original's label.


@dataclass
class Row:
    path: Path
    show: str
    season: str
    disc: str
    title_no: int
    duration_min: float
    size_gb: float
    title_tag: str
    ckey: str


def _show_of(path: Path, root: Path) -> str:
    return path.relative_to(root).parts[0]


def scan(root: Path) -> tuple[list[Row], dict[str, float]]:
    """Probe unnamed rips; return them with each show's median named-episode length (minutes)."""
    rips: list[Row] = []
    named: dict[str, list[float]] = defaultdict(list)
    for p in walk_media(root):
        show = _show_of(p, root)
        if _RIP.search(p.name):
            pr = probe(p)
            rel = "/".join(p.relative_to(root).parts)
            m_s, m_d = _SEASON.search(rel), _DISC.search(rel)
            rips.append(Row(
                p, show, f"S{int(m_s.group(1)):02d}" if m_s else "",
                m_d.group(1) if m_d else "", int(_RIP.search(p.name).group(1)),
                pr.duration_s / 60, pr.size / 1e9, pr.title_tag or "", content_key(p),
            ))
        elif _NAMED.search(p.name):
            named[show].append(probe(p).duration_s / 60)
    medians = {s: statistics.median(v) for s, v in named.items() if v}
    return rips, medians


def hint_for(ratio: float | None) -> str:
    if ratio is None:
        return "no-reference"
    if abs(ratio - 1) <= NEAR:
        return "episode-length"
    if ratio < 1 - NEAR:
        return "short"
    return "long"  # play-all titles and double episodes


def fallback_medians(rips: list[Row], medians: dict[str, float]) -> dict[str, float]:
    """For shows with no named files: the median of their unique rips of 15 to 70 minutes."""
    seen: set[str] = set()
    by_show: dict[str, list[float]] = defaultdict(list)
    for r in rips:
        if r.ckey in seen:
            continue
        seen.add(r.ckey)
        if 15 <= r.duration_min <= 70 and r.show not in medians:
            by_show[r.show].append(r.duration_min)
    return {**medians, **{s: statistics.median(v) for s, v in by_show.items()}}


def build_rows(rips: list[Row], medians: dict[str, float], root: Path) -> list[dict]:
    medians = fallback_medians(rips, medians)
    first_by_key: dict[str, Row] = {}
    out = []
    for r in rips:
        med = medians.get(r.show)
        ratio = r.duration_min / med if med else None
        dup = first_by_key.setdefault(r.ckey, r)
        out.append({
            "path": "/".join(r.path.relative_to(root).parts), "show": r.show, "season": r.season,
            "disc": r.disc, "title_no": r.title_no, "duration_min": f"{r.duration_min:.1f}",
            "size_gb": f"{r.size_gb:.2f}", "ref_min": "" if med is None else f"{med:.0f}",
            "vs_episode": "" if ratio is None else f"{ratio:.2f}",
            "hint": hint_for(ratio), "title_tag": r.title_tag,
            "same_as": "" if dup is r else "/".join(dup.path.relative_to(root).parts),
            "label": "",
        })
    rank = {"episode-length": 0, "long": 1, "no-reference": 1, "short": 2}
    out.sort(key=lambda d: (d["show"], d["season"], bool(d["same_as"]), rank[d["hint"]],
                            d["disc"].zfill(3),
                            d["title_no"], d["path"]))
    return out


def write_sheet(root: Path, dest: Path) -> list[dict]:
    rips, medians = scan(root)
    rows = build_rows(rips, medians, root)
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows
