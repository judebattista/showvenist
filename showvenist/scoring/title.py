"""Track/container title scorer (spec §9.2): fuzzy match of the MKV title tag against episode
titles. For evidence only; generic disc names and placeholders never score."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from showvenist.scoring.types import FileInfo, ScoreContext, ScoreResult

MATCH_RATIO = 90.0
MATCH_LLR = 6.0

_GENERIC = [
    re.compile(p, re.I)
    for p in (
        r"(?<![a-z])(disc|disk|dvd|blu-?ray)(?![a-z])",
        r"^(season|series|s)\s*\d+\b",
        r"^(title|track|chapter|episode|ep|part|pt|volume|vol|clip|video|t|c)[\s._-]*\d+$",
        r"^[a-z]?\d+$",
        r"^(title|track|chapter)[\s._-]*[a-z]*\d*$",
        r"^s\d+\s*e\d+$",
        r"^(untitled|unknown|main|movie|feature|extras?|bonus|play all|playall)$",
    )
]


def is_generic(text: str | None, show_title: str | None = None) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if show_title and t.casefold() == show_title.strip().casefold():
        return True
    return any(p.search(t) for p in _GENERIC)


class TitleScorer:
    name = "title"
    can_be_negative = False

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        res = ScoreResult.empty(len(files), ctx.n_candidates)
        show = ctx.bundle.show.title
        names: list[list[str]] = []
        for c in ctx.candidates:
            ts = [c.episode.title, *c.episode.alt_titles]
            names.append([t.casefold().strip() for t in ts if not is_generic(t, show)])
        for i, f in enumerate(files):
            tag = (f.title_tag or "").strip()
            if not tag:
                continue
            res.available[i] = True
            if is_generic(tag, show):
                continue
            tag_cf = tag.casefold()
            for j, ts in enumerate(names):
                if any(fuzz.ratio(tag_cf, t) >= MATCH_RATIO for t in ts):
                    res.llr[i, j] = MATCH_LLR
        return res
