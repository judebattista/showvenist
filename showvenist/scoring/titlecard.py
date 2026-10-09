"""Title-card scorer (spec §9.2, decision 54). Contract; the body is task sv-2k9.

For each file with `card_reads` not None: apply `cfg.pattern` to each read's text, normalize the
`ref` group per `cfg.kind` (roman/number: look-alike characters l, |, 1 -> I where a numeral is
expected; a run of exactly four I, which no title has, is read as three), and compare against
the candidates' titles or numbers as a closed set (kind "title": fuzzy ratio >= 90 like the title
scorer). Frames combine by consensus: a reading counts when two
or more frames agree or it matches exactly one candidate; readings that disagree, or match
nothing, give 0. Raw LLR is `MATCH_LLR` on the single matched candidate, never negative.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from showvenist.scoring.title import MATCH_RATIO, is_generic
from showvenist.scoring.types import FileInfo, ScoreContext, ScoreResult

MATCH_LLR = 8.0

_ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")
_LOOKALIKE = str.maketrans({"l": "I", "|": "I", "1": "I", "!": "I"})


def normalize_roman(text: str) -> str | None:
    """Uppercase roman numeral from OCR text (look-alikes l | 1 ! -> I), or None if invalid."""
    t = re.sub(r"[^A-Za-z]", "", text.translate(_LOOKALIKE).upper())
    return t if t and _ROMAN.match(t) else None


def repair_roman(text: str) -> str | None:
    """OCR over-counts a trailing III as four strokes (real reads: XXIIII for XXIII, XCIIII for
    XCIII). `IIII` is not a numeral in any title, so a run of exactly four I becomes three (decision
    54). Only an otherwise invalid reading is repaired; longer or shorter runs are left alone."""
    t = re.sub(r"[^A-Za-z]", "", text.translate(_LOOKALIKE).upper())
    if not t or _ROMAN.match(t) or len(re.findall(r"I{4,}", t)) != 1:
        return None
    if max(map(len, re.findall(r"I+", t))) != 4:
        return None
    fixed = re.sub(r"I{4}", "III", t)
    return fixed if _ROMAN.match(fixed) else None


def _digits(text: str) -> str | None:
    d = re.sub(r"\D", "", text)
    return d.lstrip("0") or ("0" if d else None)


class TitleCardScorer:
    name = "titlecard"
    can_be_negative = False

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        res = ScoreResult.empty(len(files), ctx.n_candidates)
        cfg = ctx.bundle.title_card
        if cfg is None:
            return res
        rx = re.compile(cfg.pattern, re.I)
        show = ctx.bundle.show.title
        names: list[list[str]] = []  # per candidate: comparable titles
        for c in ctx.candidates:
            ts = [t for t in (c.episode.title, *c.episode.alt_titles) if t and t.strip()]
            if cfg.kind == "roman":
                norm = [normalize_roman(t) for t in ts]
            elif cfg.kind == "number":
                norm = [_digits(t) if t.strip().isdigit() else None for t in ts]
            else:
                norm = [t.casefold().strip() for t in ts if not is_generic(t, show)]
            names.append([n for n in norm if n])

        for i, f in enumerate(files):
            if f.card_reads is None:
                continue
            res.available[i] = True
            matched: dict[int, list[str]] = {}  # candidate -> readings that matched it
            exact = False
            for r in f.card_reads:
                m = rx.search(r.text)
                if not m or not m.group("ref"):
                    continue
                ref = m.group("ref")
                if cfg.kind == "roman":
                    reading = normalize_roman(ref) or repair_roman(ref)
                elif cfg.kind == "number":
                    reading = _digits(ref)
                else:
                    reading = ref.casefold().strip() or None
                if reading is None:
                    continue
                for j, ts in enumerate(names):
                    if cfg.kind == "title":
                        score = max((fuzz.ratio(reading, t) for t in ts), default=0.0)
                        hit = score >= MATCH_RATIO
                        exact_hit = score >= 100.0
                    else:
                        hit = exact_hit = reading in ts
                    if hit:
                        matched.setdefault(j, []).append(reading)
                        exact = exact or exact_hit
            if len(matched) != 1:
                continue  # nothing matched, or reads disagree / duplicate titles: ambiguous
            (j, readings), = matched.items()
            if len(readings) < 2 and not exact:
                continue
            res.llr[i, j] = MATCH_LLR
            res.notes[(i, j)] = {"readings": readings, "reads": len(f.card_reads)}
        return res
