"""Reference subtitle scorer (spec §9.2, decisions 32 and 38).

Feature: IDF-weighted containment of the file's word 5-gram shingles in each candidate's
reference subtitle,
    c(i, j) = sum idf(s) [s in S_i and S_j] / sum idf(s) [s in S_i].
IDF is computed across the candidate reference set, so shingles shared by every reference (theme
songs, recaps) are discounted. Mapping (raw nats, no clamp): c <= 0.02 -> -4, c >= 0.25 -> +12,
linear between. Both signs are scaled by a smooth dialogue factor (below); negative values are
also scaled by OCR quality (0 below 0.7, 1 above 0.9, linear between; None = text track = 1).
A candidate without a usable reference contributes exactly 0 (§9.1)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from showvenist.analysis.subtitles import normalize
from showvenist.bundle.schema import lang_codes
from showvenist.scoring.types import FileInfo, ScoreContext, ScoreResult

SHINGLE_N = 5
C_LOW, C_HIGH = 0.02, 0.25
LLR_LOW, LLR_HIGH = -4.0, 12.0
FULL_SHINGLES = 300  # dialogue factor reaches 1 at this many shingles
OCR_ZERO, OCR_FULL = 0.7, 0.9
CROSS_CHECK_MIN = 0.5
TOP_SHINGLES = 3

Shingles = frozenset[str]


def shingles_of(words: list[str]) -> Shingles:
    return frozenset(
        " ".join(words[k : k + SHINGLE_N]) for k in range(len(words) - SHINGLE_N + 1)
    )


def dialogue_factor(n_shingles: int) -> float:
    """Smoothstep on the shingle count: 0 at none, 1 from FULL_SHINGLES up."""
    s = min(1.0, max(0.0, n_shingles / FULL_SHINGLES))
    return s * s * (3.0 - 2.0 * s)


def ocr_factor(q: float | None) -> float:
    if q is None:
        return 1.0
    return min(1.0, max(0.0, (q - OCR_ZERO) / (OCR_FULL - OCR_ZERO)))


def map_containment(c: float) -> float:
    t = (c - C_LOW) / (C_HIGH - C_LOW)
    t = min(1.0, max(0.0, t))
    return LLR_LOW + t * (LLR_HIGH - LLR_LOW)


def _read_ref(path: Path) -> Shingles | None:
    import pysubs2

    try:
        subs = pysubs2.load(str(path), encoding="utf-8")
    except Exception:
        try:
            subs = pysubs2.load(str(path), encoding="latin-1")
        except Exception:
            return None
    words: list[str] = []
    for ev in subs:
        if ev.is_comment:
            continue
        words.extend(normalize(ev.plaintext.replace("\\N", " ")).split())
    sh = shingles_of(words)
    return sh or None


def reference_shingles(ctx: ScoreContext) -> dict[int, Shingles]:
    """candidate index -> shingles of its usable same-language reference."""
    out: dict[int, Shingles] = {}
    if ctx.bundle_dir is None:
        return out
    codes = lang_codes(ctx.language)
    for cand in ctx.candidates:
        for ref in cand.episode.subtitles:
            if ref.lang not in codes:
                continue
            p = ctx.bundle_dir / ref.path
            if not p.is_file():
                continue
            sh = _read_ref(p)
            if sh is not None:
                out[cand.index] = sh
                break
    return out


def _idf(refs: dict[int, Shingles]) -> dict[str, float]:
    df: dict[str, int] = {}
    for sh in refs.values():
        for s in sh:
            df[s] = df.get(s, 0) + 1
    n = len(refs)
    return {s: math.log((n + 1) / (d + 0.5)) for s, d in df.items()}


def _unseen_idf(n: int) -> float:
    return math.log((n + 1) / 0.5)


def _containment(
    sh: Shingles, ref: Shingles, idf: dict[str, float], n_refs: int
) -> tuple[float, list[tuple[float, str]]]:
    unseen = _unseen_idf(n_refs)
    total = 0.0
    hit = 0.0
    matched: list[tuple[float, str]] = []
    for s in sh:
        w = idf.get(s, unseen)
        total += w
        if s in ref:
            hit += w
            matched.append((w, s))
    return (hit / total if total > 0 else 0.0), matched


def _file_shingles(f: FileInfo) -> tuple[Shingles, list[Shingles]]:
    d = f.dialogue
    if d is None or d.track is None:
        return frozenset(), []
    per_window: list[Shingles] = []
    for w in d.windows:
        words: list[str] = []
        for c in w.cues:
            words.extend(normalize(c.text).split())
        per_window.append(shingles_of(words))
    allsh: set[str] = set()
    for p in per_window:
        allsh |= p
    return frozenset(allsh), per_window


def _llr(c: float, n_sh: int, ocr: float) -> float:
    v = map_containment(c)
    v *= dialogue_factor(n_sh)
    if v < 0:
        v *= ocr
    return v


class RefsubsScorer:
    name = "refsubs"
    can_be_negative = True

    def score(self, files: list[FileInfo], ctx: ScoreContext) -> ScoreResult:
        F, E = len(files), ctx.n_candidates
        res = ScoreResult.empty(F, E)
        refs = reference_shingles(ctx)
        parsed = [_file_shingles(f) for f in files]
        W = max((len(p[1]) for p in parsed), default=0)
        win = np.zeros((F, W, E))
        if not refs:
            res.window_llr = win if W else None
            return res
        idf = _idf(refs)
        n_refs = len(refs)
        for i, f in enumerate(files):
            sh, windows = parsed[i]
            if not sh:
                continue
            res.available[i] = True
            q = f.ocr_quality
            if q is None and f.dialogue is not None:
                q = f.dialogue.ocr_quality
            ocr = ocr_factor(q)
            for j, ref in refs.items():
                c, matched = _containment(sh, ref, idf, n_refs)
                res.llr[i, j] = _llr(c, len(sh), ocr)
                matched.sort(key=lambda m: (-m[0], m[1]))
                res.notes[(i, j)] = {
                    "containment": c,
                    "shingles": len(sh),
                    "top_shingles": [s for _, s in matched[:TOP_SHINGLES]],
                }
                for w, wsh in enumerate(windows):
                    if wsh:
                        cw, _ = _containment(wsh, ref, idf, n_refs)
                        win[i, w, j] = _llr(cw, len(wsh), ocr)
        res.window_llr = win if W else None
        return res


def cross_check(ctx: ScoreContext) -> list[tuple[str, str, float]]:
    """Pairs of candidate episode ids whose references resemble each other (containment of one
    in the other >= 0.5, IDF-weighted); a reference like that is probably mislabeled (§6.7).
    Returns (id_a, id_b, max containment) sorted by descending containment."""
    refs = reference_shingles(ctx)
    if len(refs) < 2:
        return []
    idf = _idf(refs)
    n = len(refs)
    ids = {c.index: c.id for c in ctx.candidates}
    out: list[tuple[str, str, float]] = []
    keys = sorted(refs)
    for a_pos, a in enumerate(keys):
        for b in keys[a_pos + 1 :]:
            c = max(
                _containment(refs[a], refs[b], idf, n)[0],
                _containment(refs[b], refs[a], idf, n)[0],
            )
            if c >= CROSS_CHECK_MIN:
                out.append((ids[a], ids[b], c))
    out.sort(key=lambda t: -t[2])
    return out


# the M3 contract names it RefSubsScorer
RefSubsScorer = RefsubsScorer
