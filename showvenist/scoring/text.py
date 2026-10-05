"""Tokenization and BM25 helpers shared by the synopsis, guest-character and cues scorers.

Ported from the M0 spike (spike/score.py): same token rules, stopword threshold, BM25 constants
and robust z-score, so spike results carry over.
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter
from collections.abc import Iterable

import numpy as np

from showvenist.analysis.ocr import is_word
from showvenist.analysis.subtitles import normalize
from showvenist.analysis.types import Dialogue, Window

STOP_ZIPF = 5.5  # tokens at/above this Zipf frequency are stopwords for BM25
SYN_PER_SIGMA = 0.8
_PAREN = re.compile(r"\([^)]*\)")
_NAME_SPLIT = re.compile(r"[/;]")


def tokens(text: str) -> list[str]:
    """Normalized word tokens: alphabetic, length >= 3 (plus 'i'). Apostrophes are dropped."""
    return [t for t in normalize(text or "").split() if t.isalpha() and (len(t) >= 3 or t == "i")]


def name_tokens(name: str) -> list[str]:
    """Tokens of a person/character name; parenthesised notes ("(voice)") are ignored."""
    return tokens(_PAREN.sub(" ", name or ""))


def name_parts(name: str) -> list[str]:
    """A character field may hold several names separated by '/' or ';'."""
    return [p for p in _NAME_SPLIT.split(name or "") if p.strip()]


@functools.lru_cache(maxsize=65536)
def _zipf(tok: str, language: str) -> float:
    from wordfreq import zipf_frequency

    return zipf_frequency(tok, language)


def is_stopword(tok: str, language: str = "en") -> bool:
    return _zipf(tok, language) >= STOP_ZIPF


def is_dictionary_word(tok: str, language: str = "en") -> bool:
    return is_word(tok, language)


def content_tokens(text: str, strip: set[str], language: str = "en") -> list[str]:
    """Tokens of `text` minus `strip` (cast names) and stopwords."""
    return [t for t in tokens(text) if t not in strip and not is_stopword(t, language)]


def dialogue_text(d: Dialogue | None) -> str:
    return "" if d is None else d.text


def window_text(w: Window) -> str:
    return " ".join(c.text for c in w.cues)


def has_dialogue(d: Dialogue | None) -> bool:
    return d is not None and d.track is not None


class Bm25:
    """BM25 over a fixed document set; the query is deduplicated (repeats don't dominate)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = max(float(np.mean([len(d) for d in docs])) if docs else 0.0, 1e-9)
        self.df = Counter(t for d in docs for t in set(d))
        self.tfs = [Counter(d) for d in docs]

    def idf(self, t: str) -> float:
        n = self.df[t]
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def term_score(self, t: str, j: int) -> float:
        f = self.tfs[j].get(t, 0)
        if not f or t not in self.df:
            return 0.0
        k1, b = self.k1, self.b
        return self.idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(self.docs[j]) / self.avgdl))

    def scores(self, query: Iterable[str]) -> np.ndarray:
        out = np.zeros(self.N)
        for t in set(query):
            if t in self.df:
                for j in range(self.N):
                    out[j] += self.term_score(t, j)
        return out

    def top_terms(self, query: Iterable[str], j: int, k: int = 5) -> list[str]:
        sc = {t: self.term_score(t, j) for t in set(query) if t in self.df}
        return [t for t, v in sorted(sc.items(), key=lambda x: (-x[1], x[0])) if v > 0][:k]


def chance_max(E: int) -> float:
    return math.sqrt(2 * math.log(E)) if E > 1 else 0.0


def bm25_llr(bm: Bm25, query: list[str]) -> np.ndarray:
    """Robust z-score of BM25 within the file, credited 0.8 nats per sigma above the expected
    chance maximum over E candidates; raw (unclamped) and never negative."""
    E = bm.N
    s = bm.scores(query)
    if E < 3 or not np.any(s):
        return np.zeros(E)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    if mad < 1e-9:
        mad = max(float(s.std()), 1e-9)
    z = (s - med) / mad
    zc = chance_max(E)
    return np.where(z > zc, SYN_PER_SIGMA * (z - zc), 0.0)
