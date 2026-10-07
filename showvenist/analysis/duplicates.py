"""Near-duplicate rips by dialogue overlap (spec §14 "Duplicate rips").

Exact duplicates share a content key (§7.1) and are set aside before analysis. A near duplicate
is the same episode ripped twice with different bytes (another playlist or title of the same
disc, a copy with an extra recap): its content key differs, so it is found here, from the
sampled dialogue the analysis already read. Never by duration or size: a season's episodes
often share a runtime to the second.

Each file's sampled cues are normalized (§8.3) and cut into word 5-gram shingles; two files
overlap by containment, |A ∩ B| / min(|A|, |B|), so a copy with an extra recap still contains
the shorter one. Files whose overlap reaches the threshold are linked, and the groups are the
connected components.

Limits: the sampled windows sit at fixed fractions of each file's runtime (§8.4), so two
copies of the same length sample the same scenes and overlap almost completely, while copies of
different lengths sample partly different scenes and overlap less. Detection is reliable for
same-length copies and best-effort otherwise; reading more of the files to compensate is a
deliberate non-goal (it would cost a full extraction per file).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from showvenist.analysis.subtitles import normalize
from showvenist.analysis.types import Dialogue

SHINGLE_N = 5  # word n-grams, as the reference-subtitle scorer (§9.2)

Shingles = frozenset[str]


def shingle_set(dialogue: Dialogue | None) -> Shingles:
    """Word 5-gram shingles of every sampled window's cues, normalized (§8.3). Shingles don't
    span two windows (the gap between them is not dialogue)."""
    out: set[str] = set()
    if dialogue is None:
        return frozenset()
    for w in dialogue.windows:
        words: list[str] = []
        for c in w.cues:
            words.extend(normalize(c.text).split())
        out.update(
            " ".join(words[k : k + SHINGLE_N]) for k in range(len(words) - SHINGLE_N + 1)
        )
    return frozenset(out)


def overlap(a: Shingles, b: Shingles) -> float:
    """Containment of the smaller set in the larger: |a ∩ b| / min(|a|, |b|); 0 if either is
    empty."""
    if not a or not b:
        return 0.0
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    return len(small & large) / len(small)


def find_near_duplicates(
    shingles: Mapping[Path, Shingles], threshold: float, min_shingles: int
) -> list[list[Path]]:
    """Groups (connected components, two files or more) of files whose pairwise overlap
    reaches `threshold`. Files with fewer than `min_shingles` shingles are never grouped: too
    little dialogue to tell a copy from a shared scene. Members are in path order, groups by
    their first member."""
    paths = sorted(p for p, s in shingles.items() if len(s) >= min_shingles)
    parent = {p: p for p in paths}

    def root(p: Path) -> Path:
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for i, a in enumerate(paths):
        sa = shingles[a]
        for b in paths[i + 1 :]:
            if overlap(sa, shingles[b]) >= threshold:  # O(n²) set ops: ~100 files is cheap
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    groups: dict[Path, list[Path]] = {}
    for p in paths:
        groups.setdefault(root(p), []).append(p)
    return sorted((g for g in groups.values() if len(g) > 1), key=lambda g: g[0])


@dataclass(frozen=True)
class Quality:
    """What makes one copy the better primary: dialogue lines read in the sampled windows,
    their OCR quality (None = text track = 1.0), and the file size in bytes."""

    lines: int = 0
    ocr_quality: float | None = None
    size: int = 0

    @property
    def usable_lines(self) -> float:
        return self.lines * (1.0 if self.ocr_quality is None else self.ocr_quality)


def choose_primary(group: Iterable[Path], quality: Mapping[Path, Quality]) -> Path:
    """The copy that is analyzed and solved: the most usable dialogue (lines weighted by OCR
    quality), then the larger file, then path order."""
    def key(p: Path):
        q = quality.get(p, Quality())
        return (-q.usable_lines, -q.size, p)

    return min(group, key=key)


__all__ = [
    "Quality",
    "Shingles",
    "choose_primary",
    "find_near_duplicates",
    "overlap",
    "shingle_set",
]
