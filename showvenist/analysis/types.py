"""M2 contract: data types and function signatures shared by the analysis modules (spec §7-8).

Module interfaces (each owned by one task; implement exactly these signatures):

analysis/probe.py
    probe(path: Path) -> MediaProbe
state/fingerprint.py
    fingerprint(path: Path) -> str          # "<size>:<blake2b hex of first+last 64 KB>" (§7.1)
    content_key(path: Path) -> str          # "<size>:<blake2b hex of 64 KB at 25/50/75%>"
analysis/tracks.py
    select_track(tracks: list[SubtitleTrack], language: str, min_events: int = 50)
        -> SubtitleTrack | None             # language is the bundle's ISO 639-1 code (§8.2)
analysis/subtitles.py
    read_text_cues(path: Path, track: SubtitleTrack, start_s: float, end_s: float) -> list[Cue]
    normalize(text: str) -> str             # §8.3 normalization
analysis/ocr.py
    ocr_cues(path: Path, track: SubtitleTrack, start_s: float, end_s: float, language: str,
             workers: int = 0) -> list[Cue]   # PGS and VobSub -> text, unnormalized
    ocr_quality(texts: list[str], vocabulary: set[str]) -> float | None
analysis/windows.py
    window_spans(duration_s: float, fractions=(0.2, 0.5, 0.8)) -> list[tuple[float, float]]
    extract_dialogue(path: Path, language: str, vocabulary: set[str], cfg: AnalysisConfig,
                     cache: AnalysisCache | None = None, full: bool = False) -> Dialogue
analysis/cache.py
    AnalysisCache (see below)
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

TEXT_CODECS = frozenset({"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"})
BITMAP_CODECS = {"hdmv_pgs_subtitle": "pgs", "dvd_subtitle": "vobsub"}

# Bump an extractor's version to invalidate its cached entries (§7.3).
EXTRACTOR_VERSIONS: dict[str, int] = {"probe": 1, "text": 1, "ocr": 1, "window": 1}


@dataclass(frozen=True)
class SubtitleTrack:
    index: int  # container stream index
    codec: str  # ffmpeg codec name
    lang: str | None  # ISO 639-2 tag as found ("eng"), None if untagged
    forced: bool = False
    sdh: bool = False
    events: int = 0  # packet count

    @property
    def kind(self) -> Literal["text", "pgs", "vobsub", "other"]:
        if self.codec in TEXT_CODECS:
            return "text"
        return BITMAP_CODECS.get(self.codec, "other")  # type: ignore[return-value]


@dataclass(frozen=True)
class MediaProbe:
    duration_s: float
    size: int
    container: str
    title_tag: str | None  # MKV title tag (§5.1)
    chapters: int
    subtitle_tracks: list[SubtitleTrack]
    video_codec: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class Cue:
    start_s: float
    end_s: float
    text: str


@dataclass
class Window:
    start_s: float  # requested seek point
    end_s: float  # last cue end actually read
    cues: list[Cue] = field(default_factory=list)

    @property
    def lines(self) -> int:
        return len(self.cues)


@dataclass
class Dialogue:
    """Result of stage 2 (sampled windows) or stage 3 (full track)."""

    track: SubtitleTrack | None  # None = no usable track: scorers report "not available"
    windows: list[Window] = field(default_factory=list)
    ocr_quality: float | None = None  # 1.0 for text tracks; None when no track
    full: bool = False

    @property
    def text(self) -> str:
        return " ".join(c.text for w in self.windows for c in w.cues)


def params_hash(**params: object) -> str:
    blob = repr(sorted(params.items())).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


class AnalysisCache:
    """analysis.db (§7.3): analysis(fingerprint, extractor, extractor_version, params_hash,
    payload, created_at). Payloads are JSON strings. Entries whose extractor_version differs from
    EXTRACTOR_VERSIONS are ignored on get() and may be purged. Never touches state.db."""

    def __init__(self, path: Path | str):
        self._db = sqlite3.connect(str(path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS analysis (fingerprint TEXT, extractor TEXT,"
            " extractor_version INTEGER, params_hash TEXT, payload TEXT, created_at TEXT,"
            " PRIMARY KEY (fingerprint, extractor, params_hash))"
        )

    def get(self, fingerprint: str, extractor: str, params: str) -> str | None:
        row = self._db.execute(
            "SELECT payload FROM analysis WHERE fingerprint=? AND extractor=? AND params_hash=?"
            " AND extractor_version=?",
            (fingerprint, extractor, params, EXTRACTOR_VERSIONS[extractor]),
        ).fetchone()
        return row[0] if row else None

    def put(self, fingerprint: str, extractor: str, params: str, payload: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO analysis VALUES (?,?,?,?,?,datetime('now'))",
            (fingerprint, extractor, EXTRACTOR_VERSIONS[extractor], params, payload),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()
