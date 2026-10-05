"""Subtitle track selection (spec §8.2)."""

from __future__ import annotations

from showvenist.analysis.types import SubtitleTrack
from showvenist.bundle.schema import lang_codes


def select_track(
    tracks: list[SubtitleTrack], language: str, min_events: int = 50
) -> SubtitleTrack | None:
    """Language match, drop forced / few-event / unreadable tracks, most events wins.

    `language` is the bundle's ISO 639-1 code; tracks carry ISO 639-2 tags. Ties go to text
    tracks (no OCR), then the lowest stream index.
    """
    lang = language.lower()
    wanted = lang_codes(lang)
    cands = [
        t
        for t in tracks
        if t.lang
        and t.lang.lower() in wanted
        and not t.forced
        and t.events >= min_events
        and t.kind != "other"
    ]
    if not cands:
        return None
    return min(cands, key=lambda t: (-t.events, t.kind != "text", t.index))
