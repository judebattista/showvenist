"""Subtitle track selection (spec §8.2)."""

from __future__ import annotations

from showvenist.analysis.types import SubtitleTrack
from showvenist.bundle.schema import lang_codes


def cue_count(track: SubtitleTrack) -> int | None:
    """Estimated number of cues from the packet count; None when the count is unknown.

    A PGS cue is two display sets (show, then an empty clear), so its packets are halved to be
    comparable with text and VobSub tracks (one packet per cue)."""
    if track.events < 0:
        return None
    return track.events // 2 if track.kind == "pgs" else track.events


def rank_tracks(
    tracks: list[SubtitleTrack], language: str, min_events: int = 50
) -> list[SubtitleTrack]:
    """Usable tracks, best first: language match, not forced, readable, and either at least
    `min_events` estimated cues or an unknown count.

    Known counts rank first (most cues; ties to text tracks, then lowest stream index). Unknown
    counts (no frame-count tag; counting would read the whole file) follow, non-SDH first, then
    stream order: stage 2 line counts decide between them (see windows.extract_dialogue)."""
    wanted = lang_codes(language.lower())
    cands = [
        t
        for t in tracks
        if t.lang and t.lang.lower() in wanted and not t.forced and t.kind != "other"
    ]
    known = [t for t in cands if (n := cue_count(t)) is not None and n >= min_events]
    unknown = [t for t in cands if cue_count(t) is None]
    known.sort(key=lambda t: (-(cue_count(t) or 0), t.kind != "text", t.index))
    unknown.sort(key=lambda t: (t.sdh, t.index))
    return known + unknown


def select_track(
    tracks: list[SubtitleTrack], language: str, min_events: int = 50
) -> SubtitleTrack | None:
    """The best of `rank_tracks`, or None. `language` is the bundle's ISO 639-1 code; tracks
    carry ISO 639-2 tags."""
    ranked = rank_tracks(tracks, language, min_events)
    return ranked[0] if ranked else None
