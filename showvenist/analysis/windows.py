"""Sampled dialogue windows (stage 2) and full-track reads (stage 3), spec §8.4."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from showvenist.analysis.ocr import (
    BitmapDecodeError,
    BitmapEvent,
    cues_from_texts,
    decode_events,
    ocr_cues,
    ocr_quality,
)
from showvenist.analysis.probe import probe
from showvenist.analysis.subtitles import normalize, read_text_cues
from showvenist.analysis.tracks import cue_count, rank_tracks
from showvenist.analysis.types import (
    EXTRACTOR_VERSIONS,
    AnalysisCache,
    Cue,
    Dialogue,
    MediaProbe,
    SubtitleReadError,
    SubtitleTrack,
    Window,
    params_hash,
)
from showvenist.config import AnalysisConfig
from showvenist.state.fingerprint import fingerprint

CHUNK_S = 60.0
# Below this many stage-2 lines, a track picked without a known event count is suspect (an
# untagged forced or near-empty track): the next unknown-count candidate is tried.
MIN_UNKNOWN_TRACK_LINES = 10


def window_spans(
    duration_s: float | None, fractions=(0.2, 0.5, 0.8), max_s: float = 300.0
) -> list[tuple[float, float]]:
    """Seed spans (start, upper-bound end) for the sampled windows. Each window stops at the
    next one's start so short files don't read the same stretch twice. With no known duration
    (0 or None) there is nothing to sample: one span covering the whole track (a full read)."""
    if not duration_s or duration_s <= 0:
        return [(0.0, math.inf)]
    starts = [duration_s * f for f in fractions]
    ends = [*starts[1:], duration_s]
    return [(a, min(duration_s, a + max_s, b)) for a, b in zip(starts, ends, strict=True)]


def _window_json(w: Window, read_to: float) -> str:
    return json.dumps(
        {
            "start_s": w.start_s,
            "end_s": w.end_s,
            "cues": [[c.start_s, c.end_s, c.text] for c in w.cues],
            "read_to": read_to,
        }
    )


def _window_load(payload: str) -> tuple[Window, float]:
    d = json.loads(payload)
    w = Window(d["start_s"], d["end_s"], [Cue(*c) for c in d["cues"]])
    return w, d["read_to"]


def _cue_key(c: Cue) -> tuple[float, str]:
    """Identity of a cue across chunk and window edges: OCR of the same event read twice may
    differ in end time (fallback length) and trivially in text."""
    return round(c.start_s, 1), normalize(c.text)


# reader(path, track, a, b, language, cfg) -> cues of [a, b]; raises SubtitleReadError.
Reader = Callable[[Path, SubtitleTrack, float, float, str, AnalysisConfig], list[Cue]]


def _read(path: Path, track: SubtitleTrack, a: float, b: float, language: str, cfg) -> list[Cue]:
    """The default reader: decode and OCR in the calling thread."""
    if track.kind == "text":
        return read_text_cues(path, track, a, b)
    return ocr_cues(path, track, a, b, language, cfg.ocr_workers)


# The same read split in two phases, so a pipeline can run them in different stages:
# decode_chunk is the I/O (demux, decode, render bitmaps), finish_chunk turns OCR texts into cues.


@dataclass
class DecodedChunk:
    """Result of the I/O phase of one chunk read. Text tracks are complete (``cues``); bitmap
    tracks carry ``events`` whose ``images`` still need OCR. ``error`` is a read failure part
    way: what was decoded before it is kept, as SubtitleReadError does."""

    track: SubtitleTrack
    cues: list[Cue] = field(default_factory=list)
    events: list[BitmapEvent] = field(default_factory=list)
    error: str | None = None

    @property
    def images(self) -> list:
        return [e.image for e in self.events]


def decode_chunk(path: Path, track: SubtitleTrack, a: float, b: float) -> DecodedChunk:
    """I/O phase of a chunk read. Never raises SubtitleReadError: it is recorded in ``error``."""
    if track.kind == "text":
        try:
            return DecodedChunk(track, cues=read_text_cues(path, track, a, b))
        except SubtitleReadError as e:
            return DecodedChunk(track, cues=e.cues, error=str(e))
    try:
        return DecodedChunk(track, events=decode_events(path, track, a, b))
    except BitmapDecodeError as e:
        return DecodedChunk(track, events=e.events, error=str(e))


def finish_chunk(d: DecodedChunk, texts: Sequence[str] = ()) -> list[Cue]:
    """Compute phase result of a chunk read: ``texts`` are the OCR of ``d.images`` (in order).
    Returns the cues, or raises SubtitleReadError with them when the decode failed, exactly like
    the default reader."""
    cues = d.cues if d.track.kind == "text" else cues_from_texts(d.events, texts)
    if d.error is not None:
        raise SubtitleReadError(d.error, cues)
    return cues


def _read_window(
    path, track, start, limit, language, cfg, min_lines, chunk=CHUNK_S, reader: Reader | None = None
) -> tuple[Window, float]:
    """Read from start in chunks until min_lines (0 = never stop early) or limit. A read error
    ends the window early: it keeps the cues read so far and records the error."""
    seen: set[tuple[float, str]] = set()
    cues: list[Cue] = []
    error: str | None = None

    def add(batch: list[Cue]) -> None:
        for c in batch:
            k = _cue_key(c)
            if k not in seen:
                seen.add(k)
                cues.append(c)

    pos = start
    while pos < limit:
        nxt = min(pos + chunk, limit)
        try:
            add((reader or _read)(path, track, pos, nxt, language, cfg))
        except SubtitleReadError as e:
            add(e.cues)
            error = str(e) or type(e).__name__
            break
        pos = nxt
        if min_lines and len(cues) >= min_lines:
            break
    end = max((c.end_s for c in cues), default=start)
    return Window(start, end, cues, error=error), pos


def _cached_window(cache, fp, key, compute) -> tuple[Window, float]:
    if cache is not None:
        hit = cache.get(fp, "window", key)
        if hit is not None:
            return _window_load(hit)
    w, read_to = compute()
    if cache is not None and w.error is None:  # a failed read is retried next run
        cache.put(fp, "window", key, _window_json(w, read_to))
    return w, read_to


def _extractor_version(track: SubtitleTrack) -> int:
    """Window payloads are produced by the text or OCR extractor: their version is part of the
    window key, so bumping either re-reads."""
    return EXTRACTOR_VERSIONS["text" if track.kind == "text" else "ocr"]


def _sample_windows(
    path, fp, track, dur, language, cfg, cache, reader: Reader | None = None
) -> list[tuple[Window, float]]:
    max_s = cfg.window_max_minutes * 60.0
    unknown_dur = not dur or dur <= 0
    wins: list[tuple[Window, float]] = []
    for start, limit in window_spans(dur, max_s=max_s):
        key = params_hash(
            track=track.index,
            start_s=start,
            limit=limit,
            language=language,
            min_lines=cfg.window_min_lines,
            max_minutes=cfg.window_max_minutes,
            extractor=_extractor_version(track),
        )
        wins.append(
            _cached_window(
                cache,
                fp,
                key,
                lambda s=start, e=limit: _read_window(
                    path,
                    track,
                    s,
                    e,
                    language,
                    cfg,
                    0 if unknown_dur else cfg.window_min_lines,
                    chunk=math.inf if unknown_dur else CHUNK_S,
                    reader=reader,
                ),
            )
        )
    return wins


def extract_dialogue(
    path: Path,
    language: str,
    vocabulary: set[str],
    cfg: AnalysisConfig,
    cache: AnalysisCache | None = None,
    full: bool = False,
    probe_result: MediaProbe | None = None,
    *,
    reader: Reader | None = None,
) -> Dialogue:
    """Stage 2 (sampled windows) or, with `full`, stage 3. `probe_result` saves re-probing when
    the caller already has it; otherwise the probe goes through `cache`. `reader` replaces the
    in-thread chunk read (the pipeline's reader gates the I/O per storage root and OCRs in a
    shared pool); it must return what the default reader would."""
    path = Path(path)
    info = probe_result if probe_result is not None else probe(path, cache)
    fp = fingerprint(path)
    ranked = rank_tracks(info.subtitle_tracks, language)
    if not ranked:
        return Dialogue(track=None, full=full)

    dur = info.duration_s
    notes: list[str] = []
    track = ranked[0]
    wins = _sample_windows(path, fp, track, dur, language, cfg, cache, reader)
    if cue_count(track) is None:
        # No frame-count tags: the pick is by stream order, so check it with the stage-2 lines
        # and fall back to the next unknown-count track while it reads almost nothing.
        others = [t for t in ranked[1:] if cue_count(t) is None]
        best = (track, wins)
        for alt in others:
            if sum(w.lines for w, _ in best[1]) >= MIN_UNKNOWN_TRACK_LINES:
                break
            alt_wins = _sample_windows(path, fp, alt, dur, language, cfg, cache, reader)
            if sum(w.lines for w, _ in alt_wins) > sum(w.lines for w, _ in best[1]):
                best = (alt, alt_wins)
        track, wins = best
        notes.append(
            f"track #{track.index} picked without an event count"
            + (f" (of {len(others) + 1} candidates)" if others else "")
        )
    if not dur or dur <= 0:
        notes.append("duration unknown: read the whole track instead of sampling")
        full = True

    max_s = cfg.window_max_minutes * 60.0
    if full and dur and dur > 0:
        # Gaps of [0, dur] not already covered by a sample window, read in chunks.
        covered = sorted((w.start_s, rt) for w, rt in wins)
        gaps: list[tuple[float, float]] = []
        pos = 0.0
        for a, b in covered:
            if a > pos:
                gaps.append((pos, a))
            pos = max(pos, b)
        if pos < dur:
            gaps.append((pos, dur))
        for ga, gb in gaps:
            s = ga
            while s < gb:
                e = min(gb, s + max_s)
                key = params_hash(
                    track=track.index,
                    start_s=s,
                    language=language,
                    full=e,
                    extractor=_extractor_version(track),
                )
                wins.append(
                    _cached_window(
                        cache,
                        fp,
                        key,
                        lambda s=s, e=e: _read_window(
                            path, track, s, e, language, cfg, 0, chunk=max_s, reader=reader
                        ),
                    )
                )
                s = e

    windows = sorted((w for w, _ in wins), key=lambda w: w.start_s)
    # Cues straddling window edges (and overlapping windows) would otherwise count twice.
    seen: set[tuple[float, str]] = set()
    for w in windows:
        kept = []
        for c in w.cues:
            k = _cue_key(c)
            if k not in seen:
                seen.add(k)
                kept.append(c)
        w.cues = kept
    notes += [f"window at {w.start_s:.1f} s: read failed: {w.error}" for w in windows if w.error]

    if track.kind == "text":
        quality: float | None = 1.0
    else:
        quality = ocr_quality([c.text for w in windows for c in w.cues], vocabulary, language)
    return Dialogue(
        track=track,
        windows=windows,
        ocr_quality=quality,
        full=full,
        note="; ".join(notes) or None,
    )
