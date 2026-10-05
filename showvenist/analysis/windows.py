"""Sampled dialogue windows (stage 2) and full-track reads (stage 3), spec §8.4."""

from __future__ import annotations

import json
from pathlib import Path

from showvenist.analysis.ocr import ocr_cues, ocr_quality
from showvenist.analysis.probe import probe
from showvenist.analysis.subtitles import read_text_cues
from showvenist.analysis.tracks import select_track
from showvenist.analysis.types import (
    AnalysisCache,
    Cue,
    Dialogue,
    SubtitleTrack,
    Window,
    params_hash,
)
from showvenist.config import AnalysisConfig
from showvenist.state.fingerprint import fingerprint

CHUNK_S = 60.0


def window_spans(
    duration_s: float, fractions=(0.2, 0.5, 0.8), max_s: float = 300.0
) -> list[tuple[float, float]]:
    """Seed spans (start, upper-bound end) for the sampled windows."""
    return [(duration_s * f, min(duration_s, duration_s * f + max_s)) for f in fractions]


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


def _read(path: Path, track: SubtitleTrack, a: float, b: float, language: str, cfg) -> list[Cue]:
    if track.kind == "text":
        return read_text_cues(path, track, a, b)
    return ocr_cues(path, track, a, b, language, cfg.ocr_workers)


def _read_window(
    path, track, start, limit, language, cfg, min_lines, chunk=CHUNK_S
) -> tuple[Window, float]:
    """Read from start in chunks until min_lines (0 = never stop early) or limit."""
    seen: set[tuple[float, float, str]] = set()
    cues: list[Cue] = []
    pos = start
    while pos < limit:
        nxt = min(pos + chunk, limit)
        for c in _read(path, track, pos, nxt, language, cfg):
            key = (c.start_s, c.end_s, c.text)
            if key not in seen:
                seen.add(key)
                cues.append(c)
        pos = nxt
        if min_lines and len(cues) >= min_lines:
            break
    end = max((c.end_s for c in cues), default=start)
    return Window(start, end, cues), pos


def _cached_window(cache, fp, key, compute) -> tuple[Window, float]:
    if cache is not None:
        hit = cache.get(fp, "window", key)
        if hit is not None:
            return _window_load(hit)
    w, read_to = compute()
    if cache is not None:
        cache.put(fp, "window", key, _window_json(w, read_to))
    return w, read_to


def extract_dialogue(
    path: Path,
    language: str,
    vocabulary: set[str],
    cfg: AnalysisConfig,
    cache: AnalysisCache | None = None,
    full: bool = False,
) -> Dialogue:
    path = Path(path)
    info = probe(path)
    fp = fingerprint(path)
    track = select_track(info.subtitle_tracks, language)
    if track is None:
        return Dialogue(track=None, full=full)

    max_s = cfg.window_max_minutes * 60.0
    dur = info.duration_s
    wins: list[tuple[Window, float]] = []
    for start, _ in window_spans(dur, max_s=max_s):
        key = params_hash(
            track=track.index,
            start_s=start,
            language=language,
            min_lines=cfg.window_min_lines,
            max_minutes=cfg.window_max_minutes,
        )
        wins.append(
            _cached_window(
                cache,
                fp,
                key,
                lambda s=start: _read_window(
                    path, track, s, min(dur, s + max_s), language, cfg, cfg.window_min_lines
                ),
            )
        )

    if full:
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
                key = params_hash(track=track.index, start_s=s, language=language, full=e)
                wins.append(
                    _cached_window(
                        cache,
                        fp,
                        key,
                        lambda s=s, e=e: _read_window(
                            path, track, s, e, language, cfg, 0, chunk=max_s
                        ),
                    )
                )
                s = e

    windows = sorted((w for w, _ in wins), key=lambda w: w.start_s)
    if full:  # cues straddling window edges would otherwise appear twice
        seen: set[tuple[float, float, str]] = set()
        for w in windows:
            kept = []
            for c in w.cues:
                k = (c.start_s, c.end_s, c.text)
                if k not in seen:
                    seen.add(k)
                    kept.append(c)
            w.cues = kept

    if track.kind == "text":
        quality: float | None = 1.0
    else:
        quality = ocr_quality([c.text for w in windows for c in w.cues], vocabulary, language)
    return Dialogue(track=track, windows=windows, ocr_quality=quality, full=full)
