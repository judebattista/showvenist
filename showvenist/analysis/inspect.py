"""Plain-text report on one media file (spec §8; the `inspect` command)."""

from __future__ import annotations

from pathlib import Path

from showvenist.analysis.probe import probe
from showvenist.analysis.types import AnalysisCache
from showvenist.analysis.windows import extract_dialogue
from showvenist.config import AnalysisConfig
from showvenist.state.fingerprint import content_key, fingerprint


def inspect_file(
    path: Path,
    language: str,
    vocabulary: set[str],
    cfg: AnalysisConfig,
    cache: AnalysisCache | None = None,
) -> str:
    path = Path(path)
    info = probe(path, cache)
    d = extract_dialogue(path, language, vocabulary, cfg, cache, probe_result=info)
    out = [
        f"file:         {path}",
        f"duration:     {info.duration_s:.1f} s ({info.duration_s / 60:.1f} min)",
        f"size:         {info.size} bytes",
        f"fingerprint:  {fingerprint(path)}",
        f"content key:  {content_key(path)}",
        f"title tag:    {info.title_tag or '-'}",
        f"subtitle tracks ({len(info.subtitle_tracks)}):",
    ]
    for t in info.subtitle_tracks:
        mark = "*" if d.track is not None and t.index == d.track.index else " "
        flags = " forced" if t.forced else ""
        flags += " sdh" if t.sdh else ""
        events = "?" if t.events < 0 else str(t.events)
        out.append(
            f" {mark} #{t.index} {t.codec} ({t.kind}) lang={t.lang or '-'} events={events}{flags}"
        )
    if d.track is None:
        out.append(f"no usable '{language}' subtitle track: dialogue scorers not available")
        return "\n".join(out) + "\n"
    if d.note:
        out.append(f"note:         {d.note}")
    out.append("windows:")
    for w in d.windows:
        err = f" (read failed: {w.error})" if w.error else ""
        out.append(f"  {w.start_s:8.1f} - {w.end_s:8.1f} s: {w.lines} lines{err}")
    q = d.ocr_quality
    out.append(f"ocr quality:  {'n/a' if q is None else f'{q:.2f}'}")
    out.append("first cues:")
    first = [c for w in d.windows for c in w.cues][:5]
    for c in first:
        out.append(f"  [{c.start_s:8.1f}] {c.text.strip().replace(chr(10), ' / ')}")
    return "\n".join(out) + "\n"
