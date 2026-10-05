"""Media file probing via PyAV (spec §8.1, stage 1 of §8.4)."""

from __future__ import annotations

import re
from pathlib import Path

import av

from showvenist.analysis.types import MediaProbe, SubtitleTrack

# PyAV reports decoder names; map to ffprobe's codec names.
CODEC_ALIASES = {"pgssub": "hdmv_pgs_subtitle", "dvdsub": "dvd_subtitle", "cc_dec": "eia_608"}
_SDH_TITLE = re.compile(r"\bsdh\b|hearing[ -]impaired|\bdeaf\b|\bcc\b|closed caption", re.I)


def _tag(md: dict | None, key: str) -> str | None:
    for k, v in (md or {}).items():
        # MakeMKV writes statistics tags with a language suffix: NUMBER_OF_FRAMES-eng
        if k.lower() == key.lower() or k.lower().startswith(key.lower() + "-"):
            return v
    return None


def _flag(disposition, name: str) -> bool:
    flag = getattr(av.stream.Disposition, name, None)
    return bool(disposition is not None and flag is not None and disposition & flag)


def _container_name(fmt: str) -> str:
    if "matroska" in fmt:
        return "matroska"
    if "mp4" in fmt:
        return "mp4"
    return fmt.split(",")[0]


def probe(path: Path) -> MediaProbe:
    path = Path(path)
    with av.open(str(path)) as c:
        duration = c.duration / av.time_base if c.duration else 0.0
        chapters = c.chapters
        n_chapters = len(chapters() if callable(chapters) else chapters or [])
        video = next((s for s in c.streams if s.type == "video"), None)
        subs = [s for s in c.streams if s.type == "subtitle"]
        counts: dict[int, int] = {}
        need = []
        for s in subs:
            nf = _tag(s.metadata, "NUMBER_OF_FRAMES")
            if nf and nf.strip().isdigit():
                counts[s.index] = int(nf)
            else:
                need.append(s)
        if need:
            for s in need:
                counts[s.index] = 0
            for pkt in c.demux(need):
                if pkt.size:
                    counts[pkt.stream.index] += 1
        tracks = []
        for s in subs:
            title = _tag(s.metadata, "title") or ""
            name = s.codec_context.name if s.codec_context else s.name
            tracks.append(
                SubtitleTrack(
                    index=s.index,
                    codec=CODEC_ALIASES.get(name, name),
                    lang=(_tag(s.metadata, "language") or None),
                    forced=_flag(s.disposition, "forced") or "forced" in title.lower(),
                    sdh=_flag(s.disposition, "hearing_impaired") or bool(_SDH_TITLE.search(title)),
                    events=counts[s.index],
                )
            )
        vctx = video.codec_context if video is not None else None
        return MediaProbe(
            duration_s=float(duration),
            size=path.stat().st_size,
            container=_container_name(c.format.name),
            title_tag=_tag(c.metadata, "title") or None,
            chapters=n_chapters,
            subtitle_tracks=tracks,
            video_codec=vctx.name if vctx else None,
            width=vctx.width if vctx else None,
            height=vctx.height if vctx else None,
        )
