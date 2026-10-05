"""Text subtitle extraction and normalization (spec §8.3)."""

from __future__ import annotations

import re
from pathlib import Path

import av

from showvenist.analysis.types import Cue, SubtitleTrack

DEFAULT_CUE_S = 2.0

_ASS_OVERRIDE = re.compile(r"\{[^}]*\}")


def _clean_ass(text: str) -> str:
    text = _ASS_OVERRIDE.sub("", text)
    return text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ").strip()


def _packet_text(codec: str, data: bytes) -> str:
    """Cue text from a raw Matroska/MP4 subtitle packet."""
    if codec in ("ass", "ssa"):
        # ReadOrder,Layer,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        parts = data.decode("utf-8", "replace").split(",", 8)
        return _clean_ass(parts[8]) if len(parts) == 9 else ""
    if codec == "mov_text":
        n = int.from_bytes(data[:2], "big")
        text = data[2 : 2 + n].decode("utf-8", "replace")
    elif codec == "webvtt":
        # Matroska WebVTT blocks: settings line, identifier line, then payload.
        lines = data.decode("utf-8", "replace").split("\n")
        text = "\n".join(lines[2:]) if len(lines) > 2 else "\n".join(lines)
    else:
        text = data.decode("utf-8", "replace")
    return text.replace("\r\n", "\n").strip()


def read_text_cues(path: Path, track: SubtitleTrack, start_s: float, end_s: float) -> list[Cue]:
    """Cues of a text track overlapping [start_s, end_s], in time order. Seeks, then demuxes
    only the subtitle stream; no OCR."""
    cues: list[Cue] = []
    with av.open(str(path)) as c:
        stream = c.streams[track.index]
        if start_s > 0:
            c.seek(int(start_s * av.time_base), backward=True, any_frame=False)
        for pkt in c.demux(stream):
            if pkt.pts is None or not pkt.size:
                continue
            t0 = float(pkt.pts * pkt.time_base)
            if t0 > end_s:
                break
            dur = float(pkt.duration * pkt.time_base) if pkt.duration else DEFAULT_CUE_S
            if t0 + dur < start_s:
                continue
            text = _packet_text(track.codec, bytes(pkt))
            if text:
                cues.append(Cue(t0, t0 + dur, text))
    cues.sort(key=lambda q: q.start_s)
    return cues


_MARKUP = re.compile(r"<[^>]*>|\{[^}]*\}")
_SDH = re.compile(r"\[[^\]]*\]?|\([^)]*\)?")
# Speaker labels: "JOHN:" in mixed-case tracks, "Caldwell:" in all-caps (CC-style) tracks.
_SPEAKER = re.compile(
    r"^\s*-?\s*(?:[A-Z][A-Z0-9 .'#-]{1,30}|[A-Z][a-z]+(?: [A-Z][a-z]+)?):(?!\d)\s*", re.M
)
_PUNCT = re.compile(r"[^\w\s']|_")
_APOS = re.compile(r"(?<!\w)'|'(?!\w)")


def normalize(text: str) -> str:
    """§8.3: strip markup, SDH annotations and speaker labels; lowercase; drop punctuation and
    apostrophes ("don't" -> "dont"); collapse whitespace."""
    t = _MARKUP.sub(" ", text)
    t = _SDH.sub(" ", t)
    t = _SPEAKER.sub(" ", t)
    t = t.replace("’", "'").lower()
    t = _PUNCT.sub(" ", t)
    t = _APOS.sub(" ", t)
    t = t.replace("'", "")
    return " ".join(t.split())
