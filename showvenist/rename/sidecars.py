"""Sidecar discovery (spec §13.2). Contract; the body is task sv-o4f.3."""

from __future__ import annotations

from pathlib import Path


def find_sidecars(video: Path) -> list[Path]:
    """Files in the video's folder whose name is the video's stem followed by '.' and more
    (title_t00.en.srt, title_t00.nfo), excluding the video itself and other videos. Sorted."""
    raise NotImplementedError("sv-o4f.3")
