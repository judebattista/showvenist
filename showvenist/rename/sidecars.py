"""Sidecar discovery (spec §13.2)."""

from __future__ import annotations

import os
from pathlib import Path

# extensions that are videos in their own right: never a sidecar of another video
VIDEO_EXTS = frozenset({
    ".mkv", ".mp4", ".m4v", ".avi", ".ts", ".m2ts", ".mts", ".mov", ".wmv", ".mpg", ".mpeg",
    ".webm", ".vob", ".iso",
})


def find_sidecars(video: Path) -> list[Path]:
    """Files in the video's folder whose name is the video's stem followed by '.' and more
    (title_t00.en.srt, title_t00.nfo), excluding the video itself and other videos. Sorted.
    A folder that cannot be listed has no sidecars (the plan's preflight reports the folder)."""
    video = Path(video)
    prefix = video.stem + "."
    try:
        names = os.listdir(video.parent)
    except OSError:
        return []
    # a longer video stem that also prefixes a name owns it (A.part2.en.srt is A.part2.mkv's)
    longer = [Path(n).stem + "." for n in names
              if Path(n).suffix.lower() in VIDEO_EXTS and len(Path(n).stem) > len(video.stem)
              and Path(n).stem.startswith(prefix)]
    out = []
    for name in names:
        if name == video.name or not name.startswith(prefix):
            continue
        if any(name.startswith(s) for s in longer):
            continue
        p = video.parent / name
        if p.suffix.lower() in VIDEO_EXTS or not p.is_file() or p.is_symlink():
            continue
        out.append(p)
    return sorted(out)
