"""Pure naming helpers (spec §13.1, §13.2, decisions 56-57). The M5 contract; implemented."""

from __future__ import annotations

import re
from pathlib import Path

_DUPE = re.compile(r"_dupe_(\d+)(?![0-9])", re.IGNORECASE)
_BAD = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def snake(text: str) -> str:
    """Title lowercased, each run of other characters replaced by `_` ('The West Wing' ->
    'the_west_wing')."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def sanitize(text: str) -> str:
    """Filesystem-safe: forbidden characters become `_`, whitespace collapses, no trailing
    dots or spaces."""
    t = re.sub(r"\s+", " ", _BAD.sub("_", text)).strip()
    return t.rstrip(". ")


def episode_stem(template: str, show_title: str, season: int, episode: int,
                 title: str | None = None) -> str:
    """Render the naming template (`{show_snake}`, `{show}`, `{title}`, `{s}`, `{e}`)."""
    return template.format(
        show_snake=snake(show_title), show=sanitize(show_title), s=season, e=episode,
        title=sanitize(title or ""),
    )


def dupe_stem(stem: str, index: int) -> str:
    """'the_west_wing_s01e02' -> 'the_west_wing_s01e02_dupe_1' (decision 56)."""
    return f"{stem}_dupe_{index}"


def parse_dupe(name: str) -> int | None:
    """X of a `_dupe_X` suffix anywhere in a file name, else None."""
    m = _DUPE.search(name)
    return int(m.group(1)) if m else None


def extra_name(disc_folder: str | None, original: str) -> str:
    """`disc2__<original>` for a file from a disc folder; unchanged without one (decision 57)."""
    return f"{sanitize(disc_folder)}__{original}" if disc_folder else original


def with_collision_index(path: Path, n: int) -> Path:
    """'a.mkv' -> 'a_2.mkv' (n >= 2): the numeric suffix that replaces aborting (decision 57)."""
    return path.with_name(f"{path.stem}_{n}{path.suffix}")


def sidecar_target(video_old: Path, sidecar_old: Path, video_new: Path) -> Path:
    """A sidecar keeps whatever follows its video's basename: title_t00.en.srt next to
    title_t00.mkv, moved with the video renamed to show_s01e02.mkv, becomes
    show_s01e02.en.srt in the video's new folder."""
    rest = sidecar_old.name[len(video_old.stem):]
    return video_new.with_name(video_new.stem + rest)
