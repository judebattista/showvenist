"""Pure naming helpers (spec §13.1, §13.2, decisions 56-57). The M5 contract; implemented."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

_DUPE = re.compile(r"_dupe_(\d+)(?![0-9])", re.IGNORECASE)
_BAD = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def snake(text: str) -> str:
    """Title lowercased, each run of other characters replaced by `_` ('The West Wing' ->
    'the_west_wing')."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


MAX_NAME_BYTES = 255
_TAIL = re.compile(r"(_dupe_\d+|_\d+)?$")


def limit_name(name: str, max_bytes: int = MAX_NAME_BYTES, *, has_ext: bool = True) -> str:
    """Cut a file name to `max_bytes` UTF-8 bytes. The extension and a trailing `_dupe_N` /
    `_N` suffix survive; the loss comes out of the stem before them, at a character boundary."""
    if len(name.encode()) <= max_bytes:
        return name
    p = Path(name)
    ext = p.suffix if has_ext and re.fullmatch(r"\.[A-Za-z0-9]{1,8}", p.suffix) else ""
    stem = name[: len(name) - len(ext)]
    tail = _TAIL.search(stem).group(0)
    head = stem[: len(stem) - len(tail)]
    room = max_bytes - len((tail + ext).encode())
    head = head.encode()[: max(room, 0)].decode(errors="ignore").rstrip(" ._")
    return head + tail + ext


def sanitize(text: str) -> str:
    """Filesystem-safe: forbidden characters become `_`, whitespace collapses, no trailing
    dots or spaces."""
    t = re.sub(r"\s+", " ", _BAD.sub("_", text)).strip()
    return t.rstrip(". ")


def episode_stem(template: str, show_title: str, season: int, episode: int,
                 title: str | None = None) -> str:
    """Render the naming template (`{show_snake}`, `{show}`, `{title}`, `{s}`, `{e}`)."""
    out = template.format(
        show_snake=snake(show_title), show=sanitize(show_title), s=season, e=episode,
        title=sanitize(title or ""),
    )
    out = re.sub(r"__+", "_", out).lstrip("_") if "{show_snake}" in template else out
    return limit_name(out, MAX_NAME_BYTES - 12, has_ext=False)  # room for `.ext`, `_dupe_N`, `_N`


def dupe_stem(stem: str, index: int) -> str:
    """'the_west_wing_s01e02' -> 'the_west_wing_s01e02_dupe_1' (decision 56)."""
    return limit_name(f"{stem}_dupe_{index}", has_ext=False)


def parse_dupe(name: str) -> int | None:
    """X of a `_dupe_X` suffix anywhere in a file name, else None."""
    m = _DUPE.search(name)
    return int(m.group(1)) if m else None


def extra_name(disc_folder: str | None, original: str) -> str:
    """`disc2__<original>` for a file from a disc folder; unchanged without one (decision 57)."""
    return limit_name(f"{sanitize(disc_folder)}__{original}" if disc_folder else original)


def with_collision_index(path: Path, n: int) -> Path:
    """'a.mkv' -> 'a_2.mkv' (n >= 2): the numeric suffix that replaces aborting (decision 57)."""
    return path.with_name(limit_name(f"{path.stem}_{n}{path.suffix}"))


def sidecar_target(video_old: Path, sidecar_old: Path, video_new: Path) -> Path:
    """A sidecar keeps whatever follows its video's basename: title_t00.en.srt next to
    title_t00.mkv, moved with the video renamed to show_s01e02.mkv, becomes
    show_s01e02.en.srt in the video's new folder."""
    rest = sidecar_old.name[len(video_old.stem):]
    return video_new.with_name(limit_name(video_new.stem + rest))
