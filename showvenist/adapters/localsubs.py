"""Local subtitle folder adapter: maps *.srt files to episodes by SxxEyy in the filename."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from showvenist.bundle.schema import Bundle, Episode, PartialBundle, SubtitleRef, to_639_1

_SXXEYY = re.compile(r"(?<![0-9a-z])s(\d{1,3})[ ._-]?e(\d{1,4})(?![0-9])", re.I)
_NXM = re.compile(r"(?<![0-9a-z])(\d{1,2})x(\d{1,3})(?![0-9a-z])", re.I)
_LANG_SUFFIX = re.compile(r"\.([a-z]{2,3})$", re.I)


@dataclass
class ScanReport:
    unmatched: list[Path] = field(default_factory=list)  # no SxxEyy, or not in the ordering
    duplicates: dict[str, list[Path]] = field(default_factory=dict)  # episode id -> files
    missing: list[str] = field(default_factory=list)  # episode ids lacking a file
    matched: dict[str, Path] = field(default_factory=dict)  # episode id -> chosen file


def parse_season_episode(name: str) -> tuple[int, int] | None:
    m = _SXXEYY.search(name) or _NXM.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def scan_folder(
    folder: Path, bundle: Bundle, ordering: str = "aired", lang: str | None = None
) -> tuple[PartialBundle, ScanReport]:
    """Map *.srt files in folder to episodes. Returns (partial, report).

    Each Episode in the partial carries only its id and a SubtitleRef whose path is
    the file's *source* path until `install` copies it into the bundle.
    """
    entries = bundle.orderings[ordering].entries
    by_se = {(e.season, e.episode): e.episode_id for e in entries}
    default_lang = lang or bundle.show.language
    report = ScanReport()
    found: dict[str, list[tuple[Path, str]]] = {}
    for f in sorted(Path(folder).rglob("*.srt")):
        se = parse_season_episode(f.name)
        eid = by_se.get(se) if se else None
        if eid is None:
            report.unmatched.append(f)
            continue
        stem = f.name[: -len(".srt")]
        m = _LANG_SUFFIX.search(stem)
        file_lang = to_639_1(m.group(1)) if m else None
        found.setdefault(eid, []).append((f, file_lang or default_lang))
    episodes: list[Episode] = []
    for eid, files in found.items():
        if len(files) > 1:
            report.duplicates[eid] = [p for p, _ in files]
        path, flang = files[0]
        report.matched[eid] = path
        episodes.append(
            Episode(
                id=eid,
                title="",
                subtitles=[SubtitleRef(lang=flang, path=str(path), source="localsubs")],
            )
        )
    report.missing = [e.episode_id for e in entries if e.episode_id not in found]
    return PartialBundle(episodes=episodes), report


def install(partial: PartialBundle, folder: Path, bundle_dir: Path) -> PartialBundle:
    """Copy each referenced srt to <bundle_dir>/subs/<id-with-dashes>.<lang>.srt.

    Returns a new PartialBundle whose SubtitleRef paths are relative to bundle_dir.
    """
    subs = Path(bundle_dir) / "subs"
    subs.mkdir(parents=True, exist_ok=True)
    out: list[Episode] = []
    for ep in partial.episodes:
        refs: list[SubtitleRef] = []
        for ref in ep.subtitles:
            src = Path(ref.path)
            if not src.is_absolute():
                src = Path(folder) / src
            rel = f"subs/{ep.id.replace(':', '-')}.{ref.lang}.srt"
            shutil.copyfile(src, Path(bundle_dir) / rel)
            refs.append(ref.model_copy(update={"path": rel}))
        out.append(ep.model_copy(update={"subtitles": refs}))
    return partial.model_copy(update={"episodes": out})
