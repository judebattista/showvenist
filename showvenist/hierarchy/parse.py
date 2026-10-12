"""Parse show / season / disc structure from folder names and MKV title tags (spec §5.1, §5.5).

Patterns are regexes matched case-insensitively with ``search``. A custom pattern may use the named
groups ``season`` / ``disc`` (/ ``episode`` / ``suffix``); without names, groups are positional:
season: (n); disc: (disc) or (season, disc); named: (season, episode[, suffix]).
A disc pattern that matches without a ``disc`` group names a disc with no number ('Disc A').
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from showvenist.config import HierarchyConfig
from showvenist.hierarchy.types import (
    DiscNode,
    DiscRef,
    FileNode,
    NamedEpisode,
    SeasonNode,
    ShowNode,
)
from showvenist.rename.naming import parse_dupe

_CARDINALS = "one two three four five six seven eight nine ten eleven twelve".split()
_ORDINALS = (
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split()
)
_WORDS = {w: i + 1 for i, w in enumerate(_CARDINALS)} | {w: i + 1 for i, w in enumerate(_ORDINALS)}
_WORD_RE = "|".join(sorted(_WORDS, key=len, reverse=True))
_B = r"(?<![a-z0-9])"  # underscore counts as a separator, unlike \b
_E = r"(?![a-z0-9])"
_SEP = r"[\s_.-]*"

DEFAULT_SEASON_PATTERNS = (
    rf"{_B}season{_SEP}(?P<season>\d{{1,2}}|{_WORD_RE}){_E}",
    rf"{_B}(?P<season>{_WORD_RE})[\s_.-]+season{_E}",
    r"^\s*s(?P<season>\d{1,2})\s*$",
    r"^\s*(?P<season>specials?)\s*$",
)
DEFAULT_DISC_PATTERNS = (
    rf"{_B}s(?P<season>\d{{1,2}}){_SEP}d(?:is[ck])?{_SEP}(?P<disc>\d{{1,2}}){_E}",  # S3D2, S2_D1
    rf"{_B}dis[ck]{_SEP}(?P<disc>\d{{1,2}}|{_WORD_RE}){_E}",  # Disc 1, DISC_1, DISC2, (Disc 2)
    rf"{_B}d(?P<disc>\d{{1,2}}){_E}",  # D1, FIREFLY_D1
    rf"{_B}dis[ck]{_SEP}\d+[a-z]{_E}",  # disc1a: names a disc, number is ambiguous
    rf"{_B}dis[ck]{_SEP}[a-z]{_E}",  # Disc A
)
DEFAULT_NAMED_PATTERNS = (
    rf"{_B}s(?P<season>\d{{1,2}}){_SEP}e(?P<episode>\d{{1,3}})(?P<suffix>[a-z])?{_E}",
)

_VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".avi", ".ts", ".m2ts", ".mov"}


def _compile(patterns: Sequence[str] | None, defaults: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in (patterns or defaults)]


def _num(text: str | None) -> int | None:
    if text is None:
        return None
    t = text.strip().lower()
    if t.isdigit():
        return int(t)
    if t.startswith("special"):
        return 0
    return _WORDS.get(t)


def _group(m: re.Match[str], name: str, pos: int, npos: int) -> str | None:
    """Named group if the pattern has it, else a positional one (pos of npos groups)."""
    if name in m.re.groupindex:
        return m.group(name)
    if m.re.groups == npos and pos <= npos:
        return m.group(pos)
    return None


def parse_season(name: str, patterns: Sequence[str] | None = None) -> int | None:
    for rx in _compile(patterns, DEFAULT_SEASON_PATTERNS):
        m = rx.search(name)
        if not m:
            continue
        n = _num(m.group("season") if "season" in rx.groupindex else _first(m))
        if n is not None:
            return n
    return None


def _first(m: re.Match[str]) -> str | None:
    return next((g for g in m.groups() if g is not None), None)


def parse_disc(name: str, patterns: Sequence[str] | None = None) -> DiscRef | None:
    for rx in _compile(patterns, DEFAULT_DISC_PATTERNS):
        m = rx.search(name)
        if not m:
            continue
        gi = rx.groupindex
        if "disc" in gi or "season" in gi:
            disc, season = m.groupdict().get("disc"), m.groupdict().get("season")
        elif rx.groups == 1:
            disc, season = m.group(1), None
        elif rx.groups >= 2:
            season, disc = m.group(1), m.group(2)
        else:
            disc = season = None
        s = _num(season)
        if s is None:
            s = parse_season(name)
        return DiscRef(_num(disc), s, m.group(0).strip(), "folder")
    return None


def parse_named(name: str, patterns: Sequence[str] | None = None) -> NamedEpisode | None:
    for rx in _compile(patterns, DEFAULT_NAMED_PATTERNS):
        m = rx.search(name)
        if not m:
            continue
        season = _num(_group(m, "season", 1, rx.groups))
        episode = _num(_group(m, "episode", 2, rx.groups))
        if season is None or episode is None:
            continue
        suffix = _group(m, "suffix", 3, rx.groups) or ""
        return NamedEpisode(season, episode, suffix.lower(), parse_dupe(name))
    return None


_TITLE_RE = re.compile(r"_t(\d{2,3})$", re.IGNORECASE)


def parse_title_index(name: str) -> int | None:
    """MakeMKV's title index from a file name (`B1_t03.mkv` -> 3, `title_t00.mkv` -> 0); None
    when the stem doesn't end in `_tNN` (spec §11.5, decision 63)."""
    m = _TITLE_RE.search(Path(name).stem)
    return int(m.group(1)) if m else None


def parse_tag_season(tag: str, patterns: Sequence[str] | None = None) -> int | None:
    """Season named by a title tag, with or without a disc ('Babylon 5 - Season 2', 'S3D2')."""
    s = parse_season(tag, patterns)
    if s is not None:
        return s
    ref = parse_disc(tag)
    return ref.season if ref else None


DUPLICATE_DISC_PROBLEM = "disc number {number} appears twice in {where!r} ({keys})"
_SIDE_RE = re.compile(rf"{_B}dis[ck]{_SEP}(\d{{1,2}})([a-z]){_E}", re.IGNORECASE)


def disc_side(ref: DiscRef | None) -> tuple[int, str] | None:
    """(number, side) of a side-lettered label such as 'disc1a' / 'Disc 2B' (double-sided
    DVDs): DiscRef has no field for the side, so such a disc keeps number None and this is its
    secondary key. None for any other label."""
    if ref is None or ref.number is not None:
        return None
    m = _SIDE_RE.search(ref.label)
    return (int(m.group(1)), m.group(2).lower()) if m else None


def disc_sort_key(d: DiscNode) -> tuple:
    """Numbered discs by number, then side-lettered ones by (number, side), then the rest by
    natural name."""
    side = disc_side(d.ref)
    return (d.number is None, d.number or 0, side is None, side or (0, ""), _natural(d.key))


def sides_order(discs: Sequence[DiscNode]) -> list[DiscNode] | None:
    """When every disc of a season is side-lettered with distinct (number, side) keys, their
    order by that key (no guess: 1a < 1b < 2a); else None (the creation-time path, §5.1)."""
    keys = [disc_side(d.ref) for d in discs]
    if not discs or any(k is None for k in keys) or len(set(keys)) != len(keys):
        return None
    out = sorted(discs, key=disc_sort_key)
    for d in out:
        d.order_source = "name"
    return out


def mark_duplicate_discs(node: SeasonNode, where: str) -> list[str]:
    """Discs of one season that parse to the same number (two 'Disc 1' folders, identical
    volume labels): the number can't order them, so each loses it (number None, order_source
    'name') and the caller takes the unnumbered-disc path (suggest_disc_order by creation time,
    confirmed by the user, §5.1: never guessed). Returns one problem line per repeated number,
    formatted with DUPLICATE_DISC_PROBLEM."""
    by_num: dict[int, list[DiscNode]] = {}
    for d in node.discs:
        if d.number is not None:
            by_num.setdefault(d.number, []).append(d)
    out: list[str] = []
    for num, ds in by_num.items():
        if len(ds) < 2:
            continue
        out.append(DUPLICATE_DISC_PROBLEM.format(
            number=num, where=where, keys=", ".join(repr(d.key) for d in ds)))
        for d in ds:
            assert d.ref is not None
            d.ref = replace(d.ref, number=None)
            d.order_source = "name"
    return out


def is_duplicate_disc_problem(problem: str) -> bool:
    """A problem line from mark_duplicate_discs (handled like unnumbered discs, not as an
    unparseable level)."""
    return problem.startswith("disc number ") and " appears twice in " in problem


def _natural(s: str) -> list[tuple[int, int | str]]:
    return [(0, int(t)) if t.isdigit() else (1, t.lower()) for t in re.split(r"(\d+)", s) if t]


def _created(st: os.stat_result) -> float:
    return getattr(st, "st_birthtime", None) or st.st_ctime


def suggest_disc_order(discs: list[DiscNode]) -> list[DiscNode]:
    """Order by folder creation time and mark order_source (the caller must get it confirmed)."""
    ordered = sorted(
        discs,
        key=lambda d: (d.created is None, d.created or 0.0, _natural(d.key)),
    )
    for d in ordered:
        d.order_source = "creation-time"
    return ordered


IGNORE_FILE = ".showvenist-ignore"


def is_ignored(folder: Path) -> bool:
    """A folder holding a `.showvenist-ignore` file is out of scope: never analyzed, matched,
    renamed or moved, and never reported as a parse problem (movies, miniseries, bonus discs)."""
    return (folder / IGNORE_FILE).exists()


def walk_media(root: Path, patterns: tuple[str, ...] = ("*.mkv",)) -> list[Path]:
    """Files under `root` matching `patterns`, skipping ignored folders (and an ignored root)."""
    if root.is_file():
        return [root]
    out: list[Path] = []

    def visit(folder: Path) -> None:
        if is_ignored(folder):
            return
        for p in sorted(folder.iterdir()):
            if p.is_dir():
                visit(p)
            elif any(p.match(pat) for pat in patterns):
                out.append(p)

    visit(root)
    return out


class _Builder:
    def __init__(
        self,
        config: HierarchyConfig,
        tag_of: Callable[[Path], str | None],
        stat: Callable[[Path], os.stat_result],
    ) -> None:
        self.cfg = config
        self.tag_of = tag_of
        self.stat = stat
        self.problems: list[str] = []
        self.ignored: list[Path] = []
        self.disagreements: list[str] = []
        self._seen: set[tuple[str, str]] = set()

    def season(self, name: str) -> int | None:
        return parse_season(name, self.cfg.season_patterns)

    def disc(self, name: str) -> DiscRef | None:
        return parse_disc(name, self.cfg.disc_patterns)

    def disagree(self, key: tuple[str, str], msg: str) -> None:
        if key not in self._seen:
            self._seen.add(key)
            self.disagreements.append(msg)

    def file(self, path: Path) -> FileNode:
        st = self.stat(path)
        tag = self.tag_of(path)
        node = FileNode(path, st.st_size, st.st_mtime, title_tag=tag)
        if tag:
            ref = self.disc(tag)
            if ref:
                node.tag_disc = replace(ref, source="tag")
        node.named = parse_named(path.name, self.cfg.named_patterns)
        return node

    def files_in(self, folder: Path) -> list[FileNode]:
        paths = sorted(
            (
                p
                for p in folder.iterdir()
                if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in _VIDEO_EXTS
            ),
            key=lambda p: _natural(p.name),
        )
        return [self.file(p) for p in paths]

    def subdirs(self, folder: Path) -> list[Path]:
        """Sub-folders in natural order, leaving out (and recording) ignored ones."""
        out = []
        dirs = (p for p in folder.iterdir() if p.is_dir())
        for p in sorted(dirs, key=lambda p: _natural(p.name)):
            if is_ignored(p):
                self.ignored.append(p)
            else:
                out.append(p)
        return out

    def all_files_under(self, folder: Path) -> list[FileNode]:
        out = self.files_in(folder)
        for sub in self.subdirs(folder):
            out += self.all_files_under(sub)
        return out

    def disc_folder(self, folder: Path, ref: DiscRef, season: int | None) -> DiscNode:
        files = self.files_in(folder)
        node = DiscNode(
            folder.name,
            ref,
            folder,
            files,
            order_source="number" if ref.number is not None else "name",
            created=_created(self.stat(folder)),
        )
        for f in files:
            t = f.tag_disc
            if t and t.number is not None and ref.number is not None and t.number != ref.number:
                self.disagree(
                    (str(folder), t.label),
                    f"{folder.name}: folder says disc {ref.number}, title tag says "
                    f"disc {t.number} ({f.title_tag!r})",
                )
            if f.title_tag and season is not None:
                ts = parse_tag_season(f.title_tag, self.cfg.season_patterns)
                if ts is not None and ts != season:
                    self.disagree(
                        (str(folder), f"s{ts}"),
                        f"{folder.name}: season {season} by folder, season {ts} by title tag "
                        f"({f.title_tag!r})",
                    )
        return node

    def place_files(self, files: list[FileNode], season: SeasonNode, folder: Path | None) -> None:
        """Files with no disc folder: by tag disc ('tag:<label>'), else loose."""
        by_key: dict[tuple, DiscNode] = {}
        for f in files:
            t = f.tag_disc
            if t is None:
                season.loose.append(f)
                continue
            label = " ".join(t.label.split())
            k = (t.number, t.season, label.casefold())
            node = by_key.get(k)
            if node is None:
                node = by_key[k] = DiscNode(
                    f"tag:{label}",
                    t,
                    None,
                    order_source="number" if t.number is not None else "name",
                )
                season.discs.append(node)
            node.files.append(f)
        if season.number is not None and folder is not None:
            for f in files:
                ts = parse_tag_season(f.title_tag or "", self.cfg.season_patterns)
                if ts is not None and ts != season.number:
                    self.disagree(
                        (str(folder), f"s{ts}"),
                        f"{folder.name}: season {season.number} by folder, season {ts} by "
                        f"title tag ({f.title_tag!r})",
                    )

    def season_node(self, folder: Path, number: int | None) -> SeasonNode:
        node = SeasonNode(number, folder)
        for sub in self.subdirs(folder):
            ref = self.disc(sub.name)
            if ref is None:
                self.problems.append(f"cannot parse disc folder {sub.name!r} in {folder.name!r}")
                node.loose += self.all_files_under(sub)
            else:
                node.discs.append(self.disc_folder(sub, ref, number))
        self.place_files(self.files_in(folder), node, folder)
        self.check_duplicates(node, folder.name)
        return node

    def check_duplicates(self, node: SeasonNode, where: str) -> None:
        self.problems += mark_duplicate_discs(node, where)

    def sort(self, seasons: list[SeasonNode]) -> None:
        seasons.sort(
            key=lambda s: (
                s.number is None,
                s.number or 0,
                _natural(s.folder.name) if s.folder else [],
            )
        )
        for s in seasons:
            s.discs.sort(key=disc_sort_key)


def build_tree(
    root: Path,
    *,
    config: HierarchyConfig,
    tag_of: Callable[[Path], str | None],
    stat: Callable[[Path], os.stat_result] = os.stat,
) -> ShowNode:
    """Walk `root` (show folder, season folder, disc folder, or flat list of files)."""
    b = _Builder(config, tag_of, stat)
    show = ShowNode(root=root)
    if root.is_dir() and is_ignored(root):
        show.name = root.name
        show.ignored = [root]
        return show
    seasons: list[SeasonNode] = []
    root_disc = b.disc(root.name)
    root_season = b.season(root.name)

    if root_disc is not None:
        # a disc folder (so 'Season 1 Disc 2' is a disc)
        node = SeasonNode(root_disc.season, None)
        d = b.disc_folder(root, root_disc, node.number)
        node.discs.append(d)
        seasons.append(node)
    elif root_season is not None:
        seasons.append(b.season_node(root, root_season))
    else:
        show.name = root.name
        for sub in b.subdirs(root):
            ref = b.disc(sub.name)
            n = b.season(sub.name) if ref is None else None
            if n is not None:
                seasons.append(b.season_node(sub, n))
            elif ref is not None:
                s = next((x for x in seasons if x.number is None and x.folder is None), None)
                if s is None:
                    s = SeasonNode(None, None)
                    seasons.append(s)
                s.discs.append(b.disc_folder(sub, ref, None))
            else:
                b.problems.append(
                    f"cannot parse folder {sub.name!r} in {root.name!r} as season or disc"
                )
                seasons.append(SeasonNode(None, sub, loose=b.all_files_under(sub)))
        flat = b.files_in(root)
        if flat:
            # flattened files: the season comes from the title tag
            groups: dict[int | None, list[FileNode]] = {}
            for f in flat:
                s = parse_tag_season(f.title_tag or "", config.season_patterns)
                groups.setdefault(s, []).append(f)
            for num, fs in groups.items():
                target = next((x for x in seasons if x.number == num and x.folder is None), None)
                if target is None:
                    target = SeasonNode(num, None)
                    seasons.append(target)
                b.place_files(fs, target, None)
        for s in seasons:
            if s.folder is None:
                b.check_duplicates(s, root.name)
    b.sort(seasons)
    show.seasons = seasons
    show.problems = b.problems
    show.ignored = b.ignored
    show.disagreements = b.disagreements
    return show
