"""M4 contract: the parsed Show / Season / Disc tree and its helpers (spec §5, §11.5).

Module map (one task each; implement exactly these names):

hierarchy/parse.py     (sv-o2p.2)
    parse_season(name: str, patterns: Sequence[str] | None = None) -> int | None
        'Season 1', 'Season 01', 'S01', 'Season_02' -> n; 'Specials' / 'Season 0' -> 0; else None.
    parse_disc(name: str, patterns: Sequence[str] | None = None) -> DiscRef | None
        'Disc 1', 'D1', 'DISC_1', MakeMKV labels 'FIREFLY_D1', 'FRIENDS_S2_D1', title tags like
        'THE WEST WING COMPLETE SERIES S3D2' or 'The Expanse: Season One (Disc 2)'. A label that
        names a disc but gives no number ('Disc A', identical labels) -> DiscRef(number=None).
    parse_named(name: str, patterns: Sequence[str] | None = None) -> NamedEpisode | None
        S01E02, show_s01e02, s01_e02, case-insensitive, letter suffix kept ('s02e09a').
    build_tree(root: Path, *, config: HierarchyConfig, tag_of: Callable[[Path], str | None],
               stat: Callable[[Path], os.stat_result] = os.stat) -> ShowNode
        Walks root (a show folder, a season folder, a disc folder, or a flat list of files), parses
        every level from folder names, falls back to `tag_of(file)` (the MKV title tag) for disc
        identity, folder wins and a disagreement is recorded in `disagreements`; never guesses:
        an unparseable level is recorded in `problems`.
    suggest_disc_order(discs: list[DiscNode]) -> list[DiscNode]
        For discs without usable numbers: order by folder creation time (st_ctime/birthtime), and
        mark `order_source = "creation-time"` (the caller must have the user confirm).

hierarchy/marker.py    (sv-o2p.3)
    read_marker(show_dir: Path) -> Marker | None
    write_marker(show_dir: Path, marker: Marker, overwrite: bool = False) -> Path   # atomic
    jellyfin_hint(folder_name: str) -> int | None        # 'Firefly (2002) [tmdbid-1437]' -> 1437
    resolve_show(name_or_dir, bundles_root, *, interactive, pick=None, tmdb_hint=None) -> ShowChoice
    named_episodes(files: Sequence[FileNode], bundle, ordering) -> NamedReport   # §5.5

hierarchy/readiness.py (sv-o2p.4)
    ReadinessChecker(stable_seconds: float, clock=time.time, sleep=time.sleep)
        .check(paths: Sequence[Path]) -> dict[Path, Readiness]
    matroska_complete(path: Path) -> bool      # Segment has a duration and Cues

analysis/pipeline.py   (sv-o2p.5)
    analyze_files(paths, *, language, vocabulary, cfg, cache, progress=None) -> list[FileResult]

assign/blocks.py       (sv-o2p.6)   solve_blocks(), block_confidence()   (types in assign/types.py)
assign/ordering.py     (sv-o2p.6)   detect_ordering()
session.py             (sv-o2p.6, after the above): directory input, blocks, scope, --episodes
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DiscRef:
    """What a folder name or a title tag says about a disc."""

    number: int | None  # None = names a disc but gives no number
    season: int | None = None  # FRIENDS_S2_D1 / S3D2 carry the season too
    label: str = ""  # the text that was parsed
    source: Literal["folder", "tag"] = "folder"


@dataclass(frozen=True)
class NamedEpisode:
    """A file name that parses as an episode (§5.5): ground truth, never analyzed or renamed."""

    season: int
    episode: int
    suffix: str = ""  # 's02e09a' -> 'a': names differing only by suffix form a duplicate set
    dupe: int | None = None  # X of a `_dupe_X` name (decision 56): a known duplicate copy


@dataclass
class FileNode:
    path: Path
    size: int
    mtime: float
    title_tag: str | None = None
    tag_disc: DiscRef | None = None  # the disc identity read from the title tag
    named: NamedEpisode | None = None
    ready: bool = True
    not_ready_reason: str | None = None
    # given on the command line as a file, not found by walking a directory (§13.2: renamed
    # where it is, never flattened)
    explicit: bool = False


@dataclass
class DiscNode:
    key: str  # stable id inside the tree (folder name, or 'tag:<label>' for flattened files)
    ref: DiscRef | None
    folder: Path | None
    files: list[FileNode] = field(default_factory=list)
    incomplete: bool = False  # some files are not ready: the block's upper boundary stays open
    order_source: Literal["number", "creation-time", "name"] = "number"
    created: float | None = None  # folder creation time, for suggest_disc_order

    @property
    def number(self) -> int | None:
        return self.ref.number if self.ref else None

    @property
    def ready_files(self) -> list[FileNode]:
        return [f for f in self.files if f.ready]


@dataclass
class SeasonNode:
    number: int | None  # None = no season folder / not parseable
    folder: Path | None
    discs: list[DiscNode] = field(default_factory=list)
    loose: list[FileNode] = field(default_factory=list)  # files with no disc identity


@dataclass
class Marker:
    """<Show>/.showvenist.toml (§5.4); written only after the user confirms the show."""

    show: str
    tmdb_id: int | None = None
    bundle: str | None = None
    language: str = "en"
    naming_order: str = "aired"
    include_specials: bool = False


@dataclass
class ShowNode:
    root: Path
    name: str | None = None
    marker: Marker | None = None
    seasons: list[SeasonNode] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # levels that could not be parsed
    disagreements: list[str] = field(default_factory=list)  # folder vs title tag
    ignored: list[Path] = field(default_factory=list)  # folders skipped via .showvenist-ignore

    def all_files(self) -> list[FileNode]:
        out: list[FileNode] = []
        for s in self.seasons:
            for d in s.discs:
                out += d.files
            out += s.loose
        return out


@dataclass
class Readiness:
    ready: bool
    reason: str | None = None  # 'still growing', 'mtime changed', 'no cues', 'no duration', ...


def default_stat(path: Path) -> os.stat_result:
    return os.stat(path)
