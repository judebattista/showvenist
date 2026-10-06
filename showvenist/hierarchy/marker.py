"""Marker file (§5.4), show resolution and already-named files (§5.5)."""

from __future__ import annotations

import os
import re
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from showvenist.bundle.schema import Bundle, Ordering
from showvenist.bundle.store import list_bundles, load_bundle, slugify
from showvenist.hierarchy.types import FileNode, Marker

MARKER_NAME = ".showvenist.toml"

_TOP_KEYS = {"show", "tmdb_id", "bundle", "language", "naming_order", "defaults"}
_JELLYFIN_RE = re.compile(r"\[tmdbid[-=](\d+)\]", re.IGNORECASE)


class ShowResolutionError(Exception):
    """The show could not be resolved without guessing."""


@dataclass
class ShowChoice:
    slug: str
    bundle_dir: Path
    source: Literal["marker", "tmdbid-hint", "name", "pick"]
    candidates: list[str] = field(default_factory=list)


@dataclass
class NamedReport:
    resolved: dict[Path, str] = field(default_factory=dict)  # path -> episode id
    unresolved: dict[Path, str] = field(default_factory=dict)  # path -> reason
    duplicate_sets: list[list[Path]] = field(default_factory=list)


# --- marker file -----------------------------------------------------------------------------


def read_marker(show_dir: Path) -> Marker | None:
    path = Path(show_dir) / MARKER_NAME
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{path}: invalid TOML: {e}") from e
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {', '.join(sorted(unknown))}")

    def typed(key: str, types: type | tuple[type, ...], what: str, default):
        v = data.get(key, default)
        if isinstance(v, bool) and bool not in (types if isinstance(types, tuple) else (types,)):
            v = object()
        if not isinstance(v, types):
            raise ValueError(f"{path}: '{key}' must be {what}")
        return v

    if "show" not in data:
        raise ValueError(f"{path}: missing required key 'show'")
    show = typed("show", str, "a string", None)
    tmdb_id = typed("tmdb_id", (int, type(None)), "an integer", None)
    bundle = typed("bundle", (str, type(None)), "a string", None)
    language = typed("language", str, "a string", "en")
    naming_order = typed("naming_order", str, "a string", "aired")
    defaults = typed("defaults", dict, "a table", {})
    bad = set(defaults) - {"include_specials"}
    if bad:
        raise ValueError(f"{path}: unknown key(s) in [defaults]: {', '.join(sorted(bad))}")
    inc = defaults.get("include_specials", False)
    if not isinstance(inc, bool):
        raise ValueError(f"{path}: 'defaults.include_specials' must be true or false")
    return Marker(show=show, tmdb_id=tmdb_id, bundle=bundle, language=language,
                  naming_order=naming_order, include_specials=inc)


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_marker(show_dir: Path, marker: Marker, overwrite: bool = False) -> Path:
    """Atomic write. Call only after the user confirmed the show."""
    path = Path(show_dir) / MARKER_NAME
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it)")
    lines = [f"show = {_q(marker.show)}"]
    if marker.tmdb_id is not None:
        lines.append(f"tmdb_id = {marker.tmdb_id}")
    if marker.bundle is not None:
        lines.append(f"bundle = {_q(marker.bundle)}")
    lines.append(f"language = {_q(marker.language)}")
    lines.append(f"naming_order = {_q(marker.naming_order)}")
    if marker.include_specials:
        lines += ["[defaults]", "include_specials = true"]
    else:
        lines += ["# [defaults]", "# include_specials = false"]
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".showvenist-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# --- Jellyfin naming -------------------------------------------------------------------------


def jellyfin_hint(folder_name: str) -> int | None:
    m = _JELLYFIN_RE.search(folder_name)
    return int(m.group(1)) if m else None


def suggest_folder_name(title: str, year: int | None, tmdb_id: int) -> str:
    base = f"{title} ({year})" if year else title
    return f"{base} [tmdbid-{tmdb_id}]"


# --- show resolution -------------------------------------------------------------------------


def _compact(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _strip_decor(name: str) -> str:
    name = _JELLYFIN_RE.sub("", name)
    return re.sub(r"\s*\((\d{4})\)\s*$", "", name.strip()).strip()


def resolve_show(
    name_or_dir: str | Path,
    bundles_root: Path,
    *,
    interactive: bool,
    pick: Callable[[list[str]], str] | None = None,
    tmdb_hint: int | None = None,
) -> ShowChoice:
    root = Path(bundles_root)
    p = Path(name_or_dir)
    is_dir = p.is_dir()
    folder_name = p.name if (is_dir or "/" in str(name_or_dir)) else str(name_or_dir)
    slugs = list_bundles(root)

    def choice(slug: str, source: str, cands: list[str] | None = None) -> ShowChoice:
        return ShowChoice(slug, root / slug, source, cands or [slug])  # type: ignore[arg-type]

    if is_dir:
        m = read_marker(p)
        if m is not None:
            slug = m.bundle or slugify(m.show)
            if slug not in slugs and m.bundle is None:
                by_title = _match_name(root, slugs, m.show)
                if len(by_title) == 1:
                    slug = by_title[0]
            if slug not in slugs:
                raise ShowResolutionError(
                    f"{p / MARKER_NAME} names bundle {slug!r}, which is not in {root}; "
                    "run `showvenist ingest` or fix the marker")
            return choice(slug, "marker")

    tid = tmdb_hint if tmdb_hint is not None else jellyfin_hint(folder_name)
    if tid is not None:
        hits = []
        for s in slugs:
            try:
                if load_bundle(s, root, apply_overrides=False).show.ids.tmdb == tid:
                    hits.append(s)
            except Exception:  # noqa: BLE001 - a broken bundle must not hide the others
                continue
        if len(hits) == 1:
            return choice(hits[0], "tmdbid-hint")
        if len(hits) > 1:
            return _several(hits, interactive, pick, root)
        if tmdb_hint is not None:
            raise ShowResolutionError(
                f"no bundle has TMDB id {tid}; run `showvenist ingest` for it")

    hits = _match_name(root, slugs, folder_name)
    if len(hits) == 1:
        return choice(hits[0], "name")
    if hits:
        return _several(hits, interactive, pick, root)
    raise ShowResolutionError(
        f"no bundle matches {folder_name!r}; run `showvenist ingest` to build one "
        "(or use --show/--tmdb-id)")


def _match_name(root: Path, slugs: list[str], name: str) -> list[str]:
    want = _compact(_strip_decor(name))
    if not want:
        return []
    exact, loose = [], []
    for s in slugs:
        titles = []
        try:
            titles = [load_bundle(s, root, apply_overrides=False).show.title]
        except Exception:  # noqa: BLE001
            pass
        forms = {_compact(s), *(_compact(t) for t in titles)}
        if want in forms or _compact(name) in forms:
            exact.append(s)
        elif want == _compact(re.sub(r"-\d{4}$", "", s)):
            loose.append(s)  # 'the-west-wing-1999' without its year
    return exact or loose


def _several(hits: list[str], interactive: bool, pick, root: Path) -> ShowChoice:
    hits = sorted(hits)
    if interactive and pick is not None:
        slug = pick(hits)
        if slug not in hits:
            raise ShowResolutionError(f"picked {slug!r}, which is not one of {hits}")
        return ShowChoice(slug, root / slug, "pick", hits)
    raise ShowResolutionError(
        f"several bundles match: {', '.join(hits)}; use --show or --tmdb-id to choose")


# --- already-named files (§5.5) --------------------------------------------------------------


def _find_ordering(bundle: Bundle, ordering: Ordering | str) -> Ordering | None:
    if isinstance(ordering, Ordering):
        return ordering
    low = ordering.lower()
    for k, o in bundle.orderings.items():
        if k == ordering or k.lower() == low:
            return o
    typed = [o for o in bundle.orderings.values() if o.type.lower() == low]
    if len(typed) > 1:  # Jellyfin keyword: the first group of that type (type_rank 1)
        ranked = [o for o in typed if o.type_rank == 1]
        return ranked[0] if len(ranked) == 1 else None
    return typed[0] if typed else None


def named_episodes(files: Sequence[FileNode], bundle: Bundle,
                   ordering: Ordering | str) -> NamedReport:
    report = NamedReport()
    named = [f for f in files if f.named is not None]
    if not named:
        return report
    ordr = _find_ordering(bundle, ordering)
    if ordr is None:
        for f in named:
            report.unresolved[f.path] = f"unknown ordering {ordering!r}"
        return report
    index: dict[tuple[int, int], set[str]] = defaultdict(set)
    per_season: dict[int, int] = defaultdict(int)
    for e in ordr.entries:
        index[(e.season, e.episode)].add(e.episode_id)
        per_season[e.season] += 1
    known = bundle.episode_by_id()
    groups: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for f in named:
        n = f.named
        assert n is not None
        label = f"S{n.season:02d}E{n.episode:02d}"
        groups[(n.season, n.episode)].append(f.path)
        ids = index.get((n.season, n.episode), set())
        if len(ids) == 1 and next(iter(ids)) in known:
            report.resolved[f.path] = next(iter(ids))
        elif not ids:
            if n.season in per_season:
                cnt = per_season[n.season]
                why = f"{label} is not in ordering (season {n.season} has {cnt} episodes)"
            else:
                why = f"{label} is not in ordering (no season {n.season})"
            report.unresolved[f.path] = why
        elif len(ids) > 1:
            report.unresolved[f.path] = f"{label} matches {len(ids)} episodes in the ordering"
        else:
            report.unresolved[f.path] = f"{label} points to an episode missing from the bundle"
    for key in sorted(groups):
        if len(groups[key]) > 1:
            report.duplicate_sets.append(sorted(groups[key]))
    return report
