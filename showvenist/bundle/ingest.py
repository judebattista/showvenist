"""Ingest orchestration (spec §4.2). M1 sources: TMDB and local SRT."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from showvenist.adapters import localsubs
from showvenist.adapters.tmdb import TmdbAdapter
from showvenist.adapters.tvmaze import TvmazeAdapter, TvmazeError, tvmaze_id_from_ref
from showvenist.bundle.merge import merge_sources
from showvenist.bundle.schema import Bundle, PartialBundle
from showvenist.bundle.store import bundle_path, load_bundle, save_bundle, slugify

_TMDB_URL = re.compile(r"themoviedb\.org/tv/(\d+)")
_IMDB_URL = re.compile(r"imdb\.com/title/(tt\d+)")


class IngestError(Exception):
    pass


def resolve_tmdb_id(
    tmdb: TmdbAdapter,
    show: str,
    year: int | None = None,
    tmdb_id: int | None = None,
    refs: tuple[str, ...] = (),
    pick: Callable[[list[dict]], int] | None = None,
) -> int:
    """tmdb_id > --ref > search. Ambiguity: `pick` (interactive) or fail listing candidates."""
    if tmdb_id:
        return tmdb_id
    for ref in refs:
        if tvmaze_id_from_ref(ref) is not None:
            continue  # routed to TVmaze
        if m := _TMDB_URL.search(ref):
            return int(m.group(1))
        if m := _IMDB_URL.search(ref):
            found = tmdb.find_by_imdb(m.group(1))
            if not found:
                raise IngestError(f"TMDB has no show for {m.group(1)}")
            return found[0]["tmdb_id"]
        raise IngestError(f"unsupported --ref {ref!r} (TMDB, TVmaze and IMDb links are handled)")
    cands = tmdb.search(show, year)
    if not cands:
        raise IngestError(f"no TMDB results for {show!r}")
    exact = [c for c in cands if c["title"].lower() == show.lower()]
    if len(exact) == 1 or len(cands) == 1:
        return (exact or cands)[0]["tmdb_id"]
    if pick is not None:
        return pick(exact or cands)
    lines = "\n".join(f"  {c['tmdb_id']}  {c['title']} ({c['year']})" for c in (exact or cands)[:8])
    raise IngestError(f"{show!r} is ambiguous; use --year or --tmdb-id:\n{lines}")


def merge_partial(bundle: Bundle, partial: PartialBundle) -> Bundle:
    """Add an adapter's subtitle refs to existing episodes (M7 owns the full merge)."""
    by_id = {e.id: e for e in bundle.episodes}
    for pe in partial.episodes:
        ep = by_id.get(pe.id)
        if ep is None:
            continue
        have = {(s.lang, s.source) for s in ep.subtitles}
        ep.subtitles += [s for s in pe.subtitles if (s.lang, s.source) not in have]
    bundle.provenance += partial.provenance
    return bundle


def resolve_tvmaze_id(
    tvmaze: TvmazeAdapter,
    title: str,
    year: int | None,
    refs: tuple[str, ...] = (),
    known: int | None = None,
) -> int:
    """--ref (tvmaze URL or `tvmaze:ID`) > the id already in the bundle > the single exact
    title+year search match. No interactive pick in v1. Raises IngestError."""
    for ref in refs:
        if (tid := tvmaze_id_from_ref(ref)) is not None:
            return tid
    if known:
        return known
    try:
        cands = tvmaze.search(title, year)
    except TvmazeError as e:
        raise IngestError(f"tvmaze: {e}") from None
    exact = [c for c in cands if c["title"].casefold() == title.casefold()
             and (year is None or c["year"] == year)]
    if len(exact) != 1:
        raise IngestError("tvmaze: ambiguous or not found (pass --ref tvmaze:ID or a TVmaze URL)")
    return exact[0]["tvmaze_id"]


@dataclass
class Built:
    """A merged bundle in memory, ready to diff and then `commit`."""

    bundle: Bundle
    slug: str
    root: Path | None
    existing: Bundle | None = None  # the bundle on disk (no overrides applied), on a refresh
    warnings: list[str] = field(default_factory=list)
    scan: localsubs.ScanReport | None = None
    subs_dir: Path | None = None
    _subs: PartialBundle | None = None  # scanned --subs, installed by commit


def build(
    tmdb: TmdbAdapter,
    show: str,
    *,
    year: int | None = None,
    tmdb_id: int | None = None,
    refs: tuple[str, ...] = (),
    subs_dir: Path | None = None,
    root: Path | None = None,
    pick: Callable[[list[dict]], int] | None = None,
    refresh: bool = False,
    sources: Sequence[str] = ("tmdb",),
    tvmaze: TvmazeAdapter | None = None,
) -> Built:
    """Fetch the sources and merge them in memory. Writes nothing. TMDB is primary and required
    (its errors propagate); a secondary that fails or can't find the show is a warning."""
    tid = resolve_tmdb_id(tmdb, show, year, tmdb_id, refs, pick)
    partial = tmdb.fetch(tid)
    if partial.show is None:
        raise IngestError("TMDB returned no show data")
    bundle = Bundle(
        show=partial.show,
        episodes=partial.episodes,
        orderings=partial.orderings,
        provenance=partial.provenance,
        series_cast=partial.series_cast,
    )
    slug = slugify(bundle.show.title, bundle.show.year)
    existing_path = bundle_path(slug, root) / "show.json"
    old: Bundle | None = None
    if existing_path.exists():
        if not refresh:
            raise IngestError(
                f"{existing_path} exists; bundles never update silently. Re-run with --refresh."
            )
        old = load_bundle(slug, root, apply_overrides=False)
        # carry reference subtitles forward: the files stay in subs/
        old_eps = {e.id: e for e in old.episodes}
        for ep in bundle.episodes:
            if ep.id in old_eps:
                ep.subtitles = list(old_eps[ep.id].subtitles)
    warnings: list[str] = []
    secondaries: list[PartialBundle] = []
    if "tvmaze" in sources:
        try:
            if tvmaze is None:
                raise IngestError("tvmaze: no adapter available")
            tvid = resolve_tvmaze_id(
                tvmaze, bundle.show.title, bundle.show.year or year, refs,
                old.show.ids.tvmaze if old else None)
            try:
                secondaries.append(tvmaze.fetch(tvid))
            except TvmazeError as e:
                raise IngestError(f"tvmaze: {e}") from None
        except IngestError as e:
            warnings.append(f"{e}; the bundle has no TVmaze data")
    if secondaries:
        bundle = merge_sources(bundle, secondaries)
    built = Built(bundle=bundle, slug=slug, root=root, existing=old, warnings=warnings)
    if subs_dir is not None:
        subs, built.scan = localsubs.scan_folder(subs_dir, bundle)
        built.subs_dir, built._subs = subs_dir, subs
        # preview the installed paths in a scratch directory; commit copies the files for real
        with tempfile.TemporaryDirectory() as tmp:
            preview = localsubs.install(subs, subs_dir, Path(tmp))
        built.bundle = merge_partial(bundle, preview)
    return built


def commit(built: Built) -> Path:
    """Write the built bundle: install --subs, keep show.json.prev on a refresh, save
    show.json atomically. overrides.yaml is never touched. Returns the bundle directory."""
    d = bundle_path(built.slug, built.root)
    bundle = built.bundle
    if built.subs_dir is not None and built._subs is not None:
        localsubs.install(built._subs, built.subs_dir, d)  # same relative paths as the preview
    cur = d / "show.json"
    if built.existing is not None and cur.exists():
        prev = d / "show.json.prev"
        tmp = d / ".show.json.prev.tmp"
        shutil.copy2(cur, tmp)
        os.replace(tmp, prev)
    return save_bundle(bundle, built.root).parent


def ingest(
    tmdb: TmdbAdapter,
    show: str,
    **kw,
) -> tuple[Bundle, Path, localsubs.ScanReport | None, list[str]]:
    """build + commit. Returns (bundle, bundle dir, subtitle scan report, warnings)."""
    built = build(tmdb, show, **kw)
    bdir = commit(built)
    return built.bundle, bdir, built.scan, built.warnings
