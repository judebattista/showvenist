"""Ingest orchestration (spec §4.2). M1 sources: TMDB and local SRT."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from showvenist.adapters import localsubs
from showvenist.adapters.tmdb import TmdbAdapter
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
        if m := _TMDB_URL.search(ref):
            return int(m.group(1))
        if m := _IMDB_URL.search(ref):
            found = tmdb.find_by_imdb(m.group(1))
            if not found:
                raise IngestError(f"TMDB has no show for {m.group(1)}")
            return found[0]["tmdb_id"]
        raise IngestError(f"unsupported --ref {ref!r} (M1 handles TMDB and IMDb links)")
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


def ingest(
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
) -> tuple[Bundle, Path, localsubs.ScanReport | None]:
    tid = resolve_tmdb_id(tmdb, show, year, tmdb_id, refs, pick)
    partial = tmdb.fetch(tid)
    if partial.show is None:
        raise IngestError("TMDB returned no show data")
    bundle = Bundle(
        show=partial.show,
        episodes=partial.episodes,
        orderings=partial.orderings,
        provenance=partial.provenance,
    )
    slug = slugify(bundle.show.title, bundle.show.year)
    existing = bundle_path(slug, root) / "show.json"
    if existing.exists():
        if not refresh:
            raise IngestError(
                f"{existing} exists; bundles never update silently. Re-run with --refresh."
            )
        # carry reference subtitles forward: the files stay in subs/
        old = {e.id: e for e in load_bundle(slug, root, apply_overrides=False).episodes}
        for ep in bundle.episodes:
            if ep.id in old:
                ep.subtitles = list(old[ep.id].subtitles)
    report = None
    if subs_dir is not None:
        subs, report = localsubs.scan_folder(subs_dir, bundle)
        bundle = merge_partial(bundle, localsubs.install(subs, subs_dir, bundle_path(slug, root)))
    return bundle, save_bundle(bundle, root).parent, report
