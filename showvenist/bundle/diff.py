"""Refresh diff (spec §6.8): what changed between the bundle on disk and a freshly merged one.

`diff_bundles` is pure. `history_impact` reads confirmed matches from History (read-only) and
never writes. `render_diff` makes concise text: counts plus short lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from showvenist.bundle.schema import Bundle, Episode, Ordering

LIMIT = 8  # lines of any one list before "… and N more"
EXAMPLES = 3  # examples per changed field


def label(season: int, episode: int) -> str:
    return f"S{season:02d}E{episode:02d}"


@dataclass
class FieldChange:
    """Changes to one episode field across episodes present in both bundles."""

    count: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.count += 1
        if len(self.examples) < EXAMPLES:
            self.examples.append(text)


FIELDS = ("title", "alt_titles", "synopses", "credits", "runtime_min", "air_date")


@dataclass
class BundleDiff:
    episodes_added: list[tuple[str, str]] = field(default_factory=list)  # (id, title)
    episodes_removed: list[tuple[str, str]] = field(default_factory=list)
    # ordering key -> [(episode id, old label, new label)]
    renumbered: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)
    orderings_added: list[str] = field(default_factory=list)
    orderings_removed: list[str] = field(default_factory=list)
    # ordering type -> (old first-ranked key, new first-ranked key)
    first_ranked: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    changes: dict[str, FieldChange] = field(default_factory=dict)
    subtitles_added: int = 0
    conflicts_new: list[str] = field(default_factory=list)
    conflicts_resolved: list[str] = field(default_factory=list)
    show: list[str] = field(default_factory=list)  # "in_production: True -> False"

    @property
    def empty(self) -> bool:
        return not (
            self.episodes_added or self.episodes_removed or self.renumbered
            or self.orderings_added or self.orderings_removed or self.first_ranked
            or self.changes or self.subtitles_added or self.conflicts_new
            or self.conflicts_resolved or self.show
        )


@dataclass(frozen=True)
class ImpactRow:
    path: str
    episode_id: str
    old: str  # label as recorded, e.g. S01E02
    new: str  # new label, or "removed" / "not in <ordering>"


# --- helpers --------------------------------------------------------------------------------


def _numbers(ordering: Ordering) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for e in ordering.entries:
        out.setdefault(e.episode_id, (e.season, e.episode))
    return out


def first_ranked(b: Bundle) -> dict[str, str | None]:
    """Per ordering type, the key Jellyfin uses for it: the only group, else the one with
    type_rank 1 (§13.1; same rule as marker._find_ordering). None = no single first group."""
    by_type: dict[str, list[str]] = {}
    for k, o in b.orderings.items():
        by_type.setdefault(o.type, []).append(k)
    out: dict[str, str | None] = {}
    for t, keys in by_type.items():
        if len(keys) == 1:
            out[t] = keys[0]
        else:
            ranked = [k for k in keys if b.orderings[k].type_rank == 1]
            out[t] = ranked[0] if len(ranked) == 1 else None
    return out


def _conflict_key(c: dict[str, str]) -> tuple[str, str, str, str]:
    return (c.get("kind", ""), c.get("season", ""), c.get("primary", ""), c.get("secondary", ""))


def _conflict_text(c: dict[str, str]) -> str:
    where = f"season {c['season']}" if c.get("season") else "show"
    ids = " / ".join(x for x in (c.get("primary"), c.get("secondary")) if x)
    return f"{c.get('kind', '?')} ({where}): {ids}".rstrip(": ")


def _conflicts(b: Bundle) -> dict[tuple[str, str, str, str], dict[str, str]]:
    return {_conflict_key(p.conflict): p.conflict for p in b.provenance if p.conflict}


def _credits(ep: Episode) -> tuple[Any, ...]:
    return (tuple(ep.directors), tuple(ep.writers),
            tuple((c.actor, c.character) for c in ep.guest_cast))


def _syn(ep: Episode) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in ep.synopses:
        out.setdefault(s.source, s.text)
    return out


def _short(text: str | None, n: int = 40) -> str:
    t = " ".join((text or "").split())
    return repr(t if len(t) <= n else t[: n - 1] + "…")


def _compare_episode(d: BundleDiff, o: Episode, n: Episode) -> None:
    def add(name: str, text: str) -> None:
        d.changes.setdefault(name, FieldChange()).add(f"{n.id} {_short(n.title, 24)}: {text}")

    if o.title != n.title:
        add("title", f"{_short(o.title)} -> {_short(n.title)}")
    new_alts = [t for t in n.alt_titles if t not in o.alt_titles]
    if new_alts:
        add("alt_titles", "+ " + ", ".join(_short(t) for t in new_alts))
    os_, ns = _syn(o), _syn(n)
    if os_ != ns:
        parts = [f"+{src}" for src in ns if src not in os_]
        parts += [f"-{src}" for src in os_ if src not in ns]
        parts += [f"~{src}" for src in ns if src in os_ and ns[src] != os_[src]]
        add("synopses", " ".join(parts))
    if _credits(o) != _credits(n):
        add("credits", "directors/writers/guest cast changed")
    for name in ("runtime_min", "air_date"):
        a, b = getattr(o, name), getattr(n, name)
        if a != b:
            add(name, f"{a} -> {b}")
    have = {(s.lang, s.source) for s in o.subtitles}
    d.subtitles_added += sum((s.lang, s.source) not in have for s in n.subtitles)


# --- diff -----------------------------------------------------------------------------------


def diff_bundles(old: Bundle, new: Bundle) -> BundleDiff:
    d = BundleDiff()
    oe, ne = old.episode_by_id(), new.episode_by_id()
    d.episodes_added = [(i, e.title) for i, e in ne.items() if i not in oe]
    d.episodes_removed = [(i, e.title) for i, e in oe.items() if i not in ne]
    for i, n in ne.items():
        if i in oe:
            _compare_episode(d, oe[i], n)

    d.orderings_added = sorted(set(new.orderings) - set(old.orderings))
    d.orderings_removed = sorted(set(old.orderings) - set(new.orderings))
    for k in sorted(set(old.orderings) & set(new.orderings)):
        a, b = _numbers(old.orderings[k]), _numbers(new.orderings[k])
        rows = [(i, label(*a[i]), label(*b[i])) for i in a if i in b and a[i] != b[i]]
        if rows:
            d.renumbered[k] = rows
    fo, fn = first_ranked(old), first_ranked(new)
    for t in sorted(set(fo) & set(fn)):
        if fo[t] != fn[t]:
            d.first_ranked[t] = (fo[t], fn[t])

    co, cn = _conflicts(old), _conflicts(new)
    d.conflicts_new = [_conflict_text(c) for k, c in cn.items() if k not in co]
    d.conflicts_resolved = [_conflict_text(c) for k, c in co.items() if k not in cn]

    for name in ("title", "year", "language", "in_production"):
        a, b = getattr(old.show, name), getattr(new.show, name)
        if a != b:
            d.show.append(f"{name}: {a} -> {b}")
    for name in type(old.show.ids).model_fields:
        a, b = getattr(old.show.ids, name), getattr(new.show.ids, name)
        if a != b:
            d.show.append(f"ids.{name}: {a} -> {b}")
    return d


# --- history impact -------------------------------------------------------------------------


def history_impact(old: Bundle, new: Bundle, hist: Any, slug: str) -> list[ImpactRow]:
    """Confirmed matches (read-only) whose episode was removed, or whose recorded number no
    longer holds in the orderings it was recorded under. A match is judged against every old
    ordering that gave its episode the recorded (season, episode); it is unaffected if any of
    them still does. Nothing is written."""
    if hist is None:
        return []
    old_nums = {k: _numbers(o) for k, o in old.orderings.items()}
    new_nums = {k: _numbers(o) for k, o in new.orderings.items()}
    new_ids = set(new.episode_by_id())
    rows: list[ImpactRow] = []
    for m in hist.confirmed(slug):
        if m.episode_id is None or m.season is None or m.episode is None:
            continue  # extras
        path = _path(hist, m.fingerprint)
        recorded = label(m.season, m.episode)
        if m.episode_id not in new_ids:
            rows.append(ImpactRow(path, m.episode_id, recorded, "removed"))
            continue
        keys = [k for k, nums in old_nums.items()
                if nums.get(m.episode_id) == (m.season, m.episode)]
        if not keys:
            continue  # recorded under a numbering the old bundle does not have
        outcomes = [new_nums.get(k, {}).get(m.episode_id) for k in keys]
        if any(o == (m.season, m.episode) for o in outcomes):
            continue
        k = next(k for k, o in zip(keys, outcomes, strict=True) if o is not None) \
            if any(o is not None for o in outcomes) else None
        if k is None:
            rows.append(ImpactRow(path, m.episode_id, recorded, f"not in {keys[0]}"))
        else:
            rows.append(ImpactRow(path, m.episode_id, recorded, label(*new_nums[k][m.episode_id])))
    return rows


def _path(hist: Any, fingerprint: str) -> str:
    f = hist.get_file(fingerprint)
    return f.last_path if f is not None else f"(fingerprint {fingerprint[:12]})"


def load_history_impact(
    old: Bundle, new: Bundle, slug: str, db_path: Path | str | None = None
) -> tuple[list[ImpactRow], str | None]:
    """Open state.db read-only and compute the impact. No state.db, or one that needs an
    upgrade, gives ([], note)."""
    from showvenist.state.db import SchemaOutdated, StateDb, StateDbMissing
    from showvenist.state.history import History

    try:
        db = StateDb.open_readonly(db_path)
    except StateDbMissing:
        return [], "no history yet (no state.db)"
    except SchemaOutdated as e:
        return [], f"history not checked: {e}"
    try:
        return history_impact(old, new, History(db), slug), None
    finally:
        db.close()


# --- rendering ------------------------------------------------------------------------------


def _capped(lines: list[str], limit: int = LIMIT) -> list[str]:
    if len(lines) <= limit:
        return lines
    return lines[:limit] + [f"… and {len(lines) - limit} more"]


def render_diff(
    diff: BundleDiff, impact: list[ImpactRow] | None = None, note: str | None = None
) -> str:
    out: list[str] = []

    def section(head: str, lines: list[str]) -> None:
        if lines:
            out.append(head)
            out.extend("  " + ln for ln in _capped(lines))

    if diff.empty:
        out.append("No changes to the bundle.")
    section(f"Episodes added ({len(diff.episodes_added)}):",
            [f"{i}  {t}" for i, t in diff.episodes_added])
    section(f"Episodes removed ({len(diff.episodes_removed)}):",
            [f"{i}  {t}" for i, t in diff.episodes_removed])
    for k, rows in diff.renumbered.items():
        section(f"Renumbered in {k} ({len(rows)}):", [f"{i}  {a} -> {b}" for i, a, b in rows])
    section(f"Orderings added ({len(diff.orderings_added)}):", diff.orderings_added)
    section(f"Orderings removed ({len(diff.orderings_removed)}):", diff.orderings_removed)
    section("First-ranked group changed (what Jellyfin uses for the type):",
            [f"{t}: {a or '(none)'} -> {b or '(none)'}"
             for t, (a, b) in diff.first_ranked.items()])
    for name in FIELDS:
        ch = diff.changes.get(name)
        if ch:
            section(f"{name.replace('_', ' ').capitalize()} changed ({ch.count}):",
                    ch.examples + ([f"… and {ch.count - len(ch.examples)} more"]
                                   if ch.count > len(ch.examples) else []))
    if diff.subtitles_added:
        out.append(f"Reference subtitles added: {diff.subtitles_added}")
    section(f"Conflicts new ({len(diff.conflicts_new)}):", diff.conflicts_new)
    section(f"Conflicts resolved ({len(diff.conflicts_resolved)}):", diff.conflicts_resolved)
    section("Show:", diff.show)
    if impact:
        section(f"History impact ({len(impact)} confirmed matches):",
                [f"{r.path}: {r.old} -> {r.new} ({r.episode_id})" for r in impact])
    elif note:
        out.append(f"History impact: {note}")
    else:
        out.append("History impact: none")
    return "\n".join(out)


def staleness_notice(
    bundle: Bundle, slug: str, days: int, now: datetime | None = None
) -> str | None:
    """identify's notice (§6.8): the bundle was fetched more than `days` days ago and the show
    is still in production (as recorded at the last fetch; identify stays offline)."""
    if not bundle.show.in_production:
        return None
    stamps = [p.fetched_at for p in bundle.provenance if p.adapter == "tmdb" and p.fetched_at]
    if not stamps:
        return None
    try:
        fetched = datetime.fromisoformat(max(stamps).replace("Z", "+00:00"))
    except ValueError:
        return None
    age = ((now or datetime.now(UTC)) - fetched).days
    if age <= days:
        return None
    return (f"notice: bundle {slug} was fetched {age} days ago and the show is still in "
            f"production; see what changed with `showvenist ingest {slug} --refresh --diff`")
