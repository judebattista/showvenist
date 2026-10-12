"""Build a rename plan from a Report (spec §13.1-13.3, decisions 56-58).

Choices made where the spec leaves room (all on the safe side):
- The season folder of a file is its parent, or the parent's parent when the parent is the
  file's disc folder (`FileReport.disc` equals the parent's name) or an `extras` folder. Plain
  files outside a disc folder are therefore renamed in place; so are explicit files
  (`FileReport.explicit`, given as file arguments), even inside a disc folder (§13.2).
- When the naming order puts an episode in another season than the season folder's number (a
  folder name that parses as a season), the file needs a pin; a pinned file moves to an existing
  sibling folder of that season, and is skipped (`out_of_scope`) when no such folder exists:
  the planner never invents a folder name for a season.
- Unpinned files flagged `play_all` that were decided as an episode stay out of the plan
  (decision 49: a play-all is never given an episode name without review).
- A duplicate of an extra is left alone (`duplicate_of_extra`). A duplicate is renamed only
  when its primary got a target in this plan (a move, or already in place); a primary that is
  not eligible, unticked, out of scope or skipped for any other reason leaves its copies
  alone (`duplicate_primary_missing`). Apply also renames a copy only after its primary's
  move succeeded (`PlannedMove.primary`).
- A `dupe_X` that is already taken on disk bumps X (numbers are never reused) instead of
  adding a collision suffix.
- Paths are made absolute (`os.path.abspath`), so the move log works from any directory.
"""

from __future__ import annotations

import getpass
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from showvenist.bundle.schema import Bundle
from showvenist.hierarchy.marker import _find_ordering
from showvenist.hierarchy.parse import is_extras_folder, parse_season
from showvenist.rename import naming
from showvenist.rename.sidecars import find_sidecars
from showvenist.rename.types import Plan, PlannedMove, PlanOptions, SkippedFile
from showvenist.report_schema import FileReport, Report
from showvenist.state.history import History, episode_code

IGNORE_FILE = ".showvenist-ignore"
EXTRAS_DIR = "extras"


def writable_dir(path: Path) -> bool:
    """Write and search permission for this process, and not on a read-only mount. Writes
    nothing (no probe file)."""
    try:
        if not os.access(path, os.W_OK | os.X_OK):
            return False
        return not (os.statvfs(path).f_flag & os.ST_RDONLY)
    except OSError:
        return False


def device_of(path: Path) -> int:
    return os.stat(path).st_dev


def _listdir(folder: Path) -> list[str]:
    try:
        return os.listdir(folder)
    except OSError:
        return []


def resolve_naming_order(bundle: Bundle, name: str | None) -> str:
    """Bundle key of the naming order: `name` as a key, type or Jellyfin keyword (§13.1);
    None = 'aired' when the bundle has it, else its first ordering. Raises ValueError."""
    if name is None:
        if "aired" in bundle.orderings:
            return "aired"
        if not bundle.orderings:
            raise ValueError("the bundle has no orderings")
        return next(iter(bundle.orderings))
    o = _find_ordering(bundle, name)
    if o is None:
        raise ValueError(f"naming order {name!r} is not in the bundle "
                         f"(it has: {', '.join(sorted(bundle.orderings))})")
    return next(k for k, v in bundle.orderings.items() if v is o)


@dataclass
class _Video:
    """A file that passed the eligibility rules, before its target is fixed."""

    fr: FileReport
    old: Path
    kind: str  # episode | extra | duplicate
    target: Path | None = None
    episode_id: str | None = None
    season: int | None = None
    episode: int | None = None
    episode_end: int | None = None  # a two-episode file's last episode (decision 64)
    end_id: str | None = None  # and its id
    title: str | None = None
    stem: str | None = None  # the episode name (episodes and duplicates)
    pinned: bool = False
    confirmed: bool = False
    dupe_index: int | None = None
    primary_old: Path | None = None  # a duplicate's primary
    reason: str = ""


def _season_dir(
    old: Path, disc: str | None, explicit: bool = False
) -> tuple[Path, str | None]:
    """(season folder, disc folder name or None) for a file (see the module docstring). An
    explicit file (given as a file argument, §13.2) stays in its own folder, disc folder or not."""
    parent = old.parent
    if explicit and parent.name.lower() != EXTRAS_DIR:
        return parent, None
    if disc is not None and parent.name == disc:
        return parent.parent, parent.name
    if parent.name.lower() == EXTRAS_DIR:
        return parent.parent, None
    return parent, None


def _ignored(old: Path, exists: Callable[[Path], bool]) -> bool:
    return any(exists(d / IGNORE_FILE) for d in old.parents)


def _code(v: _Video) -> str:
    return episode_code(v.season, v.episode, v.episode_end)


def _flags_text(fr: FileReport) -> str:
    return f" [{', '.join(fr.flags)}]" if fr.flags else ""


def build_plan(
    report: Report,
    bundle: Bundle,
    opts: PlanOptions,
    *,
    history: History | None = None,
    exists: Callable[[Path], bool] = os.path.exists,
    sidecars: Callable[[Path], list[Path]] = find_sidecars,
    writable: Callable[[Path], bool] = writable_dir,
    device: Callable[[Path], int] = device_of,
) -> Plan:
    """Pure with respect to the media files: reads the Report and the bundle, asks `exists`
    and `sidecars`, never moves anything.

    In the plan (§13.2): files decided `episode` at high confidence, `extra` at high confidence,
    and pinned files at any confidence (flag `pinned`). Never: flags `named`, `not_ready`,
    `unreadable`, files under `.showvenist-ignore`, medium/low unpinned files, unpinned
    `out_of_scope` files. Skipped files are listed with a `SkipReason`.
    - Episodes: `<naming_order season folder>/<template>.<ext>`, flattened out of the disc
      folder; season/episode numbers come from `bundle.orderings[naming_order]` for the
      decided episode id (not necessarily the ordering that was solved). A two-episode file
      (`EpisodeRef.episode_end_id`, decision 64) is named with the range (`s01e11-e12`,
      `naming.range_template`); it is skipped (`cannot_name`) when its two episodes are not
      consecutive in the naming order or the template has no `{e}`.
    - Extras: `<Season>/extras/` with `naming.extra_name(disc folder, original)`. A confirmed
      extra (flag `confirmed`, or confirmed in `history`) is planned like any other extra
      (`assign --extra` without `--rename` moves nothing); an extra already inside an extras
      folder (flag `in_extras`, §5.1) is left there (`already_in_place`).
    - Duplicates (`FileReport.duplicate_of`): stay in their own folder, renamed in place with
      `naming.dupe_stem`, index from `max(history.highest_dupe_index, parse_dupe over the
      folder) + 1` upward in `duplicate_rank` order.
    - Collisions (planned vs planned, planned vs `exists`) get `naming.with_collision_index`
      (`_2`, `_3`, ...); existing files are never overwritten.
    - Sidecars follow their video (`naming.sidecar_target`), `parent` = the video's index.
    - Preflight (§13.3): write permission on every source and target directory, cross-
      filesystem moves listed in `cross_filesystem`; any failure goes into `errors`.

    `writable` (directory -> bool) and `device` (existing path -> st_dev) are injectable for
    tests, like `exists`.
    """
    order_key = resolve_naming_order(bundle, opts.naming_order)
    plan = Plan(show_slug=report.show_slug, naming_order=order_key,
                bundle_revision=report.bundle_revision or None)
    entries = {e.episode_id: e for e in bundle.orderings[order_key].entries}
    episodes = bundle.episode_by_id()
    exclude = {Path(os.path.abspath(p)) for p in opts.exclude}

    def skip(fr_or_path, reason: str, detail: str = "") -> None:
        path = fr_or_path if isinstance(fr_or_path, Path) else Path(fr_or_path.path)
        plan.skipped.append(SkippedFile(path, reason, detail))  # type: ignore[arg-type]

    eligible: dict[Path, _Video] = {}  # primary path -> its decision (for duplicates)
    videos: list[_Video] = []
    dups: list[tuple[FileReport, Path]] = []

    for fr in report.files:
        old = Path(os.path.abspath(fr.path))
        flags = set(fr.flags)
        if "named" in flags:
            skip(old, "named", "already named as an episode: never renamed automatically")
            continue
        if "not_ready" in flags:
            skip(old, "not_ready", "still being written")
            continue
        if "unreadable" in flags:
            skip(old, "unreadable")
            continue
        if _ignored(old, exists):
            skip(old, "ignored", f"inside a folder with {IGNORE_FILE}")
            continue
        if fr.duplicate_of is not None:
            dups.append((fr, old))
            continue
        if "duplicate" in flags or "near_duplicate" in flags:
            skip(old, "duplicate_primary_missing", "a duplicate whose primary is not known")
            continue
        if "in_extras" in flags or (fr.decision.kind == "extra"
                                    and is_extras_folder(old.parent.name)):
            skip(old, "already_in_place", f"already in {old.parent.name}/")
            continue
        pinned = "pinned" in flags
        confirmed = "confirmed" in flags
        if fr.decision.kind == "extra" and not confirmed and history is not None \
                and fr.fingerprint and history.is_confirmed_extra(fr.fingerprint):
            confirmed = True  # decided earlier: moved like any extra, its match linked
        if "out_of_scope" in flags and not pinned:
            skip(old, "out_of_scope", "evidence points to another season: pin it to move it")
            continue
        if fr.confidence.level != "high" and not pinned:
            skip(old, "below_high", f"{fr.confidence.level} confidence: needs review")
            continue
        if "play_all" in flags and fr.decision.kind == "episode" and not pinned:
            skip(old, "below_high", "play-all title decided as an episode: needs review")
            continue
        v = _Video(fr, old, fr.decision.kind, pinned=pinned, confirmed=confirmed)
        if fr.decision.kind == "episode":
            ref = fr.decision.episode
            ent = entries.get(ref.episode_id) if ref is not None else None
            if ref is None or ent is None:
                skip(old, "out_of_scope",
                     f"the episode has no number in the naming order {order_key!r}")
                continue
            v.episode_id, v.season, v.episode = ref.episode_id, ent.season, ent.episode
            if ref.episode_end_id is not None:  # a two-episode file (decision 64)
                end = entries.get(ref.episode_end_id)
                if end is None or end.season != ent.season or end.episode != ent.episode + 1:
                    skip(old, "cannot_name",
                         f"its two episodes are not consecutive in the naming order "
                         f"{order_key!r}")
                    continue
                v.episode_end, v.end_id = end.episode, ref.episode_end_id
            ep = episodes.get(ref.episode_id)
            v.title = ep.title if ep is not None else ref.title
            try:
                v.stem = naming.episode_stem(opts.template, bundle.show.title, ent.season,
                                             ent.episode, v.title, episode_end=v.episode_end)
            except ValueError as e:
                skip(old, "cannot_name", str(e))
                continue
        eligible[old] = v
        if old in exclude:
            skip(old, "excluded", "unticked")
            continue
        if fr.decision.kind == "extra" and not opts.move_extras:
            skip(old, "extras_disabled", "move_extras is off")
            continue
        videos.append(v)

    # --- targets: episodes in the naming order, then extras, then duplicates -------------------
    taken: set[Path] = set()

    def free(p: Path) -> bool:
        return p not in taken and not exists(p)

    def settle(target: Path, old: Path) -> tuple[Path, int | None]:
        if free(target) or target == old:
            return target, None
        n = 2
        while not free(naming.with_collision_index(target, n)):
            n += 1
        return naming.with_collision_index(target, n), n

    planned: list[tuple[_Video, Path, int | None]] = []
    placed: dict[Path, Path | None] = {}  # primaries with a target: new path (None = in place)
    eps = sorted((v for v in videos if v.kind == "episode"),
                 key=lambda v: (v.season, v.episode, str(v.old)))
    ends = {v.end_id: v for v in eps if v.end_id is not None}
    for v in eps:  # a two-episode file holds its second episode too (decision 64)
        if v.episode_id in ends:
            plan.notices.append(f"{v.old.name}: {_code(v)} is also held by the two-episode "
                                f"file {ends[v.episode_id].old.name} ({_code(ends[v.episode_id])})")
    extras = sorted((v for v in videos if v.kind == "extra"), key=lambda v: str(v.old))
    for v in eps + extras:
        season_dir, disc_dir = _season_dir(v.old, v.fr.disc, v.fr.explicit)
        ext = v.old.suffix
        if v.kind == "episode":
            folder_season = parse_season(season_dir.name)
            if folder_season is not None and folder_season != v.season:
                if not v.pinned:
                    skip(v.old, "out_of_scope",
                         f"{_code(v)} in {order_key!r} is not in "
                         f"{season_dir.name!r}: pin it to change season")
                    continue
                sib = [season_dir.parent / n for n in sorted(_listdir(season_dir.parent))
                       if parse_season(n) == v.season and (season_dir.parent / n).is_dir()]
                if not sib:
                    skip(v.old, "out_of_scope",
                         f"no folder for season {v.season} next to {season_dir.name!r}: "
                         "create it and plan again")
                    continue
                plan.notices.append(f"{v.old.name}: moves to another season folder, "
                                    f"{sib[0].name!r} (pinned)")
                season_dir = sib[0]
            target = season_dir / f"{v.stem}{ext}"
            v.reason = (f"{_code(v)} {v.title}, "
                        f"{v.fr.confidence.level} {v.fr.confidence.odds}{_flags_text(v.fr)}")
        else:
            target = season_dir / EXTRAS_DIR / naming.extra_name(disc_dir, v.old.name)
            v.reason = f"extra, {v.fr.confidence.level} {v.fr.confidence.odds}{_flags_text(v.fr)}"
            if v.confirmed and "confirmed" not in v.fr.flags:
                v.reason += " [confirmed]"
        if target == v.old:
            skip(v.old, "already_in_place")
            placed[v.old] = None
            continue
        target, n = settle(target, v.old)
        if n is not None:
            plan.notices.append(f"{target.name}: suffixed _{n}, the name was taken")
        taken.add(target)
        planned.append((v, target, n))
        placed[v.old] = target

    # duplicates: renamed in place, X per episode in rank order, never reused
    by_primary: dict[str, list[tuple[FileReport, Path]]] = {}
    for fr, old in dups:
        prim = eligible.get(Path(os.path.abspath(fr.duplicate_of)))  # type: ignore[arg-type]
        if prim is None:
            skip(old, "duplicate_primary_missing",
                 f"its primary {Path(fr.duplicate_of).name} is not in the plan")  # type: ignore[arg-type]
            continue
        if prim.kind != "episode":
            skip(old, "duplicate_of_extra", f"a copy of the extra {prim.old.name}: left as is")
            continue
        if prim.old not in placed:
            skip(old, "duplicate_primary_missing",
                 f"its primary {prim.old.name} is not moved in this plan: left as is")
            continue
        if old in exclude:
            skip(old, "excluded", "unticked")
            continue
        by_primary.setdefault(prim.episode_id, []).append((fr, old))  # type: ignore[arg-type]
    for eid in sorted(by_primary):
        copies = sorted(by_primary[eid],
                        key=lambda c: (c[0].duplicate_rank if c[0].duplicate_rank is not None
                                       else 1 << 30, str(c[1])))
        prim = eligible[Path(os.path.abspath(copies[0][0].duplicate_of))]  # type: ignore[arg-type]
        stem = prim.stem
        assert stem is not None
        mark = (stem + "_dupe_").lower()
        folders = {old.parent for _, old in copies} | {
            _season_dir(prim.old, prim.fr.disc, prim.fr.explicit)[0]}
        used = [naming.parse_dupe(n) or 0 for f in folders for n in _listdir(f)
                if n.lower().startswith(mark)]
        x = max([history.highest_dupe_index(report.show_slug, eid) if history else 0, *used])
        for fr, old in copies:
            while True:
                x += 1
                target = old.parent / f"{naming.dupe_stem(stem, x)}{old.suffix}"
                if free(target):
                    break
            v = _Video(fr, old, "duplicate", episode_id=eid, season=prim.season,
                       episode=prim.episode, episode_end=prim.episode_end, title=prim.title,
                       stem=stem, pinned=prim.pinned, dupe_index=x, primary_old=prim.old)
            v.reason = (f"duplicate of {prim.old.name} ({_code(prim)}),"
                        f" stays in its folder{_flags_text(fr)}")
            taken.add(target)
            planned.append((v, target, None))

    # --- moves with their sidecars --------------------------------------------------------
    claimed = {v.old for v, _, _ in planned}
    index_of: dict[Path, int] = {}  # video path -> its index in plan.moves
    for v, target, n in planned:
        level = v.fr.confidence.level
        plan.moves.append(PlannedMove(
            kind=v.kind,  # type: ignore[arg-type]
            old=v.old, new=target, fingerprint=v.fr.fingerprint, show_slug=report.show_slug,
            episode_id=v.episode_id, season=v.season, episode=v.episode,
            episode_end=v.episode_end, title=v.title, dupe_index=v.dupe_index,
            collision_index=n, pinned=v.pinned,
            confidence=level,  # type: ignore[arg-type]
            reason=v.reason, confirmed=v.confirmed, disc=v.fr.disc,
            primary=index_of.get(v.primary_old) if v.primary_old is not None else None,
        ))
        vi = len(plan.moves) - 1
        index_of[v.old] = vi
        for sc in sidecars(v.old):
            sc = Path(os.path.abspath(sc))
            if sc in claimed:
                continue
            claimed.add(sc)
            st = naming.sidecar_target(v.old, sc, target)
            st, sn = settle(st, sc)
            if sn is not None:
                plan.notices.append(f"{st.name}: suffixed _{sn}, the name was taken")
            taken.add(st)
            plan.moves.append(PlannedMove(
                kind="sidecar", old=sc, new=st, fingerprint=v.fr.fingerprint,
                show_slug=report.show_slug, parent=vi, collision_index=sn,
                reason=f"sidecar of {v.old.name}",
            ))

    _preflight(plan, exists, writable, device)
    return plan


def _nearest_existing(d: Path, exists: Callable[[Path], bool]) -> Path | None:
    for p in (d, *d.parents):
        if exists(p):
            return p
    return None


def _preflight(plan: Plan, exists, writable, device) -> None:
    """§13.3: every check before anything moves; failures land in `errors`."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = f"uid {os.getuid()}"
    checked: dict[Path, bool] = {}

    def can_write(d: Path) -> bool:
        if d not in checked:
            checked[d] = writable(d)
        return checked[d]

    for m in plan.moves:
        if not exists(m.old):
            plan.errors.append(f"{m.old}: the file is gone")
        if not can_write(m.old.parent):
            plan.errors.append(f"{m.old.parent}: no write permission for {user}")
        anc = _nearest_existing(m.new.parent, exists)
        if anc is None:
            plan.errors.append(f"{m.new.parent}: no existing folder above it")
            continue
        if os.path.lexists(anc) and not os.path.isdir(anc):
            plan.errors.append(f"{anc}: exists but is not a folder")
            continue
        if not can_write(anc):
            plan.errors.append(f"{anc}: no write permission for {user}")
        if m.parent is not None:
            continue  # a sidecar follows its video's filesystem decision
        try:
            cross = device(m.old) != device(anc)
        except OSError as e:
            plan.errors.append(f"{m.old}: cannot stat ({e})")
            continue
        if cross:
            plan.cross_filesystem.append(m)
    sidecar_cross = [m for m in plan.moves if m.parent is not None
                     and plan.moves[m.parent] in plan.cross_filesystem]
    plan.cross_filesystem += sidecar_cross
    if plan.cross_filesystem:
        plan.notices.append(
            f"{len(plan.cross_filesystem)} move(s) cross filesystems: they are copied, checked "
            "and only then removed from the old place, and need their own confirmation")
    # deterministic: errors in plan order, without repeats
    plan.errors = list(dict.fromkeys(plan.errors))


__all__ = ["build_plan", "device_of", "resolve_naming_order", "writable_dir"]
