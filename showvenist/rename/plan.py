"""Build a rename plan from a Report (spec §13.1-13.3, decisions 56-58). Contract; the body is
task sv-o4f.3."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from showvenist.bundle.schema import Bundle
from showvenist.rename.sidecars import find_sidecars
from showvenist.rename.types import Plan, PlanOptions
from showvenist.report_schema import Report
from showvenist.state.history import History


def build_plan(
    report: Report,
    bundle: Bundle,
    opts: PlanOptions,
    *,
    history: History | None = None,
    exists: Callable[[Path], bool] = os.path.exists,
    sidecars: Callable[[Path], list[Path]] = find_sidecars,
) -> Plan:
    """Pure with respect to the media files: reads the Report and the bundle, asks `exists`
    and `sidecars`, never moves anything.

    In the plan (§13.2): files decided `episode` at high confidence, `extra` at high confidence,
    and pinned files at any confidence (flag `pinned`). Never: flags `named`, `not_ready`,
    `unreadable`, files under `.showvenist-ignore`, medium/low unpinned files, unpinned
    `out_of_scope` files. Skipped files are listed with a `SkipReason`.
    - Episodes: `<naming_order season folder>/<template>.<ext>`, flattened out of the disc
      folder; season/episode numbers come from `bundle.orderings[naming_order]` for the
      decided episode id (not necessarily the ordering that was solved).
    - Extras: `<Season>/extras/` with `naming.extra_name(disc folder, original)`; a confirmed
      extra in `history` is skipped.
    - Duplicates (`FileReport.duplicate_of`): stay in their own folder, renamed in place with
      `naming.dupe_stem`, index from `max(history.highest_dupe_index, parse_dupe over the
      folder) + 1` upward in `duplicate_rank` order.
    - Collisions (planned vs planned, planned vs `exists`) get `naming.with_collision_index`
      (`_2`, `_3`, ...); existing files are never overwritten.
    - Sidecars follow their video (`naming.sidecar_target`), `parent` = the video's index.
    - Preflight (§13.3): write permission on every source and target directory, cross-
      filesystem moves listed in `cross_filesystem`; any failure goes into `errors`.
    """
    raise NotImplementedError("sv-o4f.3")
