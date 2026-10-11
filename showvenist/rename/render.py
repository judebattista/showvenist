"""Text rendering of plans, apply results and undo listings (rich tables, as report mode)."""

from __future__ import annotations

import io
import os
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from showvenist.rename.types import ApplyResult, Plan, RecoveryReport, UndoResult
from showvenist.state.db import RenameOp


def _console(buf: io.StringIO) -> Console:
    return Console(file=buf, width=160, force_terminal=False, color_system=None,
                   markup=False, highlight=False)


def _row(*cells: object) -> list[Text]:
    """Literal cells: paths and flags in brackets ('[tmdbid-1437]', '[pinned]') are not markup."""
    return [Text(str(c)) for c in cells]


def _root(paths: Iterable[Path]) -> Path | None:
    ps = [str(p) for p in paths]
    if not ps:
        return None
    try:
        return Path(os.path.commonpath(ps))
    except ValueError:
        return None


def _rel(p: Path, root: Path | None) -> str:
    if root is None or str(root) in ("/", ""):
        return str(p)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def format_plan(plan: Plan) -> str:
    """Every move (old -> new, kind, reason with flags), every skipped file with its reason,
    the notices, the cross-filesystem moves and the preflight errors."""
    buf = io.StringIO()
    con = _console(buf)
    root = _root([m.old for m in plan.moves] + [m.new for m in plan.moves]
                 + [s.path for s in plan.skipped])
    if root is not None and root.suffix:  # one file: show its folder as the root
        root = root.parent
    con.print(f"Plan for {plan.show_slug} (naming order {plan.naming_order})"
              + (f", under {root}" if root is not None else ""))
    if plan.moves:
        t = Table(title=f"Moves ({len(plan.moves)})", title_justify="left")
        for col in ("#", "Kind", "From", "To", "Why"):
            t.add_column(col, overflow="fold")
        for i, m in enumerate(plan.moves, 1):
            why = m.reason + (" [cross-filesystem]" if m in plan.cross_filesystem else "")
            t.add_row(*_row(i, m.kind, _rel(m.old, root), _rel(m.new, root), why))
        con.print(t)
    else:
        con.print("No moves.")
    if plan.skipped:
        t = Table(title=f"Left where they are ({len(plan.skipped)})", title_justify="left")
        for col in ("File", "Reason", "Detail"):
            t.add_column(col, overflow="fold")
        for s in plan.skipped:
            t.add_row(*_row(_rel(s.path, root), s.reason, s.detail))
        con.print(t)
    for n in plan.notices:
        con.print(n)
    if plan.errors:
        con.print("Errors (nothing can be applied):")
        for e in plan.errors:
            con.print(f"  {e}")
    return buf.getvalue()


def format_recovery(rec: RecoveryReport | None) -> str:
    """What recovery did with leftovers of an interrupted run; '' when there were none."""
    if rec is None or (rec.empty and not rec.notes):
        return ""
    lines = [f"Recovered from an earlier interrupted run: {len(rec.completed)} completed, "
             f"{len(rec.rolled_back)} rolled back, {len(rec.unresolved)} unresolved"
             + (f", {len(rec.removed_partials)} partial copies removed"
                if rec.removed_partials else "")]
    lines += [f"  {n}" for n in rec.notes]
    return "\n".join(lines) + "\n"


def format_apply(res: ApplyResult) -> str:
    buf = io.StringIO()
    con = _console(buf)
    rec = format_recovery(res.recovered)
    if rec:
        con.print(rec, end="")
    if res.cancelled:
        con.print("Nothing was moved.")
        return buf.getvalue()
    con.print(f"Moved {len(res.done)} file(s), {len(res.failed)} failed (run {res.run_id}).")
    for m in res.done:
        if m.collision_index is not None:
            con.print(f"  {m.old.name} -> {m.new.name} (suffixed _{m.collision_index})")
    for m, why in res.failed:
        con.print(f"  FAILED {m.old}: {why}")
    if res.done:
        con.print(f"Undo with: showvenist undo {res.run_id}")
    return buf.getvalue()


def format_ops(ops: list[RenameOp]) -> str:
    """An undo listing: each move back, in the order it will happen (`ops` as `undo_run`
    passes them, already in undo order)."""
    buf = io.StringIO()
    con = _console(buf)
    paths = [Path(o.old_path) for o in ops] + [Path(o.new_path) for o in ops]
    root = _root(paths)
    t = Table(title=Text(f"Moves to reverse ({len(ops)})"
                         + (f", under {root}" if root is not None else "")),
              title_justify="left")
    for col in ("Kind", "Now at", "Back to"):
        t.add_column(col, overflow="fold")
    for o in ops:
        t.add_row(*_row(o.kind, _rel(Path(o.new_path), root), _rel(Path(o.old_path), root)))
    con.print(t)
    return buf.getvalue()


def format_undo(res: UndoResult) -> str:
    rec = format_recovery(res.recovered)
    if res.cancelled:
        return rec + "Nothing was moved."
    lines = [rec.rstrip("\n")] if rec else []
    lines += [f"Run {res.run_id}: moved back {len(res.undone)}, skipped {len(res.skipped)}."]
    lines += [f"  skipped {o.new_path}: {why}" for o, why in res.skipped]
    return "\n".join(lines)
