"""The plan screen (`p`, spec §12.2) and the apply summary.

The screen shows exactly what `build_plan` returns for every finished season: each move as
`old -> new` grouped by season and disc, sidecars indented under their video, extras and
`_dupe_N` copies marked. Unticking a row adds its path to `PlanOptions.exclude` and rebuilds
the plan (unticking a primary also unticks its copies), so the collision check stays live.
The footer shows the plan token and the counts. The app owns building and applying.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from showvenist.rename.token import plan_token
from showvenist.rename.types import ApplyResult, Plan, PlannedMove
from showvenist.tui.dialogs import ConfirmDialog

Plans = list[tuple[str, Plan]]  # (season title, plan)


def _root(plan: Plan) -> Path | None:
    ps = [str(m.old) for m in plan.moves] + [str(m.new) for m in plan.moves] + [
        str(s.path) for s in plan.skipped]
    if not ps:
        return None
    try:
        root = Path(os.path.commonpath(ps))
    except ValueError:
        return None
    return root.parent if root.suffix else root


def _rel(p: Path, root: Path | None) -> str:
    if root is None:
        return str(p)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _marks(m: PlannedMove, plan: Plan) -> str:
    out = []
    if m.kind == "extra":
        out.append("extra")
    if m.kind == "duplicate":
        out.append(f"_dupe_{m.dupe_index}")
    if m.pinned:
        out.append("pinned")
    if m.collision_index is not None:
        out.append(f"suffixed _{m.collision_index}")
    if any(m is c for c in plan.cross_filesystem):
        out.append("cross-filesystem")
    return f"  [{', '.join(out)}]" if out else ""


def counts(plans: Plans, exclude: set[Path]) -> str:
    moves = [m for _, p in plans for m in p.moves]
    kinds = {k: sum(m.kind == k for m in moves) for k in ("episode", "extra", "duplicate",
                                                           "sidecar")}
    left = sum(1 for _, p in plans for s in p.skipped if s.reason != "excluded")
    unticked = sum(1 for _, p in plans for s in p.skipped if s.reason == "excluded")
    return (f"{len(moves)} moves ({kinds['episode']} episodes, {kinds['extra']} extras, "
            f"{kinds['duplicate']} copies, {kinds['sidecar']} sidecars) · {unticked} unticked"
            f" · {left} left where they are")


class PlanScreen(Screen[None]):
    BINDINGS = [
        Binding("space", "toggle", "Tick/untick"),
        Binding("enter", "apply", "Apply"),
        Binding("escape,q", "back", "Back"),
    ]
    DEFAULT_CSS = """
    PlanScreen #plan-list { height: 1fr; }
    PlanScreen #plan-footer { height: auto; padding: 0 1; background: $panel; }
    PlanScreen #plan-status { height: auto; padding: 0 1; color: $warning; }
    """

    def __init__(self, build: Callable[[set[Path]], Plans], exclude: set[Path],
                 can_apply: Callable[[], str | None],
                 start_apply: Callable[[Plans, bool], None]):
        super().__init__()
        self.build = build
        self.exclude = exclude  # the app's set: it survives leaving and reopening the screen
        self.can_apply = can_apply
        self.start_apply = start_apply
        self.plans: Plans = []
        self.dupes_of: dict[Path, list[Path]] = {}
        self.token = ""

    def compose(self) -> ComposeResult:
        yield Static("Plan: space ticks/unticks, Enter applies, Esc goes back", id="plan-title")
        yield OptionList(id="plan-list")
        yield Static("", id="plan-status")
        yield Static("", id="plan-footer")
        yield Footer()

    def on_mount(self) -> None:
        self.rebuild()
        self.query_one(OptionList).focus()

    # ---- content ------------------------------------------------------------------------

    def rebuild(self, keep: str | None = None) -> None:
        self.plans = self.build(set(self.exclude))
        self.token = plan_token([p for _, p in self.plans])
        opts: list[Option] = []
        for title, plan in self.plans:
            opts.append(Option(Text(title, style="bold reverse"), disabled=True))
            root = _root(plan)
            disc: object = object()
            videos = sorted((i for i, m in enumerate(plan.moves) if m.parent is None),
                            key=lambda i: (plan.moves[i].disc is None, plan.moves[i].disc or ""))
            for i in videos:
                m = plan.moves[i]
                if m.disc != disc:
                    disc = m.disc
                    opts.append(Option(Text(f" {m.disc or '(no disc)'}", style="bold"),
                                       disabled=True))
                opts.append(Option(Text(f"  [x] {_rel(m.old, root)} → {_rel(m.new, root)}"
                                        f"{_marks(m, plan)}"), id=f"m:{m.old}"))
                for s in plan.moves:
                    if s.parent == i:
                        opts.append(Option(Text(f"        ↳ {_rel(s.old, root)} → "
                                                f"{_rel(s.new, root)}", style="dim"),
                                           disabled=True))
            unticked = [s for s in plan.skipped if s.reason == "excluded"]
            if unticked:
                opts.append(Option(Text(" Unticked", style="bold"), disabled=True))
                opts += [Option(Text(f"  [ ] {_rel(s.path, root)}"), id=f"x:{s.path}")
                         for s in unticked]
            left = [s for s in plan.skipped if s.reason != "excluded"]
            if left:
                opts.append(Option(Text(f" Left where they are ({len(left)})", style="bold"),
                                   disabled=True))
                opts += [Option(Text(f"      {_rel(s.path, root)}: {s.reason}"
                                     + (f" ({s.detail})" if s.detail else ""), style="dim"),
                                disabled=True) for s in left]
            opts += [Option(Text(f" {n}", style="dim"), disabled=True) for n in plan.notices]
            opts += [Option(Text(f" ERROR {e}", style="bold red"), disabled=True)
                     for e in plan.errors]
        if not self.plans:
            opts.append(Option(Text("No finished season to plan yet."), disabled=True))
        ol = self.query_one(OptionList)
        ol.set_options(opts)
        ids = [o.id for o in ol.options]
        target = None
        if keep is not None:
            path = keep[2:]
            for cand in (f"m:{path}", f"x:{path}"):
                if cand in ids:
                    target = ids.index(cand)
                    break
        if target is None:
            target = next((k for k, o in enumerate(ol.options) if not o.disabled), None)
        ol.highlighted = target
        errors = sum(len(p.errors) for _, p in self.plans)
        self.query_one("#plan-footer", Static).update(
            f"Plan token {self.token} · {counts(self.plans, self.exclude)}"
            + (f" · {errors} error(s): nothing can be applied" if errors else ""))

    @property
    def moves(self) -> int:
        return sum(len(p.moves) for _, p in self.plans)

    def status(self, text: str) -> None:
        self.query_one("#plan-status", Static).update(text)

    # ---- keys ---------------------------------------------------------------------------

    def action_toggle(self) -> None:
        ol = self.query_one(OptionList)
        if ol.highlighted is None:
            return
        oid = ol.get_option_at_index(ol.highlighted).id
        if not oid:
            return
        path = Path(oid[2:])
        if oid.startswith("m:"):
            dupes = []
            for _, plan in self.plans:
                idx = next((i for i, m in enumerate(plan.moves) if m.old == path), None)
                if idx is not None:
                    dupes += [m.old for m in plan.moves if m.primary == idx]
            self.dupes_of[path] = dupes
            self.exclude.add(path)
            self.exclude.update(dupes)
        else:
            self.exclude.discard(path)
            for d in self.dupes_of.pop(path, []):
                self.exclude.discard(d)
        self.rebuild(keep=oid)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_apply()

    def action_apply(self) -> None:
        why = self.can_apply()
        if why:
            self.status(why)
            return
        if any(p.errors for _, p in self.plans):
            self.status("The plan has errors (listed above): nothing can be applied.")
            return
        if not self.moves:
            self.status("Nothing to move.")
            return
        plans = list(self.plans)
        cross = sum(len(p.cross_filesystem) for _, p in plans)

        def first(yes: bool | None) -> None:
            if not yes:
                self.status("Nothing was moved.")
                return
            if not cross:
                self.start_apply(plans, False)
                return
            self.app.push_screen(ConfirmDialog(
                f"{cross} move(s) cross filesystems: each file is copied, checked, and only "
                "then removed from its old place. Move those too?"),
                lambda ok: self.start_apply(plans, bool(ok)))

        self.app.push_screen(ConfirmDialog(
            f"Move {self.moves} file(s) as listed (plan {self.token})?"), first)

    def action_back(self) -> None:
        self.dismiss(None)


def summary_text(results: list[tuple[str, ApplyResult]], recovery: str = "",
                 error: str | None = None) -> str:
    lines = [recovery.rstrip("\n")] if recovery else []
    if error:
        lines.append(f"Not applied: {error}")
    run = next((r.run_id for _, r in results if r.run_id), None)
    for title, res in results:
        lines.append(f"{title}: moved {len(res.done)} file(s), {len(res.failed)} failed")
        lines += [f"  {m.old.name} → {m.new}" for m in res.done]
        lines += [f"  FAILED {m.old}: {why}" for m, why in res.failed]
    if run:
        lines.append(f"Run {run}. Undo with: showvenist undo --last")
    elif not error:
        lines.append("Nothing was moved.")
    return "\n".join(lines)


class SummaryScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,enter,q", "close", "Close")]
    DEFAULT_CSS = """
    SummaryScreen > Static { width: 100%; height: auto; padding: 1 2; border: thick $accent;
                             background: $panel; }
    """

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        yield Static(self.text + "\n\nEnter closes", id="summary-text")

    def action_close(self) -> None:
        self.dismiss(None)
