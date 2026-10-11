"""Left pane: the season -> disc -> file list (spec §12.2).

Rows come from the season's last `Report` (after a solve) or from the analysis progress (while
it reads); nothing here decides a match. `build_rows` is pure so it can be tested without a
screen; `SeasonTree` shows the rows in an OptionList (disc headers are disabled options).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual.binding import Binding
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from showvenist.report_schema import FileReport, Report
from showvenist.session import disc_label

GLYPH_HIGH = "✓"
GLYPH_REVIEW = "?"
GLYPH_EXTRA = "✗"
GLYPH_PINNED = "•"
GLYPH_DUPLICATE = "="
GLYPH_WAITING = "…"

LEVEL_STYLE = {"high": "green", "medium": "yellow", "low": "red"}
INERT_STYLE = "grey50"

# analysis states of a file while its season reads (§12.2 "Analysis shows progress")
QUEUED, READING, READY, UNREADABLE, NOT_READY = (
    "queued", "reading…", "ready", "unreadable", "not ready")


@dataclass
class Row:
    path: Path
    disc: str | None
    glyph: str
    style: str
    text: str  # the label after the glyph
    in_queue: bool = False  # needs review and can be acted on (n/N, the default filter)
    visible: bool = True  # shown in the default "needs review" view
    actionable: bool = False  # a/b/c, e, x, u, l make sense on it
    delta: float = math.inf  # confidence margin (the least confident file has the smallest)
    report: FileReport | None = None
    inert_reason: str | None = None  # why nothing can be done with it


def _short(path: Path) -> str:
    return path.stem


def _mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}m{s % 60:02d}s"


def _ep_label(fr: FileReport, season: int | None) -> str:
    ep = fr.decision.episode
    if fr.decision.kind == "extra" or ep is None:
        return "extra" + (f" ({_mmss(fr.duration_s)})" if fr.duration_s else "")
    if ep.season == season:
        return f"E{ep.episode:02d} {ep.title}"
    return f"S{ep.season:02d}E{ep.episode:02d} {ep.title}"


def row_for(fr: FileReport, pre_pin_level: dict[Path, str], primary_pinned: set[Path],
            season: int | None = None) -> Row:
    """One file of a solved season."""
    path = Path(fr.path)
    flags = set(fr.flags)
    level = fr.confidence.level
    delta = math.inf if fr.confidence.delta is None else fr.confidence.delta
    odds = "" if level == "high" else f"  {fr.confidence.odds}"
    text = f"{_short(path)}  {_ep_label(fr, season)}{odds}"
    if "named" in flags or "confirmed" in flags:
        why = "already named as an episode" if "named" in flags else "confirmed in history"
        return Row(path, fr.disc, GLYPH_HIGH, INERT_STYLE, f"{text}  ({why})", visible=False,
                   report=fr, inert_reason=f"{path.name} is {why}: nothing to do here")
    if "not_ready" in flags or "unreadable" in flags:
        why = NOT_READY if "not_ready" in flags else UNREADABLE
        return Row(path, fr.disc, GLYPH_WAITING, INERT_STYLE, f"{_short(path)}  ({why})",
                   report=fr, inert_reason=f"{path.name} is {why}: it has no evidence")
    if fr.duplicate_of is not None:
        prim = Path(fr.duplicate_of)
        pinned = prim in primary_pinned
        return Row(path, fr.disc, GLYPH_DUPLICATE, LEVEL_STYLE[level],
                   f"{text}  (copy of {prim.name})", visible=level != "high" or pinned,
                   actionable=True, delta=delta, report=fr)
    if "pinned" in flags:
        style = LEVEL_STYLE.get(pre_pin_level.get(path, "high"), "green")
        return Row(path, fr.disc, GLYPH_PINNED, style, text, visible=True, actionable=True,
                   delta=delta, report=fr)
    if level == "high":
        glyph = GLYPH_EXTRA if fr.decision.kind == "extra" else GLYPH_HIGH
        return Row(path, fr.disc, glyph, LEVEL_STYLE[level], text, visible=False,
                   actionable=True, delta=delta, report=fr)
    return Row(path, fr.disc, GLYPH_REVIEW, LEVEL_STYLE[level], text, in_queue=True,
               visible=True, actionable=True, delta=delta, report=fr)


def build_rows(report: Report, pre_pin_level: dict[Path, str], pinned: set[Path]) -> list[Row]:
    """Rows of a solved season, in the report's order (grouped by disc)."""
    return [row_for(fr, pre_pin_level, pinned, report.season) for fr in report.files]


def progress_rows(season_node, states: dict[Path, str], waiting: bool) -> list[Row]:
    """Rows of a season that is queued or reading: each file's analysis state."""
    rows: list[Row] = []
    groups = [(d, disc_label(d)) for d in season_node.discs] + [(None, None)]
    for d, label in groups:
        files = d.files if d is not None else season_node.loose
        for f in files:
            p = Path(f.path)
            st = states.get(p, NOT_READY if not f.ready else (QUEUED if waiting else READING))
            glyph = {READY: "·", UNREADABLE: "!"}.get(st, GLYPH_WAITING)
            rows.append(Row(p, label, glyph, INERT_STYLE if st != READY else "",
                            f"{_short(p)}  ({st})",
                            inert_reason="this season is still being analyzed"))
    return rows


def least_confident(rows: list[Row]) -> Row | None:
    queue = [r for r in rows if r.in_queue]
    return min(queue, key=lambda r: r.delta) if queue else None


class SeasonTree(OptionList):
    """The file list. Option ids are file paths; disc headers are disabled options."""

    BINDINGS = [
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.rows: list[Row] = []
        self.by_path: dict[str, Row] = {}

    def show_rows(self, rows: list[Row], show_all: bool, cursor: Path | None,
                  ignored: list[Path] | None = None) -> None:
        shown = [r for r in rows if show_all or r.visible]
        self.rows = shown
        self.by_path = {str(r.path): r for r in shown}
        opts: list[Option] = []
        disc: object = object()
        for r in shown:
            if r.disc != disc:
                disc = r.disc
                opts.append(Option(Text(r.disc or "(no disc)", style="bold"), disabled=True))
            label = Text("  ")
            label.append(r.glyph, style=r.style or None)
            label.append(" " + r.text, style=INERT_STYLE if r.inert_reason else None)
            opts.append(Option(label, id=str(r.path)))
        if show_all:
            for folder in ignored or []:
                opts.append(Option(Text(f"{folder.name}  (ignored folder)", style=INERT_STYLE),
                                   disabled=True))
        if not shown and not (show_all and ignored):
            opts.append(Option(Text("Nothing needs review here (f shows all files)",
                                    style=INERT_STYLE), disabled=True))
        self.set_options(opts)
        self.move_to(cursor)

    def move_to(self, path: Path | None) -> None:
        target = None
        if path is not None and str(path) in self.by_path:
            target = self.get_option_index(str(path))
        elif self.rows:
            target = self.get_option_index(str(self.rows[0].path))
        self.highlighted = target

    @property
    def current(self) -> Row | None:
        if self.highlighted is None:
            return None
        opt = self.get_option_at_index(self.highlighted)
        return self.by_path.get(opt.id) if opt.id is not None else None
