"""Right pane: the evidence for the file under the cursor (spec §12.2, §12.4).

Everything shown comes from `Session.explain` (top candidates, per-signal contributions, notes)
and the season's `Report` row (decision, confidence, flags); no scoring here.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from showvenist.report_schema import FileDetail, FileReport
from showvenist.tui.tree import Row, _mmss

LETTERS = "abc"

FLAG_TEXT = {
    "pinned": "pinned by you",
    "possible_two_parter": "possible two-parter",
    "play_all": "play-all",
    "history_duplicate": "history: this episode is already confirmed for another file",
    "out_of_scope": "evidence points to another season",
    "no_dialogue": "no usable subtitles: dialogue signals unavailable",
    "disc_window_failed": "the LLM found no fit in the disc window",
    "confirmed": "confirmed in history",
    "named": "already named as an episode",
    "in_extras": "already in extras/",
    "not_ready": "still being written",
    "unreadable": "unreadable",
    "near_duplicate": "near duplicate (same dialogue, different bytes)",
    "duplicate_allowed": "kept as another version (--allow-duplicates)",
}


def flags_banner(fr: FileReport) -> list[str]:
    out = []
    if fr.duplicate_of is not None:
        out.append(f"duplicate of {Path(fr.duplicate_of).name}")
    for f in fr.flags:
        if f == "duplicate":
            if fr.duplicate_of is None:
                out.append("duplicate")
            continue
        out.append(FLAG_TEXT.get(f, f))
    return out


def _num(v: float) -> str:
    if abs(v) < 0.05:
        return "0"
    return f"{v:+.1f}".replace("-", "−")


def render_detail(fr: FileReport | None, detail: FileDetail | None, path: Path,
                  message: str | None = None, notices: list[str] | None = None
                  ) -> RenderableType:
    """The pane's content. `detail` is None for files without evidence (`message` says why)."""
    parts: list[RenderableType] = []
    dur = fr.duration_s if fr is not None else (detail.duration_s if detail else 0.0)
    head = Text(f"{path.parent.name} / {path.name}", style="bold")
    if dur:
        head.append(f"   {_mmss(dur)}")
    parts.append(head)
    if fr is not None:
        ep = fr.decision.episode
        chosen = ("extra" if fr.decision.kind == "extra" or ep is None
                  else f"{ep.code} {ep.title}")
        parts.append(Text(f"Chosen: {chosen}  ({fr.confidence.level}, {fr.confidence.odds})"))
        banner = flags_banner(fr)
        if banner:
            parts.append(Text("⚑ " + " · ".join(banner), style="bold yellow"))
    if message:
        parts.append(Text(message, style="italic"))
    if detail is not None and detail.candidates:
        cands = detail.candidates[: len(LETTERS)]
        lines = Text()
        for k, c in enumerate(cands):
            mark = "●" if k == 0 else " "
            odds = "" if k == 0 else f"  {c.odds}"
            lines.append(f"{mark} {LETTERS[k]}) {c.label} {c.title}{odds}\n")
        parts.append(lines)
        t = Table(box=None, pad_edge=False, show_edge=False, header_style="bold")
        t.add_column("Evidence")
        for k in range(len(cands)):
            t.add_column(LETTERS[k], justify="right")
        names = list(dict.fromkeys(s.name for c in cands for s in c.signals))
        for name in names:
            cells = []
            for c in cands:
                s = next((s for s in c.signals if s.name == name), None)
                cells.append("–" if s is None or not s.available else _num(s.winner))
            t.add_row(name, *cells)
        parts.append(t)
        rt = [f"{LETTERS[k]}) {c.runtime_min}m" for k, c in enumerate(cands)
              if c.runtime_min]
        if rt:
            parts.append(Text(f"Duration {_mmss(detail.duration_s)} against runtime "
                              + ", ".join(rt)))
        for k, c in enumerate(cands):
            if c.notes:
                parts.append(Text(f"Notes ({LETTERS[k]}): " + " · ".join(c.notes[:6])))
        if detail.title_order:
            parts.append(Text(detail.title_order[:1].upper() + detail.title_order[1:] + "."))
        for k, c in enumerate(cands):
            if c.synopsis:
                parts.append(Text(f"Synopsis ({LETTERS[k]}): {c.synopsis}", style="dim"
                                  if k else ""))
    if notices:
        parts.append(Text("\nNotices", style="bold"))
        parts += [Text(f"- {n}", style="dim") for n in notices]
    return Group(*parts)


class EvidencePane(VerticalScroll):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._shown: RenderableType = Text("")

    def compose(self):
        yield Static(id="ev-body")

    def show(self, renderable: RenderableType) -> None:
        self._shown = renderable
        self.query_one("#ev-body", Static).update(renderable)

    @property
    def text(self) -> str:
        """Plain text of what is shown (for tests and debugging)."""
        import io

        from rich.console import Console

        buf = io.StringIO()
        Console(file=buf, width=120, color_system=None).print(self._shown)
        return buf.getvalue()


__all__ = ["EvidencePane", "Row", "flags_banner", "render_detail"]
