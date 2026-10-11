"""Report mode (spec §12.3, §4.5): build the Report from solver output, render it, exit code.

Pure presentation: nothing here touches media files.
"""

from __future__ import annotations

import io
import math
from importlib import metadata

from rich.console import Console
from rich.table import Table

from showvenist.report_schema import (
    Confidence,
    Decision,
    EpisodeRef,
    FileReport,
    Report,
    SignalContribution,
    Summary,
)
from showvenist.scoring.types import (
    REGIME_SCORERS,
    Assignment,
    Candidate,
    FileConfidence,
    FileInfo,
    Fused,
)

CONFIRMED_FLAGS = {"already_confirmed", "already confirmed", "confirmed"}
NOT_READY_FLAG = "not_ready"
ODDS_CAP = 999


def format_odds(delta: float | None) -> str:
    """Odds string from e^delta: whole numbers from 10 up, one decimal below, capped at >999:1."""
    if delta is None or math.isinf(delta) or delta >= math.log(ODDS_CAP + 0.5):
        return f">{ODDS_CAP}:1"
    odds = math.exp(max(delta, 0.0))
    if odds >= 10:
        return f"{round(odds)}:1"
    return f"{odds:.1f}:1"


def _version() -> str:
    try:
        return metadata.version("showvenist")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _decision(col: int, candidates: list[Candidate]) -> Decision:
    if col < 0:
        return Decision(kind="extra")
    c = candidates[col]
    return Decision(
        kind="episode",
        episode=EpisodeRef(
            episode_id=c.id, season=c.season, episode=c.number, title=c.episode.title
        ),
    )


def _signals(
    i: int, regime: str, fused: Fused, win: int, runner: int | None, runner_known: bool
) -> list[SignalContribution]:
    """Per-signal contributions for the winner and runner-up (extras contribute 0.0).

    Only scorers summed in the file's regime are listed. `available` is the scorer's own
    availability for the file (`ScoreResult.available`, carried in `fused.available`): an
    available scorer that contributes exactly 0 is still available. Without that bit (a
    hand-built Fused) it falls back to "non-zero for the winner or the runner-up".
    """
    out = []
    for name in REGIME_SCORERS[regime]:  # type: ignore[index]
        arr = fused.contributions.get(name)
        if arr is None:
            continue
        w = float(arr[i, win]) if win >= 0 else 0.0
        r = None
        if runner_known:
            r = float(arr[i, runner]) if runner is not None and runner >= 0 else 0.0
        av = fused.available.get(name)
        out.append(
            SignalContribution(
                name=name,
                available=bool(av[i]) if av is not None else bool(w != 0.0 or (r or 0.0) != 0.0),
                winner=w,
                runner_up=r,
            )
        )
    return out


def note_lines(name: str, note: dict) -> list[str]:
    """Readable lines for one scorer's `ScoreResult.notes` entry (one file, one candidate)."""
    out: list[str] = []
    if name == "refsubs":
        if "containment" in note:
            out.append(f"refsubs: {float(note['containment']):.0%} of the reference matched")
        out += [f"refsubs line: {s}" for s in note.get("top_shingles", [])]
    elif name == "synopsis" and note.get("terms"):
        out.append("synopsis terms: " + ", ".join(map(str, note["terms"])))
    elif name == "guests" and note.get("names"):
        out.append("guests heard: " + ", ".join(map(str, note["names"])))
    elif name == "titlecard":
        out += [f"title card read: {r}" for r in note.get("readings", [])]
    elif name == "llm":
        if "p" in note:
            out.append(f"llm: p={float(note['p']):.2f}")
        out += [f"llm quote: {q}" for q in note.get("quotes", [])]
        if note.get("quote_failed"):
            out.append("llm: a quote failed verification")
    return out


def build_report(
    files: list[FileInfo],
    candidates: list[Candidate],
    fused: Fused,
    assignment: Assignment,
    confidences: list[FileConfidence],
    *,
    show_slug: str,
    bundle_revision: str,
    ordering: str,
    season: int | None = None,
    flags: dict[int, list[str]] | None = None,
    tool_version: str | None = None,
) -> Report:
    """Assemble the Report. `flags` is keyed by file index; files are sorted by (disc, path)."""
    flags = flags or {}
    rows: list[FileReport] = []
    for i, f in enumerate(files):
        col = assignment.column[i]
        conf = confidences[i]
        infinite = conf.delta is None or math.isinf(conf.delta)
        has_runner = conf.runner_up is not None
        rows.append(
            FileReport(
                path=str(f.path),
                fingerprint=f.fingerprint,
                disc=f.disc,
                duration_s=f.duration_s,
                regime=fused.regimes[i],
                decision=_decision(col, candidates),
                confidence=Confidence(
                    level=conf.level,
                    delta=None if infinite else float(conf.delta),
                    odds=format_odds(None if infinite else conf.delta),
                ),
                runner_up=_decision(conf.runner_up, candidates) if has_runner else None,
                signals=_signals(i, fused.regimes[i], fused, col, conf.runner_up, has_runner),
                flags=list(flags.get(i, [])),
            )
        )
    rows.sort(key=lambda r: (r.disc is None, r.disc or "", r.path))
    summary = Summary(
        files=len(rows),
        high=sum(r.confidence.level == "high" for r in rows),
        medium=sum(r.confidence.level == "medium" for r in rows),
        low=sum(r.confidence.level == "low" for r in rows),
        extras=sum(r.decision.kind == "extra" for r in rows),
    )
    return Report(
        tool_version=tool_version or _version(),
        show_slug=show_slug,
        bundle_revision=bundle_revision,
        ordering=ordering,
        season=season,
        files=rows,
        summary=summary,
    )


def exit_code(report: Report) -> int:
    """0 if every file is high confidence (or already confirmed); 2 if any needs review."""
    for r in report.files:
        fl = set(r.flags)
        if NOT_READY_FLAG in fl:
            return 2
        if r.confidence.level != "high" and not (fl & CONFIRMED_FLAGS):
            return 2
    return 0


def to_json(report: Report) -> str:
    """Stable JSON: schema field order (report_version first), indent 2."""
    return report.model_dump_json(indent=2)


def _label(d: Decision) -> str:
    if d.kind == "extra" or d.episode is None:
        return "extra"
    return f"S{d.episode.season:02d}E{d.episode.episode:02d} {d.episode.title}"


def _top_signals(r: FileReport, n: int = 3) -> str:
    if r.runner_up is None:
        sigs = sorted(r.signals, key=lambda s: -abs(s.winner))
        return ", ".join(f"{s.name} {s.winner:+.1f}" for s in sigs[:n] if s.available)
    sigs = sorted(r.signals, key=lambda s: -abs(s.winner - (s.runner_up or 0.0)))
    return ", ".join(
        f"{s.name} {s.winner:+.1f} vs {(s.runner_up or 0.0):+.1f}"
        for s in sigs[:n]
        if s.available
    )


def render_tables(report: Report) -> str:
    """One rich table per season (extras in their own table), rows grouped by disc."""
    groups: dict[int | None, list[FileReport]] = {}
    for r in report.files:
        key = r.decision.episode.season if r.decision.episode else None
        groups.setdefault(key, []).append(r)
    buf = io.StringIO()
    console = Console(file=buf, width=140, force_terminal=False, color_system=None)
    for key in sorted(groups, key=lambda k: (k is None, k or 0)):
        title = "Extras" if key is None else f"Season {key}"
        table = Table(title=title, title_justify="left")
        for col in ("Disc", "File", "Chosen", "Odds", "Signals (vs runner-up)", "Flags"):
            table.add_column(col, overflow="fold")
        for r in groups[key]:
            table.add_row(
                r.disc or "-",
                r.path.rsplit("/", 1)[-1],
                _label(r.decision),
                f"{r.confidence.odds} ({r.confidence.level})",
                _top_signals(r),
                ", ".join(r.flags),
            )
        console.print(table)
    s = report.summary
    console.print(
        f"{s.files} files: {s.high} high, {s.medium} medium, {s.low} low; {s.extras} extras"
    )
    for note in report.notices:
        console.print(note)
    return buf.getvalue()


__all__ = ["build_report", "render_tables", "exit_code", "to_json", "format_odds", "note_lines"]
