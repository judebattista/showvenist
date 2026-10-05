"""Data-quality report (spec §6.7)."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from showvenist.bundle.schema import Bundle, Episode

_GENERIC_ROLE = re.compile(
    r"#\s*\d+|\b(cop|waitress|waiter|bartender|man|woman|guard|nurse|doctor|"
    r"officer|driver|clerk|reporter|student|girl|boy|extra|passenger|customer|"
    r"voice|bystander|woman|policeman|policewoman)\b",
    re.I,
)


UNORDERED = -1  # season key for episodes missing from the aired ordering


def season_label(season: int) -> str:
    return "Unordered episodes" if season == UNORDERED else (
        "Specials" if season == 0 else f"Season {season}")


@dataclass
class SeasonQuality:
    season: int
    episodes: int = 0
    synopsis_coverage: float = 0.0
    median_synopsis_words: float = 0.0
    guest_cast_coverage: float = 0.0
    crew_coverage: float = 0.0
    subtitle_coverage: float = 0.0
    runtime_coverage: float = 0.0
    runtime_placeholder: bool = False  # all runtimes identical


@dataclass
class QualityReport:
    overall: SeasonQuality
    seasons: list[SeasonQuality] = field(default_factory=list)
    duplicate_synopses: list[tuple[str, str]] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    conflicts: list[dict[str, str]] = field(default_factory=list)
    unordered: list[str] = field(default_factory=list)  # episodes not in the aired ordering


def _words(ep: Episode) -> int:
    return max((len(s.text.split()) for s in ep.synopses if s.text.strip()), default=0)


def _has_guest(ep: Episode) -> bool:
    return any(c.character.strip() and not _GENERIC_ROLE.search(c.character) for c in ep.guest_cast)


def _sub_ok(ep: Episode, bundle_dir: Path | None) -> bool:
    if bundle_dir is None:
        return bool(ep.subtitles)
    return any((bundle_dir / s.path).is_file() for s in ep.subtitles)


def _measure(label: int, eps: list[Episode], bundle_dir: Path | None) -> SeasonQuality:
    n = len(eps)
    q = SeasonQuality(season=label, episodes=n)
    if not n:
        return q
    wc = [_words(e) for e in eps]
    withsyn = [w for w in wc if w > 0]
    q.synopsis_coverage = len(withsyn) / n
    q.median_synopsis_words = float(statistics.median(withsyn)) if withsyn else 0.0
    q.guest_cast_coverage = sum(_has_guest(e) for e in eps) / n
    q.crew_coverage = sum(bool(e.directors or e.writers) for e in eps) / n
    q.subtitle_coverage = sum(_sub_ok(e, bundle_dir) for e in eps) / n
    rts = [e.runtime_min for e in eps if e.runtime_min]
    q.runtime_coverage = len(rts) / n
    q.runtime_placeholder = len(rts) > 1 and len(set(rts)) == 1
    return q


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _dupes(eps: list[Episode]) -> list[tuple[str, str]]:
    toks = [(e.id, _tokens(e.synopses[0].text) if e.synopses else set()) for e in eps]
    toks = [(i, t) for i, t in toks if len(t) >= 5]
    out = []
    for a in range(len(toks)):
        for b in range(a + 1, len(toks)):
            ta, tb = toks[a][1], toks[b][1]
            if len(ta & tb) / len(ta | tb) >= 0.85:
                out.append((toks[a][0], toks[b][0]))
    return out


def quality_report(bundle: Bundle, bundle_dir: Path | None = None) -> QualityReport:
    byid = bundle.episode_by_id()
    ordering = bundle.orderings.get("aired")
    by_season: dict[int, list[Episode]] = {}
    seen: set[str] = set()
    if ordering:
        for ent in ordering.entries:
            ep = byid.get(ent.episode_id)
            if ep is not None and not ep.excluded:
                by_season.setdefault(ent.season, []).append(ep)
                seen.add(ep.id)
    active = [e for e in bundle.episodes if not e.excluded]
    unordered = [e.id for e in active if e.id not in seen]
    if unordered:
        by_season.setdefault(UNORDERED, []).extend(byid[i] for i in unordered)
    order = sorted(by_season.items(), key=lambda kv: (kv[0] < 0, kv[0]))
    seasons = [_measure(s, eps, bundle_dir) for s, eps in order]
    return QualityReport(
        overall=_measure(-1, active, bundle_dir),
        seasons=seasons,
        duplicate_synopses=_dupes(active),
        excluded=[e.id for e in bundle.episodes if e.excluded],
        conflicts=[p.conflict for p in bundle.provenance if p.conflict],
        unordered=unordered,
    )


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def _prediction(q: SeasonQuality, name: str) -> str | None:
    problems = []
    if q.synopsis_coverage < 0.7:
        problems.append(f"synopsis coverage {_pct(q.synopsis_coverage)}")
    if q.subtitle_coverage == 0:
        problems.append("no reference subtitles")
    elif q.subtitle_coverage < 0.5:
        problems.append(f"reference subtitles for only {_pct(q.subtitle_coverage)}")
    if not problems:
        return None
    msg = f"{name}: {', '.join(problems)}. Expect many files to need review."
    hints = []
    if q.subtitle_coverage < 0.5:
        hints.append("`--subs`")
    if q.synopsis_coverage < 0.7:
        hints.append("enabling the LLM scorer")
    return msg + " Consider " + " or ".join(hints) + "."


def render(report: QualityReport) -> str:
    lines = ["Bundle data quality"]

    def row(label: str, q: SeasonQuality) -> str:
        rt = (
            "placeholder runtimes"
            if q.runtime_placeholder
            else f"runtime {_pct(q.runtime_coverage)}"
        )
        return (
            f"{label}: {q.episodes} eps | synopsis {_pct(q.synopsis_coverage)} "
            f"(median {q.median_synopsis_words:.0f} words) | "
            f"guest cast {_pct(q.guest_cast_coverage)} "
            f"| crew {_pct(q.crew_coverage)} | ref subs {_pct(q.subtitle_coverage)} | {rt}"
        )

    for s in report.seasons:
        lines.append(row(season_label(s.season), s))
    lines.append(row("Overall", report.overall))
    if report.duplicate_synopses:
        lines.append("Duplicate/near-duplicate synopses:")
        lines += [f"  {a} ~ {b}" for a, b in report.duplicate_synopses]
    if report.excluded:
        lines.append("Excluded episodes: " + ", ".join(report.excluded))
    if report.conflicts:
        lines.append(f"Unresolved conflicts ({len(report.conflicts)}):")
        lines += ["  " + "; ".join(f"{k}={v}" for k, v in c.items()) for c in report.conflicts]
    preds = [
        p
        for s in report.seasons
        if (p := _prediction(s, season_label(s.season)))
    ]
    lines.append("")
    lines += preds or ["Data looks good: no major gaps expected."]
    return "\n".join(lines)
