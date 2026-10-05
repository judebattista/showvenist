"""Text rendering of eval results and the bridge from evidence rows to fit items."""

from __future__ import annotations

from collections.abc import Iterable

from showvenist.eval.fit import FitItem
from showvenist.eval.runner import EvidenceRow
from showvenist.eval.types import EvalResult, MetricSet


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def _line(name: str, m: MetricSet) -> str:
    p = m.precision_high
    prec = "   n/a" if p is None else f"{p.errors}/{p.n} (≤{100 * p.upper_95:.1f}%)"
    return (
        f"{name:<30} {m.files:>5} {_pct(m.top1)} {_pct(m.coverage_high)}  {prec:<18} "
        f"{_pct(m.unmatched_precision)} {_pct(m.unmatched_recall)}"
    )


def format_eval(result: EvalResult) -> str:
    head = (
        f"{'':<30} {'files':>5} {'top-1':>6} {'cov@hi':>6}  {'errors@high (95% bound)':<18} "
        f"{'unm.P':>6} {'unm.R':>6}"
    )
    lines = [head, _line("OVERALL", result.overall)]
    if result.per_show:
        lines += ["", "per show"] + [_line("  " + k, v) for k, v in sorted(result.per_show.items())]
    if result.per_season:
        lines += ["", "per season"] + [
            _line("  " + k, v) for k, v in sorted(result.per_season.items())
        ]
    o = result.overall
    if o.episode_unknown_files:
        lines += [
            "",
            f"episode? files: {o.episode_unknown_files}, decided as an episode at high: "
            f"{o.episode_unknown_high_as_episode}, flagged for review: "
            f"{o.episode_unknown_flagged}, provable errors: {o.episode_unknown_provable_errors}",
        ]
    if o.reliability:
        lines += ["", "reliability (correct share): " + ", ".join(
            f"{k} {100 * v:.1f}%" for k, v in o.reliability.items())]
    if result.problems:
        lines += ["", f"{len(result.problems)} labels/files left out:"] + [
            f"  {p.row.path}: {p.reason}" for p in result.problems[:10]
        ]
        if len(result.problems) > 10:
            lines.append(f"  … and {len(result.problems) - 10} more")
    return "\n".join(lines)


def fit_items(rows: Iterable[EvidenceRow], season_key: str) -> list[FitItem]:
    """Labeled evidence rows -> FitItems (decision 41: `episode?` is a set label)."""
    out: list[FitItem] = []
    for r in rows:
        llr = {k: v for k, v in r.llr.items() if r.available.get(k, False)}
        common = dict(
            season_key=season_key, regime=r.regime, llr=llr, n_candidates=len(r.candidates)
        )
        if r.label_kind == "episode" and r.target is not None and r.target >= 0:
            out.append(FitItem(target="episode", target_index=r.target, **common))
        elif r.label_kind == "extra":
            out.append(FitItem(target="extra", **common))
        elif r.label_kind == "episode?":
            out.append(FitItem(target="any_episode", **common))
    return out
