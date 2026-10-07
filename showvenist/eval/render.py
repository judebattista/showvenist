"""Text rendering of eval results and the bridge from evidence rows to fit items."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from showvenist.assign.basic import assign_basic
from showvenist.assign.confidence import confidence
from showvenist.config import WeightsConfig
from showvenist.eval.fit import FitItem
from showvenist.eval.harness import LEVELS, clopper_pearson_upper
from showvenist.eval.runner import EvidenceRow, GroupRun
from showvenist.eval.types import EvalResult, MetricSet, RateWithBound
from showvenist.fusion import fuse
from showvenist.scoring.types import DEFAULT_CLAMPS, PI_X_DEFAULT, REGIME_SCORERS, ScoreResult


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
    if o.extra_multi_files:
        lines += [
            "",
            f"extra-multi files (several episodes on one track): {o.extra_multi_files}, "
            f"decided extra: {o.extra_multi_as_extra}, taken for an episode at high confidence: "
            f"{o.extra_multi_as_episode_high}",
        ]
    if o.reliability:
        lines += ["", "reliability (correct share): " + ", ".join(
            f"{k} {100 * v:.1f}%" for k, v in o.reliability.items())]
    if o.cost_usd:
        lines += ["", f"LLM cost: ${o.cost_usd:.4f} (new API spend; cached responses are free)"]
    if result.problems:
        lines += ["", f"{len(result.problems)} problems (left out unless noted):"] + [
            f"  {p.row.path}: {p.reason}" for p in result.problems[:10]
        ]
        if len(result.problems) > 10:
            lines.append(f"  … and {len(result.problems) - 10} more")
    return "\n".join(lines)


NO_EXTRAS_NOTE = (
    "no labeled extras: disc file counts equal episode counts, so the block lift is an upper bound"
)


def eval_caveats(groups: Sequence[GroupRun], blocks: bool = True) -> list[str]:
    """Lines that qualify the numbers (decision 47's measurements had both leaks): files whose
    hard season filter came from their label, and, with blocks, a corpus with no labeled
    extras (every disc holds exactly its episodes, which flatters the block model)."""
    out: list[str] = []
    n = sum(g.season_from_label for g in groups)
    if n:
        out.append(f"{n} file{'s' if n != 1 else ''} took their season from the label "
                   "(no season folder)")
    if blocks and groups and not any(lab.kind == "extra" for g in groups for lab in g.labels):
        out.append(NO_EXTRAS_NOTE)
    return out


def _item(r: EvidenceRow, season_key: str, regime: str, names: Iterable[str]) -> FitItem | None:
    llr = {k: r.llr[k] for k in names if k in r.llr and r.available.get(k, False)}
    common = dict(season_key=season_key, regime=regime, llr=llr, n_candidates=len(r.candidates))
    if r.label_kind == "episode" and r.target_set:
        return FitItem(target="any_episode", target_set=tuple(r.target_set), **common)
    if r.label_kind == "episode" and r.target is not None and r.target >= 0:
        return FitItem(target="episode", target_index=r.target, **common)
    if r.label_kind == "extra":
        return FitItem(target="extra", **common)
    if r.label_kind == "episode?":
        tset = None if r.target_set is None else tuple(r.target_set)
        if tset is not None and not tset:
            return None  # every episode is held by a labeled file: nothing to learn from it
        return FitItem(target="any_episode", target_set=tset, **common)
    return None


def format_high_errors(labels, decisions, bundles) -> str:
    """Files whose decision was wrong at high confidence (the cases the 1% bound is about)."""
    from showvenist.eval.harness import _is_correct

    by_path = {d.path: d for d in decisions}
    from showvenist.eval.harness import find_ordering

    ep_name = {}
    for b in bundles.values():
        for e in b.episodes:
            ep_name[e.id] = e.title

    def number(show: str, order: str, episode_id: str | None) -> str:
        b = bundles.get(show)
        ordering = find_ordering(b, order) if b is not None else None
        if ordering is None or episode_id is None:
            return ""
        for ent in ordering.entries:
            if ent.episode_id == episode_id:
                return f"S{ent.season:02d}E{ent.episode:02d} "
        return ""
    lines = []
    for lab in labels:
        d = by_path.get(lab.row.path)
        if d is None or d.level != "high" or lab.kind == "episode?":
            continue
        if lab.duplicate_set is not None or _is_correct(lab, d):
            continue
        want = lab.row.label if lab.kind == "episode" else "extra"
        got = (
            number(lab.row.show, lab.row.order, d.episode_id)
            + repr(ep_name.get(d.episode_id, d.episode_id))
            if d.episode_id else "extra"
        )
        lines.append(
            f"  {lab.row.path.rsplit('/', 1)[-1]}: labeled {want} ({lab.row.show}), "
            f"decided {got} at Δ={d.delta:.1f}" if d.delta is not None else
            f"  {lab.row.path.rsplit('/', 1)[-1]}: labeled {want}, decided {got}"
        )
    return "errors at high confidence:\n" + ("\n".join(lines) if lines else "  none")


def fit_items(
    rows: Iterable[EvidenceRow],
    season_key: str,
    llm_eligible: Sequence[bool] | None = None,
) -> list[FitItem]:
    """Labeled evidence rows -> FitItems for the two fits (§16.3).

    Every row gives a "base" item (base scorers only: base weights are fitted on all files,
    without the LLM). A row the LLM scored also gives an "llm" item when `llm_eligible[i]`
    (the files `--llm auto` would send; None = all of them). `episode?` rows are set labels
    over their `target_set` (decision 41).
    """
    out: list[FitItem] = []
    for i, r in enumerate(rows):
        it = _item(r, season_key, "base", REGIME_SCORERS["base"])
        if it is None:
            continue
        out.append(it)
        if r.regime == "llm" and (llm_eligible is None or llm_eligible[i]):
            llm_it = _item(r, season_key, "llm", REGIME_SCORERS["llm"])
            if llm_it is not None:
                out.append(llm_it)
    return out


def solve_items(rows: Iterable[EvidenceRow], season_key: str) -> list[FitItem]:
    """One item per labeled row in the regime it was actually solved in (for
    `heldout_metrics` over both regimes)."""
    out: list[FitItem] = []
    for r in rows:
        it = _item(r, season_key, r.regime, REGIME_SCORERS[r.regime])
        if it is not None:
            out.append(it)
    return out


# ---- metrics from stored evidence -----------------------------------------------------------


def _season_solve(
    items: Sequence[FitItem], weights: WeightsConfig, pi_x: float
) -> tuple[list[int], list[str]]:
    E = items[0].n_candidates
    if any(it.n_candidates != E for it in items):
        raise ValueError("items of one season must share the candidate set")
    F = len(items)
    names = {k for it in items for k in it.llr}
    results: dict[str, ScoreResult] = {}
    for k in sorted(names):
        res = ScoreResult.empty(F, E)
        for i, it in enumerate(items):
            if k in it.llr:
                res.llr[i] = np.asarray(it.llr[k], dtype=float)
                res.available[i] = True
        results[k] = res
    if results:
        fused = fuse(results, weights, DEFAULT_CLAMPS, pi_x)
    else:
        fused = fuse({}, weights, DEFAULT_CLAMPS, pi_x, history_penalty=np.zeros((F, E)))
    a = assign_basic(fused.S)
    return list(a.column), [c.level for c in confidence(fused.S, a)]


def heldout_metrics(
    items_by_season: Mapping[str, Sequence[FitItem]],
    weights: WeightsConfig | Mapping[str, WeightsConfig],
    pi_x: float = PI_X_DEFAULT,
) -> MetricSet:
    """Top-1, coverage and precision at high, re-solving each season's assignment (fusion +
    basic assignment + confidence) from the items' stored raw evidence.

    `weights` is one WeightsConfig for every season, or one per season key (the weights fitted
    with that season held out). An item's regime is that of its evidence (an "llm" key makes
    it an LLM-regime file). `episode?` items are scored as in decision 41: only an `extra` at
    high, or a high decision outside their `target_set`, counts (as an error).
    """
    known: list[tuple[bool, str]] = []  # (correct, level)
    proven = 0
    n_unknown = 0
    unm: list[tuple[bool, bool]] = []  # (labeled extra, decided extra); `episode?` = not extra
    for key, items in items_by_season.items():
        if not items:
            continue
        w = weights if isinstance(weights, WeightsConfig) else weights[key]
        cols, levels = _season_solve(items, w, pi_x)
        for it, col, level in zip(items, cols, levels, strict=True):
            unm.append((it.target == "extra", col < 0))
            if it.target == "any_episode":
                n_unknown += 1
                if level == "high" and (
                    col < 0 or (it.target_set is not None and col not in it.target_set)
                ):
                    proven += 1
                continue
            ok = col == it.target_index if it.target == "episode" else col < 0
            known.append((ok, level))
    ms = MetricSet(files=len(known) + n_unknown, episode_unknown_files=n_unknown,
                   episode_unknown_provable_errors=proven)
    if known:
        ms.top1 = sum(ok for ok, _ in known) / len(known)
        ms.coverage_high = sum(lv == "high" for _, lv in known) / len(known)
        for level in LEVELS:
            at = [ok for ok, lv in known if lv == level]
            if at:
                ms.reliability[level] = sum(at) / len(at)
    tp = sum(a and d for a, d in unm)
    predicted, actual = sum(d for _, d in unm), sum(a for a, _ in unm)
    ms.unmatched_precision = tp / predicted if predicted else None
    ms.unmatched_recall = tp / actual if actual else None
    high = [ok for ok, lv in known if lv == "high"]
    n = len(high) + proven
    errors = sum(not ok for ok in high) + proven
    if n:
        ms.precision_high = RateWithBound(errors, n, clopper_pearson_upper(errors, n))
    return ms


def format_heldout(title: str, before: MetricSet, after: MetricSet) -> str:
    head = (
        f"{'':<30} {'files':>5} {'top-1':>6} {'cov@hi':>6}  {'errors@high (95% bound)':<18} "
        f"{'unm.P':>6} {'unm.R':>6}"
    )
    return "\n".join([title, head, _line("default weights", before),
                      _line("fitted (season held out)", after)])


def format_ablation(rows: Sequence[tuple[str, MetricSet]]) -> str:
    """Per-signal ablation (§16.1): the metrics with each scorer removed in turn."""
    head = (
        f"{'ablation':<30} {'files':>5} {'top-1':>6} {'cov@hi':>6}  "
        f"{'errors@high (95% bound)':<18} {'unm.P':>6} {'unm.R':>6}"
    )
    return "\n".join([head] + [_line(name, m) for name, m in rows])


def format_times(groups: Sequence[GroupRun]) -> str:
    """Time per season (wall clock), split into analysis (I/O + OCR), scoring and solve, and
    the LLM spend when it ran."""
    lines = [
        "time per season",
        f"{'season':<30} {'files':>5} {'analyze':>8} {'score':>7} {'solve':>7} {'total':>8} "
        f"{'LLM $':>8}",
    ]
    tot = {"analyze": 0.0, "score": 0.0, "solve": 0.0, "total": 0.0}
    n = 0
    usd = 0.0
    for g in groups:
        s = g.seconds
        f = len(g.solution.files) + len(g.session.duplicates)
        lines.append(
            f"{g.key:<30} {f:>5} {s.get('analyze', 0):>7.1f}s {s.get('score', 0):>6.1f}s "
            f"{s.get('solve', 0):>6.1f}s {s.get('total', 0):>7.1f}s {g.llm_cost_usd:>8.4f}"
        )
        for k in tot:
            tot[k] += s.get(k, 0.0)
        n += f
        usd += g.llm_cost_usd
    lines.append(
        f"{'total':<30} {n:>5} {tot['analyze']:>7.1f}s {tot['score']:>6.1f}s "
        f"{tot['solve']:>6.1f}s {tot['total']:>7.1f}s {usd:>8.4f}"
    )
    return "\n".join(lines)


def _d(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "   n/a"
    return f"{100 * (a - b):+5.1f}"


def _errs(m: MetricSet) -> str:
    p = m.precision_high
    return "n/a" if p is None else f"{p.errors}/{p.n}"


def format_compare_blocks(
    with_blocks: EvalResult,
    without: EvalResult,
    discs: Mapping[str, int] | None = None,
) -> str:
    """Blocks vs. no blocks (§16.1, M4 exit): both metric blocks, then per season the two
    runs side by side with the delta (blocks − no blocks, percentage points). `discs` gives the
    number of discs per season key; seasons with fewer than two discs are marked, since the
    block model can only change a multi-disc season."""
    discs = discs or {}
    out = ["with disc blocks", format_eval(with_blocks), "", "without blocks (--no-blocks)",
           format_eval(without), "", "blocks vs. no blocks per season (Δ = blocks − no blocks)"]
    out.append(
        f"{'season':<30} {'discs':>5} {'top-1':>13} {'Δ':>6} {'cov@hi':>13} {'Δ':>6} "
        f"{'errors@high':>15}"
    )
    keys = sorted(set(with_blocks.per_season) | set(without.per_season))
    rows = [("OVERALL", with_blocks.overall, without.overall, None)]
    rows += [(k, with_blocks.per_season.get(k, MetricSet()), without.per_season.get(k, MetricSet()),
              discs.get(k)) for k in keys]
    for name, a, b, n in rows:
        nd = "" if n is None else str(n) + ("" if n >= 2 else "*")
        out.append(
            f"{name:<30} {nd:>5} {_pct(a.top1)} {_pct(b.top1)} {_d(a.top1, b.top1):>6} "
            f"{_pct(a.coverage_high)} {_pct(b.coverage_high)} "
            f"{_d(a.coverage_high, b.coverage_high):>6} {_errs(a):>7} {_errs(b):>7}"
        )
    if any(n < 2 for n in discs.values()):
        out.append("* fewer than two discs: the block model cannot change this season much")
    return "\n".join(out)
