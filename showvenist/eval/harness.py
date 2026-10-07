"""Evaluation harness (spec §16.1, decisions 24, 25, 41).

Pure functions: labels are resolved against bundles, then compared with hand-built or
pipeline-produced `FileDecision`s. The harness never runs the pipeline itself.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePath

from scipy.stats import beta

from showvenist.bundle.schema import Bundle, Ordering
from showvenist.eval.manifest import read_manifest  # noqa: F401  (re-exported, module map)
from showvenist.eval.types import (
    EvalResult,
    FileDecision,
    ManifestRow,
    MetricSet,
    Problem,
    RateWithBound,
    ResolvedLabel,
)

CONFIDENCE = 0.95
LEVELS = ("high", "medium", "low")
_EPISODE_RE = re.compile(r"^s(\d{1,3})e(\d{1,4})$", re.IGNORECASE)
_SEASON_DIR_RE = re.compile(r"^season[\s._-]*(\d{1,3})$", re.IGNORECASE)


# ---- label resolution ---------------------------------------------------------------------


def find_ordering(bundle: Bundle, name: str) -> Ordering | None:
    """The ordering a manifest's `order` names: its key (any case), else a unique type match."""
    if name in bundle.orderings:
        return bundle.orderings[name]
    by_key = [o for k, o in bundle.orderings.items() if k.lower() == name.lower()]
    if len(by_key) == 1:
        return by_key[0]
    by_type = [o for o in bundle.orderings.values() if o.type.lower() == name.lower()]
    return by_type[0] if len(by_type) == 1 else None


def _resolve_episode(row: ManifestRow, bundle: Bundle) -> tuple[str | None, int | None, str]:
    """(episode_id, season, problem reason); reason is "" on success."""
    m = _EPISODE_RE.match(row.label)
    if not m:
        return None, None, f"unparseable label {row.label!r}"
    season, number = int(m.group(1)), int(m.group(2))
    ordering = find_ordering(bundle, row.order)
    if ordering is None:
        return None, None, f"unknown ordering {row.order!r}"
    hits = {e.episode_id for e in ordering.entries if e.season == season and e.episode == number}
    if len(hits) != 1:
        what = "no entry" if not hits else f"{len(hits)} episodes"
        return None, None, f"{row.label} has {what} in ordering {row.order!r}"
    (eid,) = hits
    if eid not in bundle.episode_by_id():
        return None, None, f"{row.label} points to unknown episode {eid}"
    return eid, season, ""


def _split_multi(label: str) -> list[str] | None:
    """'S01E01+S01E02' or 'S01E01+E02' -> ['S01E01', 'S01E02']; None for a single label."""
    parts = [x.strip() for x in label.split("+")]
    if len(parts) < 2:
        return None
    m = re.match(r"^(s\d{1,3})e\d{1,4}$", parts[0], re.IGNORECASE)
    out = [parts[0]]
    for x in parts[1:]:
        out.append(f"{m.group(1)}{x}" if m and re.match(r"^e\d{1,4}$", x, re.IGNORECASE) else x)
    return out


def _resolve_multi(row: ManifestRow, bundle: Bundle, parts: list[str]):
    ids: list[str] = []
    season: int | None = None
    for part in parts:
        eid, s, reason = _resolve_episode(
            ManifestRow(row.path, row.show, part, row.order, row.disc), bundle
        )
        if reason:
            return None, None, reason
        ids.append(eid)
        season = season if season is not None else s
    if len(set(ids)) != len(ids):
        return None, None, f"{row.label} names the same episode twice"
    return tuple(ids), season, ""


def resolve_labels(
    rows: Iterable[ManifestRow], bundles: Mapping[str, Bundle]
) -> tuple[list[ResolvedLabel], list[Problem]]:
    """Resolve each row's label to an episode ID through its ordering (§16.1).

    Unresolvable rows become Problems (reported, left out, never errors). Episode labels that
    share an episode within a show form a duplicate set (`duplicate_set` = the episode id).
    """
    resolved: list[ResolvedLabel] = []
    problems: list[Problem] = []
    for row in rows:
        bundle = bundles.get(row.show)
        if bundle is None:
            problems.append(Problem(row, f"no bundle for show {row.show!r}"))
            continue
        label = row.label.strip().lower()
        if label == "extra":
            resolved.append(ResolvedLabel(row, "extra"))
        elif label in ("extra-multi", "extra_multi", "extramulti"):
            resolved.append(ResolvedLabel(row, "extra", multi=True))
        elif label == "episode?":
            resolved.append(ResolvedLabel(row, "episode?"))
        elif (parts := _split_multi(row.label.strip())) is not None:
            ids, _, reason = _resolve_multi(row, bundle, parts)
            if reason:
                problems.append(Problem(row, reason))
            else:
                resolved.append(ResolvedLabel(row, "episode", episode_id=ids[0], episode_ids=ids))
        else:
            eid, _, reason = _resolve_episode(row, bundle)
            if reason:
                problems.append(Problem(row, reason))
            else:
                resolved.append(ResolvedLabel(row, "episode", episode_id=eid))

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in resolved:
        if r.kind == "episode" and not r.episode_ids:
            counts[(r.row.show, r.episode_id)] += 1
    out = [
        ResolvedLabel(r.row, r.kind, r.episode_id, duplicate_set=r.episode_id)
        if r.kind == "episode" and not r.episode_ids and counts[(r.row.show, r.episode_id)] > 1
        else r
        for r in resolved
    ]
    return out, problems


# ---- metrics ------------------------------------------------------------------------------


def clopper_pearson_upper(errors: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided Clopper-Pearson upper bound on an error rate. n = 0 or errors >= n -> 1.0."""
    if n <= 0 or errors >= n:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, n - errors))


@dataclass
class _Scored:
    label: ResolvedLabel
    decision: FileDecision
    correct: bool | None  # None for `episode?` (unverifiable unless provably wrong)
    provable_error: bool = False

    @property
    def high(self) -> bool:
        return self.decision.level == "high"


def _is_correct(label: ResolvedLabel, d: FileDecision) -> bool:
    if label.kind == "extra":
        return d.kind == "extra"
    if label.duplicate_set is not None:
        # Scored on the duplicate flag (§14): the solved copy names the episode, the others are
        # flagged as its duplicates.
        if d.duplicate:
            return d.episode_id in (None, label.episode_id)
    if label.episode_ids:
        return d.kind == "episode" and d.episode_id in label.episode_ids
    return d.kind == "episode" and d.episode_id == label.episode_id


def _provable_error(
    label: ResolvedLabel, d: FileDecision, held: Mapping[str, set[str]]
) -> bool:
    """Decision 41: an `episode?` file decided extra at high, or decided at high for an episode a
    labeled file already holds (unless the two are marked as a duplicate set)."""
    if d.level != "high":
        return False
    if d.kind == "extra":
        return True
    if d.episode_id not in held.get(label.row.show, set()):
        return False
    same_set = d.duplicate or label.duplicate_set == d.episode_id
    return not same_set


def _metric_set(items: list[_Scored]) -> MetricSet:
    ms = MetricSet(files=len(items))
    known = [s for s in items if s.label.kind != "episode?"]
    unknown = [s for s in items if s.label.kind == "episode?"]

    if known:
        ms.top1 = sum(s.correct for s in known) / len(known)
        ms.coverage_high = sum(s.high for s in known) / len(known)
        for level in LEVELS:
            at = [s for s in known if s.decision.level == level]
            if at:
                ms.reliability[level] = sum(s.correct for s in at) / len(at)

    counted = [s for s in known if s.high and s.label.duplicate_set is None]
    # decision 41: provably wrong `episode?` decisions are verifiable, so they count in both the
    # errors and n; only unverifiable decisions are left out
    proven = sum(s.provable_error for s in unknown)
    n = len(counted) + proven
    errors = sum(not s.correct for s in counted) + proven
    if n or errors:
        ms.precision_high = RateWithBound(errors, n, clopper_pearson_upper(errors, n))

    if items:
        predicted = [s for s in items if s.decision.kind == "extra"]
        actual = [s for s in items if s.label.kind == "extra"]
        tp = sum(s.label.kind == "extra" for s in predicted)
        ms.unmatched_precision = tp / len(predicted) if predicted else None
        ms.unmatched_recall = tp / len(actual) if actual else None

    multi = [s for s in items if s.label.multi]
    ms.extra_multi_files = len(multi)
    ms.extra_multi_as_extra = sum(s.decision.kind == "extra" for s in multi)
    ms.extra_multi_as_episode_high = sum(
        s.high and s.decision.kind == "episode" for s in multi
    )
    ms.episode_unknown_files = len(unknown)
    ms.episode_unknown_high_as_episode = sum(
        s.high and s.decision.kind == "episode" for s in unknown
    )
    ms.episode_unknown_flagged = sum(not s.high for s in unknown)
    ms.episode_unknown_provable_errors = sum(s.provable_error for s in unknown)
    return ms


def _path_season(path: str) -> int | None:
    for part in reversed(PurePath(path).parts[:-1]):
        m = _SEASON_DIR_RE.match(part)
        if m:
            return int(m.group(1))
    return None


def _season(label: ResolvedLabel, bundles: Mapping[str, Bundle] | None) -> int | None:
    """Season of an episode label from its ordering entry; else from a `Season N` folder."""
    if label.kind == "episode" and bundles and label.row.show in bundles:
        first = (_split_multi(label.row.label.strip()) or [label.row.label.strip()])[0]
        _, season, reason = _resolve_episode(
            ManifestRow(label.row.path, label.row.show, first, label.row.order, label.row.disc),
            bundles[label.row.show],
        )
        if not reason:
            return season
    if label.kind == "episode":
        first = (_split_multi(label.row.label.strip()) or [label.row.label.strip()])[0]
        m = _EPISODE_RE.match(first)
        if m:
            return int(m.group(1))
    return _path_season(label.row.path)


def compute_metrics(
    labels: Iterable[ResolvedLabel],
    decisions: Iterable[FileDecision] | Mapping[str, FileDecision],
    bundles: Mapping[str, Bundle] | None = None,
) -> EvalResult:
    """Every §16.1 metric that doesn't need the pipeline, overall, per show and per season.

    Decisions for unlabeled paths are ignored; labels without a decision become Problems.
    """
    labels = list(labels)
    if isinstance(decisions, Mapping):
        by_path = dict(decisions)
    else:
        by_path = {d.path: d for d in decisions}

    held: dict[str, set[str]] = defaultdict(set)
    for lab in labels:
        if lab.kind == "episode" and lab.episode_id is not None:
            held[lab.row.show].update(lab.episode_ids or (lab.episode_id,))

    problems: list[Problem] = []
    scored: list[_Scored] = []
    for lab in labels:
        d = by_path.get(lab.row.path)
        if d is None:
            problems.append(Problem(lab.row, "no decision"))
        elif lab.kind == "episode?":
            scored.append(_Scored(lab, d, None, _provable_error(lab, d, held)))
        else:
            scored.append(_Scored(lab, d, _is_correct(lab, d)))

    per_show: dict[str, list[_Scored]] = defaultdict(list)
    per_season: dict[str, list[_Scored]] = defaultdict(list)
    for s in scored:
        per_show[s.label.row.show].append(s)
        season = _season(s.label, bundles)
        if season is not None:
            per_season[f"{s.label.row.show}-S{season:02d}"].append(s)

    return EvalResult(
        overall=_metric_set(scored),
        per_season={k: _metric_set(v) for k, v in sorted(per_season.items())},
        per_show={k: _metric_set(v) for k, v in sorted(per_show.items())},
        problems=problems,
    )


def evaluate(
    rows: Iterable[ManifestRow],
    bundles: Mapping[str, Bundle],
    decisions: Iterable[FileDecision] | Mapping[str, FileDecision],
) -> EvalResult:
    """resolve_labels + compute_metrics, with both kinds of Problem in the result."""
    labels, problems = resolve_labels(rows, bundles)
    result = compute_metrics(labels, decisions, bundles)
    result.problems = problems + result.problems
    return result
