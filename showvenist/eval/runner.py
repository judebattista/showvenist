"""Eval runner (spec §16.1): runs the pipeline over a labeled manifest and computes the metrics.

    decisions, result = run_eval(rows, bundles, config, bundle_dirs, cache)

Labeled files are grouped by (show, season); each group is one `Session` (flat list, basic
assignment) in the ordering its labels are written in. The history prior is always off. Files
that can't be read become `Problem`s; the run continues.

Evidence for the weight fit (`eval/fit.py`)
-------------------------------------------
Pass `on_group=` to receive a `GroupRun` per solved group, then call
`evidence_table(group.solution, group.labels)`. It returns `list[EvidenceRow]`, one per primary
(non-duplicate) file of the solve, in `solution.files` order:

    EvidenceRow.path         str, the manifest path
    EvidenceRow.candidates   list[str], episode ids; column j of every array below
    EvidenceRow.llr          dict[str, np.ndarray (E,)]: each scorer's RAW evidence
                             (unclamped, unweighted; 0 = no information), keyed by scorer name
    EvidenceRow.available    dict[str, bool]: the scorer had an observation for the file
    EvidenceRow.regime       "base" | "llm" (which scorers fusion sums, §11.1)
    EvidenceRow.log_prior    float, the prior added to every episode column (§11.2)
    EvidenceRow.label_kind   "episode" | "extra" | "episode?" | None (file not labeled)
    EvidenceRow.target       int | None: the labeled column; -1 for "extra"; None when the
                             label is `episode?`, missing, or its episode isn't a candidate
    EvidenceRow.target_set   list[int] | None: for `episode?` only (decision 41), the columns
                             not held by another labeled file of the same show
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

import numpy as np

from showvenist.analysis.types import AnalysisCache
from showvenist.bundle.schema import Bundle
from showvenist.config import Config
from showvenist.eval.harness import compute_metrics, find_ordering, resolve_labels
from showvenist.eval.types import (
    EvalResult,
    FileDecision,
    LabelKind,
    ManifestRow,
    Problem,
    ResolvedLabel,
)
from showvenist.scoring.types import Regime, Scorer
from showvenist.session import Session, Solution, bundle_vocabulary

# progress(done, total, path, error) over every labeled file of the run
Progress = Callable[[int, int, Path, str | None], None]

_SEASON_RE = re.compile(r"^season[\s._-]*(\d{1,3})$", re.IGNORECASE)
_LABEL_RE = re.compile(r"^s(\d{1,3})e(\d{1,4})$", re.IGNORECASE)


@dataclass
class GroupRun:
    show: str
    season: int | None
    ordering: str
    session: Session
    solution: Solution
    labels: list[ResolvedLabel]


@dataclass
class EvidenceRow:
    path: str
    candidates: list[str]
    llr: dict[str, np.ndarray]
    available: dict[str, bool]
    regime: Regime
    log_prior: float
    label_kind: LabelKind | None
    target: int | None
    target_set: list[int] | None = None


def path_season(path: str) -> int | None:
    """Season from the nearest `Season N` / `SeasonNN` / `Season_NN` folder."""
    for part in reversed(PurePath(path).parts[:-1]):
        m = _SEASON_RE.match(part)
        if m:
            return int(m.group(1))
    return None


def _label_season(lab: ResolvedLabel) -> int | None:
    if lab.kind == "episode":
        m = _LABEL_RE.match(lab.row.label.strip())
        if m:
            return int(m.group(1))  # the label is written in its ordering's numbering
    return path_season(lab.row.path)


def _ordering_key(bundle: Bundle, name: str) -> str | None:
    o = find_ordering(bundle, name)
    if o is None:
        return None
    return next(k for k, v in bundle.orderings.items() if v is o)


def evidence_table(
    solution: Solution, labels: Iterable[ResolvedLabel] | None = None
) -> list[EvidenceRow]:
    """Per file: each scorer's raw (E,) LLR and the label target (structure: module doc)."""
    labels = list(labels or [])
    by_path = {lab.row.path: lab for lab in labels}
    by_resolved = {str(Path(lab.row.path)): lab for lab in labels}
    col_of = {c.id: c.index for c in solution.candidates}
    held = {lab.episode_id for lab in labels if lab.kind == "episode"}
    out: list[EvidenceRow] = []
    for i, f in enumerate(solution.files):
        lab = by_path.get(str(f.path)) or by_resolved.get(str(f.path))
        kind = lab.kind if lab else None
        target: int | None = None
        tset: list[int] | None = None
        if lab is not None and lab.kind == "extra":
            target = -1
        elif lab is not None and lab.kind == "episode":
            target = col_of.get(lab.episode_id or "")
        elif lab is not None and lab.kind == "episode?":
            tset = [c.index for c in solution.candidates if c.id not in held]
        out.append(
            EvidenceRow(
                path=lab.row.path if lab else str(f.path),
                candidates=[c.id for c in solution.candidates],
                llr={k: np.asarray(r.llr[i], dtype=float).copy() for k, r in
                     solution.results.items()},
                available={k: bool(r.available[i]) for k, r in solution.results.items()},
                regime=solution.fused.regimes[i],
                log_prior=solution.fused.log_prior,
                label_kind=kind,
                target=target,
                target_set=tset,
            )
        )
    return out


def _decisions(session: Session, solution: Solution) -> dict[Path, FileDecision]:
    out: dict[Path, FileDecision] = {}
    for i, f in enumerate(solution.files):
        col = solution.assignment.column[i]
        conf = solution.confidences[i]
        out[f.path] = FileDecision(
            path=str(f.path),
            kind="extra" if col < 0 else "episode",
            episode_id=None if col < 0 else solution.candidates[col].id,
            level=conf.level,
            delta=conf.delta,
        )
    for dup, primary in session.duplicates.items():
        p = out[primary]
        out[dup] = FileDecision(
            path=str(dup), kind=p.kind, episode_id=p.episode_id, level=p.level,
            delta=p.delta, duplicate=True,
        )
    return out


def run_eval(
    manifest_rows: Iterable[ManifestRow],
    bundles: Mapping[str, Bundle],
    config: Config,
    bundle_dirs: Mapping[str, Path | None],
    cache: AnalysisCache | None,
    progress: Progress | None = None,
    scorers: Sequence[Scorer] | None = None,
    on_group: Callable[[GroupRun], None] | None = None,
) -> tuple[list[FileDecision], EvalResult]:
    """Analyze, score and solve every labeled file, then compute the §16.1 metrics.

    Groups are (show, season): an episode label's season comes from the label itself (written
    in its row's ordering); extras and `episode?` take it from a `Season N` folder. Files with
    no season form a whole-show group only when the show has no seasoned group; otherwise
    they're Problems. Each group is solved in the ordering most of its rows name. History prior
    is off. `scorers` replaces the default scorer set (ablation; an LlmScorer may be included).
    """
    labels, problems = resolve_labels(manifest_rows, bundles)
    groups: dict[tuple[str, int | None], list[ResolvedLabel]] = defaultdict(list)
    for lab in labels:
        groups[(lab.row.show, _label_season(lab))].append(lab)
    seasoned = {show for show, season in groups if season is not None}
    for show, season in list(groups):
        if season is None and show in seasoned:
            for lab in groups.pop((show, None)):
                problems.append(Problem(lab.row, "no season: no `Season N` folder in its path"))

    total = sum(len(v) for v in groups.values())
    done = 0
    vocab: dict[str, set[str]] = {}
    decisions: dict[str, FileDecision] = {}
    kept: list[ResolvedLabel] = []

    for (show, season), labs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        bundle = bundles[show]
        bdir = bundle_dirs.get(show)
        orders = Counter(lab.row.order for lab in labs if lab.kind == "episode") or Counter(
            lab.row.order for lab in labs
        )
        order = _ordering_key(bundle, orders.most_common(1)[0][0])
        try:
            if order is None:
                raise ValueError(f"unknown ordering {orders.most_common(1)[0][0]!r}")
            if show not in vocab:
                vocab[show] = bundle_vocabulary(bundle, bdir)
            session = Session(
                bundle, bdir, config, ordering=order, season=season, cache=cache,
                scorers=list(scorers) if scorers is not None else None, vocabulary=vocab[show],
            )
        except Exception as e:
            for lab in labs:
                problems.append(Problem(lab.row, f"group {show} S{season}: {e}"))
                done += 1
                if progress is not None:
                    progress(done, total, Path(lab.row.path), str(e))
            continue

        by_path = {Path(lab.row.path): lab for lab in labs}

        def _progress(n: int, t: int, path: Path, err: str | None) -> None:
            nonlocal done
            done += 1
            if progress is not None:
                progress(done, total, path, err)

        session.analyze(list(by_path), progress=_progress)
        for path, err in session.errors.items():
            problems.append(Problem(by_path[path].row, f"unreadable: {err}"))
        if not session.files:
            continue
        session.score()
        solution = session.solve()
        for path, d in _decisions(session, solution).items():
            lab = by_path[path]
            decisions[lab.row.path] = FileDecision(
                path=lab.row.path, kind=d.kind, episode_id=d.episode_id, level=d.level,
                delta=d.delta, duplicate=d.duplicate,
            )
            kept.append(lab)
        if on_group is not None:
            on_group(GroupRun(show, season, order, session, solution, labs))

    result = compute_metrics(kept, decisions, bundles)
    result.problems = problems + result.problems
    return list(decisions.values()), result
