"""Eval runner (spec §16.1): runs the pipeline over a labeled manifest and computes the metrics.

    decisions, result = run_eval(rows, bundles, config, bundle_dirs, cache)

Labeled files are grouped by (show, season); each group is one `Session` in the ordering its
labels are written in. The history prior is always off. Files that can't be read become
`Problem`s; the run continues.

Disc structure (spec §5.1, §11.5): each group is handed a one-season tree (`group_tree`) whose
discs come from the manifest's `disc` column, else the file's `Disc N` / `D1` / volume-label
folder, else the MKV title tag (the library is flattened, but many files keep a disc-level tag).
With `use_blocks` (the default) the Session solves with the disc block model and ordering
detection; `use_blocks=False` is the basic assignment (`eval --no-blocks`).

Labeled files are UNLABELED INPUT: their tree nodes carry `named=None`, so a file whose name
parses as an episode (`show_s01e02.mkv`) is analyzed and scored, never taken as a pin (§5.5's
ground-truth rule is identify's, not eval's).

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

import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from showvenist.analysis.types import AnalysisCache
from showvenist.assign.basic import assign_basic
from showvenist.assign.blocks import block_confidence, solve_blocks
from showvenist.assign.confidence import confidence
from showvenist.bundle.schema import Bundle
from showvenist.config import Config, HierarchyConfig, WeightsConfig
from showvenist.eval.harness import compute_metrics, find_ordering, resolve_labels
from showvenist.eval.types import (
    EvalResult,
    FileDecision,
    LabelKind,
    ManifestRow,
    Problem,
    ResolvedLabel,
)
from showvenist.fusion import fuse
from showvenist.hierarchy.parse import parse_disc
from showvenist.hierarchy.types import DiscNode, DiscRef, FileNode, SeasonNode, ShowNode
from showvenist.scoring.types import Assignment, FileConfidence, Regime, Scorer
from showvenist.session import Session, Solution, bundle_vocabulary

# progress(done, total, path, error) over every labeled file of the run
Progress = Callable[[int, int, Path, str | None], None]

_SEASON_RE = re.compile(r"^season[\s._-]*(\d{1,3})$", re.IGNORECASE)
_LABEL_RE = re.compile(r"^s(\d{1,3})e(\d{1,4})$", re.IGNORECASE)


# approve(estimated_usd) -> may the LLM scorer make (paid) calls for this group?
Approve = Callable[[float], bool]


@dataclass
class GroupRun:
    show: str
    season: int | None
    ordering: str
    session: Session
    solution: Solution
    labels: list[ResolvedLabel]
    # wall-clock seconds: "analyze" (I/O + OCR), "score", "solve" (incl. the LLM pass), "total"
    seconds: dict[str, float] = field(default_factory=dict)
    first_levels: list[str] = field(default_factory=list)  # per file, solve without the LLM
    llm_sent: list[int] = field(default_factory=list)  # file indices sent to the LLM
    llm_cost_usd: float = 0.0  # new API spend (cache hits are free)
    llm_estimate_usd: float = 0.0
    llm_refused: bool = False  # `approve` said no: the group was solved without the LLM

    @property
    def key(self) -> str:
        return f"{self.show}-S{self.season or 0:02d}"

    @property
    def n_discs(self) -> int:
        """Discs of the group with at least one analyzed file (block model input)."""
        return sum(1 for d in self.solution.discs if d.files)


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


def _group_season(lab: ResolvedLabel) -> tuple[int | None, str | None]:
    """(season, mismatch note). Production reads the season from the folder (§11.3), so the
    `Season N` folder wins; the label's season (in its own ordering) is the fallback when the
    path has none. A folder and label that disagree (e.g. chronology labels) get a note."""
    folder = path_season(lab.row.path)
    label = None
    if lab.kind == "episode":
        m = _LABEL_RE.match(lab.row.label.strip())
        if m:
            label = int(m.group(1))
    if folder is None:
        return label, None
    if label is not None and label != folder:
        return folder, (
            f"season mismatch: folder says Season {folder}, label {lab.row.label} "
            f"({lab.row.order}); kept, scored in the folder's season"
        )
    return folder, None


def _ordering_key(bundle: Bundle, name: str) -> str | None:
    o = find_ordering(bundle, name)
    if o is None:
        return None
    return next(k for k, v in bundle.orderings.items() if v is o)


def _title_tag(path: Path, cache: AnalysisCache | None) -> str | None:
    from showvenist.analysis.probe import probe

    try:
        return probe(path, cache).title_tag
    except Exception:  # noqa: BLE001 - unreadable: the analysis reports it
        return None


def group_tree(
    labels: Sequence[ResolvedLabel],
    season: int | None,
    tag_of: Callable[[Path], str | None],
    hierarchy: HierarchyConfig | None = None,
) -> ShowNode:
    """A one-season tree for a group of labeled files (module doc). Disc identity, in order:
    the manifest's `disc` column, the parent folder's name, the MKV title tag. Files with none
    are loose. Every FileNode has `named=None` (eval input is unlabeled)."""
    pats = hierarchy.disc_patterns if hierarchy is not None else None
    node = SeasonNode(season, None)
    discs: dict[str, DiscNode] = {}
    for lab in labels:
        path = Path(lab.row.path)
        try:
            st = os.stat(path)
            size, mtime = st.st_size, st.st_mtime
        except OSError:
            size, mtime = 0, 0.0
        tag = tag_of(path)
        tag_ref = parse_disc(tag, pats) if tag else None
        fn = FileNode(path, size, mtime, title_tag=tag,
                      tag_disc=DiscRef(tag_ref.number, tag_ref.season, tag_ref.label, "tag")
                      if tag_ref else None, named=None)
        ref: DiscRef | None = None
        key = folder = None
        col = lab.row.disc.strip()
        if col:
            ref = parse_disc(col, pats) or DiscRef(int(col) if col.isdigit() else None, None, col)
            key = f"manifest:{col}"
        elif (fref := parse_disc(path.parent.name, pats)) is not None:
            ref, key, folder = fref, str(path.parent), path.parent
        elif tag_ref is not None:
            ref = fn.tag_disc
            key = "tag:" + " ".join(tag_ref.label.split()).casefold()
        if ref is None or key is None:
            node.loose.append(fn)
            continue
        d = discs.get(key)
        if d is None:
            d = discs[key] = DiscNode(
                key, ref, folder, order_source="number" if ref.number is not None else "name"
            )
        d.files.append(fn)
    node.discs = sorted(discs.values(), key=lambda d: (d.number is None, d.number or 0, d.key))
    return ShowNode(root=Path("/"), seasons=[node])


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


def _decisions(
    session: Session,
    solution: Solution,
    column: Sequence[int] | None = None,
    confs: Sequence[FileConfidence] | None = None,
) -> dict[Path, FileDecision]:
    column = solution.assignment.column if column is None else column
    confs = solution.confidences if confs is None else confs
    out: dict[Path, FileDecision] = {}
    for i, f in enumerate(solution.files):
        col = column[i]
        conf = confs[i]
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
    llm_mode: str = "off",
    llm_factory: Callable[[], Any] | None = None,
    approve: Approve | None = None,
    use_blocks: bool = True,
    tag_of: Callable[[Path], str | None] | None = None,
) -> tuple[list[FileDecision], EvalResult]:
    """Analyze, score and solve every labeled file, then compute the §16.1 metrics.

    Groups are (show, season): the season comes from the `Season N` folder, as in production
    (§11.3); only a path without one falls back to the episode label's season. A folder and
    label that disagree are reported as a Problem (the file is kept, in the folder's season).
    Files with no season form a whole-show group only when the show has no seasoned group;
    otherwise they're Problems. Each group is solved in the ordering most of its rows name.
    History prior is off. `scorers` replaces the default scorer set.

    LLM (§10.2): with `llm_mode` "auto" or "all", `llm_factory()` builds an LlmScorer per group;
    after the first solve its `send` set is chosen, its cost estimated, and it runs only if
    `approve(estimate)` is true (default: only when the estimate is 0, i.e. all cached).
    Actual spend lands in `cost_usd` (overall and per season).

    Blocks: every group gets a disc tree (`group_tree`); `use_blocks=False` solves without
    the block model. `tag_of(path)` reads the MKV title tag (default: the cached probe).
    """
    if llm_mode not in ("off", "auto", "all"):
        raise ValueError(f"unknown LLM mode {llm_mode!r} (off, auto, all)")
    labels, problems = resolve_labels(manifest_rows, bundles)
    groups: dict[tuple[str, int | None], list[ResolvedLabel]] = defaultdict(list)
    for lab in labels:
        season, note = _group_season(lab)
        if note:
            problems.append(Problem(lab.row, note))
        groups[(lab.row.show, season)].append(lab)
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
    cost: dict[str, float] = defaultdict(float)
    read_tag = tag_of or (lambda p: _title_tag(p, cache))

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
            tree = group_tree(labs, season, read_tag, config.hierarchy)
            session = Session(
                bundle, bdir, config, ordering=order, season=season, cache=cache,
                scorers=list(scorers) if scorers is not None else None, vocabulary=vocab[show],
                tree=tree, use_blocks=use_blocks,
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

        t0 = time.perf_counter()
        session.analyze(list(by_path), progress=_progress)
        t1 = time.perf_counter()
        for path, err in session.errors.items():
            problems.append(Problem(by_path[path].row, f"unreadable: {err}"))
        if not session.files:
            continue
        session.score()
        t2 = time.perf_counter()
        solution = session.solve()
        run = GroupRun(show, season, order, session, solution, labs,
                       first_levels=[c.level for c in solution.confidences])
        if llm_mode != "off" and llm_factory is not None:
            llm = llm_factory()
            llm.send = set(session.llm_targets(llm_mode))
            run.llm_sent = sorted(llm.send)
            if llm.send:
                run.llm_estimate_usd = llm.estimate_cost(session.files, session.context())
                ok = approve(run.llm_estimate_usd) if approve else run.llm_estimate_usd == 0
                if ok:
                    solution = session.apply_llm(llm)
                    run.solution = solution
                    run.llm_cost_usd = llm.total_cost_usd
                    cost[run.key] += llm.total_cost_usd
                else:
                    run.llm_refused = True
        t3 = time.perf_counter()
        run.seconds = {"analyze": t1 - t0, "score": t2 - t1, "solve": t3 - t2, "total": t3 - t0}
        for path, d in _decisions(session, solution).items():
            lab = by_path[path]
            decisions[lab.row.path] = FileDecision(
                path=lab.row.path, kind=d.kind, episode_id=d.episode_id, level=d.level,
                delta=d.delta, duplicate=d.duplicate,
            )
            kept.append(lab)
        if on_group is not None:
            on_group(run)

    result = compute_metrics(kept, decisions, bundles)
    result.problems = problems + result.problems
    _add_cost(result, cost)
    return list(decisions.values()), result


def _add_cost(result: EvalResult, cost: Mapping[str, float]) -> None:
    result.overall.cost_usd = float(sum(cost.values()))
    for key, usd in cost.items():
        if key in result.per_season:
            result.per_season[key].cost_usd = usd
        show = key.rsplit("-S", 1)[0]
        if show in result.per_show:
            result.per_show[show].cost_usd += usd


# ---- re-solving stored evidence (ablation, fitted weights) -----------------------------------


def group_decisions(
    group: GroupRun,
    weights: WeightsConfig | None = None,
    drop: Collection[str] = (),
) -> dict[str, FileDecision]:
    """Re-fuse a solved group's stored raw evidence (minus the `drop`ped scorers, with
    `weights` or the config's) and re-solve the way the group was solved: with the disc block
    model in the ordering it ended in when the solve used blocks (ordering detection is not
    re-run), else the basic assignment. Keyed by manifest path, duplicates included. Nothing
    is re-analyzed or re-scored."""
    sol = group.solution
    w = group.session.config.weights if weights is None else weights
    results = {k: v for k, v in sol.results.items() if k not in drop}
    if results:
        fused = fuse(results, w)
    else:
        fused = fuse({}, w, history_penalty=np.zeros((len(sol.files), len(sol.candidates))))
    if sol.used_blocks and sol.discs and not sol.pins:
        params = group.session.block_params
        bs = solve_blocks(fused.S, sol.discs, sol.loose, params)
        assignment = Assignment(list(bs.column), bs.total)
        confs = block_confidence(fused.S, sol.discs, bs, sol.loose, params)
    else:
        assignment = assign_basic(fused.S)
        confs = confidence(fused.S, assignment)
    by_path = {Path(lab.row.path): lab for lab in group.labels}
    out: dict[str, FileDecision] = {}
    for path, d in _decisions(group.session, sol, assignment.column, confs).items():
        lab = by_path[path]
        out[lab.row.path] = FileDecision(
            path=lab.row.path, kind=d.kind, episode_id=d.episode_id, level=d.level,
            delta=d.delta, duplicate=d.duplicate,
        )
    return out


def resolve_metrics(
    groups: Iterable[GroupRun],
    bundles: Mapping[str, Bundle],
    weights: WeightsConfig | None = None,
    drop: Collection[str] = (),
) -> EvalResult:
    """The §16.1 metrics over stored groups, re-solved by `group_decisions` (ablation:
    `drop={"synopsis"}` is the run without the synopsis scorer)."""
    decisions: dict[str, FileDecision] = {}
    kept: list[ResolvedLabel] = []
    for g in groups:
        got = group_decisions(g, weights, drop)
        decisions.update(got)
        kept += [lab for lab in g.labels if lab.row.path in got]
    return compute_metrics(kept, decisions, bundles)
