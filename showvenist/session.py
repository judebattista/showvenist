"""Engine API used by the TUI and report mode (spec §12.1).

    s = Session(bundle, bundle_dir, config, ordering="aired", season=1, cache=cache,
                tree=tree)        # tree: the parsed Show/Season/Disc hierarchy (§5), optional
    s.analyze(paths)          # stage 1 + 2, concurrent and cached (§8.5); skips not-ready and
                              # already-named files; exact and near duplicates set aside (§14)
    s.score()                 # raw evidence per scorer, in the matching order
    sol = s.solve()           # fusion + disc block DP / ordering detection or basic + confidence
    rep = s.report()          # report_schema.Report
    s.pin(path, "S01E03")     # or "extra"; re-solves (block scores are reused)
    s.unpin(path)

    # optional LLM pass (§10.2 "Which files are sent"); the caller checks the cost first
    llm.send = s.llm_targets("auto")      # after a first solve: the files below high
    cost = llm.estimate_cost(s.files, s.context())
    s.apply_llm(llm)                      # asks disc by disc, adds its evidence, re-solves

Rows of the solve (`Solution.files`, rows of S): the analyzed primary files (`files`, the rows
the scorers see), then one row per **named** file (§5.5) whose episode is in scope. A named file
is never analyzed or scored; its row is a pin holding its episode and it is a member of its disc,
so the block DP knows that disc covers the position (a deliberate reading of "it acts as a pin
and its disc is known": pins plus disc membership, no separate fixed-block logic; a disc whose
files are all named is pinned position by position, which fixes its block in practice).

Disc structure comes only from `tree` (§5.1: folder or title-tag discs). Without a tree, or with
`use_blocks=False`, the season is solved with the basic assignment (§11.5).

Ordering detection (§11.6) runs whenever the evidence changes (first solve, after the LLM pass),
with every pin applied in each ordering; pin/unpin re-solve in the ordering already chosen,
reusing the cached block scores of that ordering's S (pins are passed to the DP as forbidden
pairs, so S itself never changes on a pin).

No matching logic lives outside this module and the modules it calls.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from showvenist.analysis.cardframes import read_cards
from showvenist.analysis.duplicates import (
    Quality,
    Shingles,
    choose_primary,
    find_near_duplicates,
    overlap,
    shingle_set,
)
from showvenist.analysis.pipeline import analyze_files
from showvenist.analysis.subtitles import normalize
from showvenist.analysis.types import AnalysisCache
from showvenist.assign.basic import NEG, assign_basic
from showvenist.assign.blocks import _BlockCache, _ordered, _solve
from showvenist.assign.confidence import confidence
from showvenist.assign.ordering import detect_ordering
from showvenist.assign.types import BlockParams, DiscSpec, OrderingChoice
from showvenist.bundle.schema import Bundle, lang_codes
from showvenist.bundle.store import slugify
from showvenist.config import Config
from showvenist.edgeflags import play_all, possible_two_parter
from showvenist.fusion import fuse
from showvenist.hierarchy.marker import _find_ordering, named_episodes
from showvenist.hierarchy.types import DiscNode, FileNode, Marker, SeasonNode, ShowNode
from showvenist.report import _decision, _signals, build_report, format_odds, note_lines
from showvenist.report import _version as report_version
from showvenist.report_schema import (
    CandidateDetail,
    Confidence,
    Decision,
    EpisodeRef,
    FileDetail,
    FileReport,
    Report,
    Summary,
)
from showvenist.scoring.candidates import build_candidates
from showvenist.scoring.characters import GuestScorer
from showvenist.scoring.cues import CuesScorer
from showvenist.scoring.duration import DurationScorer
from showvenist.scoring.llm import floor_evidence
from showvenist.scoring.refsubs import RefsubsScorer
from showvenist.scoring.synopsis import SynopsisScorer
from showvenist.scoring.title import TitleScorer
from showvenist.scoring.titlecard import TitleCardScorer
from showvenist.scoring.types import (
    HIGH_DELTA,
    REGIME_SCORERS,
    Assignment,
    Candidate,
    FileConfidence,
    FileInfo,
    Fused,
    Regime,
    ScoreContext,
    Scorer,
    ScoreResult,
    level_of,
)
from showvenist.state.db import Match
from showvenist.state.fingerprint import fingerprint
from showvenist.state.history import History

# progress(done, total, path, error): called once per input path after it is handled
Progress = Callable[[int, int, Path, str | None], None]

PIN_BONUS = 1e6  # added to a pinned (file, episode) cell; far above any evidence, far below NEG
OUT_OF_SCOPE_DELTA = HIGH_DELTA  # §11.3: warn when out-of-scope evidence wins by > 4.6 nats
_DISC_RE = re.compile(r"(^|[\s._-])(disc|disk|d)[\s._-]*\d+$", re.IGNORECASE)
_LABEL_RE = re.compile(r"^s(\d{1,3})e(\d{1,4})$", re.IGNORECASE)

FLAG_DUPLICATE = "duplicate"
FLAG_NEAR_DUPLICATE = "near_duplicate"  # same dialogue, different bytes (§14); with `duplicate`
FLAG_DUPLICATE_ALLOWED = "duplicate_allowed"  # a near duplicate kept as another version
FLAG_NO_DIALOGUE = "no_dialogue"  # no usable subtitle track: dialogue scorers not available
FLAG_PINNED = "pinned"
FLAG_NOT_READY = "not_ready"  # still being written (§5.2): not analyzed
FLAG_NAMED = "named"  # name already parses as an episode (§5.5): ground truth, never renamed
FLAG_OUT_OF_SCOPE = "out_of_scope"  # evidence points to another season (§11.3)
FLAG_DISC_WINDOW_FAILED = "disc_window_failed"  # §10.2
FLAG_PLAY_ALL = "play_all"  # long file whose windows point to different episodes (§14)
FLAG_POSSIBLE_TWO_PARTER = "possible_two_parter"  # long as two episodes, both fit (§14)
FLAG_UNREADABLE = "unreadable"  # could not be read or analyzed: needs review (§4.5)
FLAG_CONFIRMED = "confirmed"  # decided earlier and read straight from history (§4.3)
FLAG_HISTORY_DUPLICATE = "history_duplicate"  # its episode is confirmed for another file (§11.4)


def ordering_key(bundle: Bundle, name: str) -> str | None:
    """The bundle key of the ordering `name` refers to: a key (any case), an ordering type or a
    Jellyfin keyword (the marker's resolver, §5.4, §5.5); None when it names none."""
    o = _find_ordering(bundle, name)
    return None if o is None else next(k for k, v in bundle.orderings.items() if v is o)


def default_scorers() -> list[Scorer]:
    return [
        DurationScorer(),
        TitleScorer(),
        TitleCardScorer(),
        RefsubsScorer(),
        SynopsisScorer(),
        GuestScorer(),
        CuesScorer(),
    ]


def disc_of(path: Path) -> str | None:
    """Name of the nearest parent folder that looks like a disc (`Disc 1`, `DISC_2`, `D3`,
    MakeMKV volume labels like `FIREFLY_D1`), or None."""
    for part in reversed(Path(path).parts[:-1]):
        if _DISC_RE.search(part):
            return part
    return None


def disc_label(disc: DiscNode) -> str:
    """How a tree disc is shown in reports: its folder name, else the title-tag label."""
    if disc.folder is not None:
        return disc.folder.name
    if disc.ref is not None and disc.ref.label:
        return disc.ref.label
    return disc.key


def _ref_words(path: Path) -> set[str]:
    import pysubs2

    for enc in ("utf-8", "latin-1"):
        try:
            subs = pysubs2.load(str(path), encoding=enc)
            break
        except Exception:
            continue
    else:
        return set()
    out: set[str] = set()
    for ev in subs:
        if not ev.is_comment:
            out.update(normalize(ev.plaintext.replace("\\N", " ")).split())
    return out


def bundle_vocabulary(bundle: Bundle, bundle_dir: Path | None = None) -> set[str]:
    """Normalized words of titles, synopses and cast/crew names (as `inspect --show` builds it),
    plus the words of every same-language reference subtitle under `bundle_dir`."""
    vocab: set[str] = set()
    codes = lang_codes(bundle.show.language)
    for e in bundle.episodes:
        names = [*e.directors, *e.writers]
        for c in e.guest_cast:
            names += [c.actor, c.character]
        vocab.update(
            normalize(" ".join([e.title, *e.alt_titles, *(s.text for s in e.synopses), *names]))
            .split()
        )
        if bundle_dir is not None:
            for ref in e.subtitles:
                p = Path(bundle_dir) / ref.path
                if ref.lang in codes and p.is_file():
                    vocab |= _ref_words(p)
    return vocab


@dataclass
class Solution:
    """One solve over the session's rows: the primary analyzed files, then named-file pins.

    `S` is the fused matrix with pins applied (PIN_BONUS / NEG), as solved by the basic
    assignment; the block DP solves `fused.S` with the pins as forbidden pairs. `results` are
    the raw per-scorer results (unclamped, unweighted; rows = analyzed files only), so eval and
    the weight fit can read them. `candidates` are numbered in `ordering`, the ordering actually
    used (§11.6).
    """

    files: list[FileInfo]
    candidates: list[Candidate]
    results: dict[str, ScoreResult]
    fused: Fused
    S: np.ndarray
    assignment: Assignment
    confidences: list[FileConfidence]
    pins: dict[int, int] = field(default_factory=dict)  # row -> column (-1 = extra)
    ordering: str = "aired"
    used_blocks: bool = False
    switched: bool = False  # ordering detection chose another ordering than the starting one
    fits_nowhere: bool = False  # no ordering fits the discs: basic assignment used
    blocks: dict[str, tuple[int, int]] = field(default_factory=dict)  # disc key -> [a, b)
    discs: list[DiscSpec] = field(default_factory=list)
    loose: list[int] = field(default_factory=list)  # rows with no disc
    out_of_scope: dict[int, Candidate] = field(default_factory=dict)  # row -> what it points to
    notices: list[str] = field(default_factory=list)

    @property
    def regimes(self) -> list[Regime]:
        return self.fused.regimes


@dataclass
class _Evidence:
    """Everything derived from the analyzed files in one ordering; rebuilt only when the
    evidence changes (analysis, LLM), never on pin/unpin."""

    candidates: list[Candidate]
    base: dict[str, ScoreResult]  # non-LLM scorers, rows = analyzed files
    results: dict[str, ScoreResult] | None = None  # base + LLM remapped into these columns
    fused: Fused | None = None  # rows = analyzed files + named rows
    cache: _BlockCache | None = None


@dataclass
class _Named:
    """A file whose name parses as an episode (§5.5)."""

    info: FileInfo
    episode_id: str | None  # resolved in the naming order; None = unresolved (see reason)
    reason: str | None = None
    row: bool = False  # True when it is a pinned row of the solve
    confirmed: bool = False  # read from history, not from the name


class Session:
    def __init__(
        self,
        bundle: Bundle,
        bundle_dir: Path | None,
        config: Config,
        ordering: str = "aired",
        season: int | None = None,
        episodes: Iterable[tuple[int, int]] | None = None,
        include_specials: bool = False,
        cache: AnalysisCache | None = None,
        scorers: Sequence[Scorer] | None = None,
        llm: Scorer | None = None,
        vocabulary: set[str] | None = None,
        *,
        tree: ShowNode | None = None,
        use_blocks: bool = True,
        block_params: BlockParams | None = None,
        marker: Marker | None = None,
        allow_duplicates: bool = False,
        history: History | None = None,
        reanalyze: bool = False,
    ):
        self.bundle = bundle
        # History prior (§11.4) and confirmed files (§4.3); None = off, as eval requires
        self.history = history
        self.reanalyze = reanalyze  # ignore confirmed files in history (the prior stays)
        self.confirmed_extras: dict[Path, FileInfo] = {}  # confirmed extras: not analyzed
        self.allow_duplicates = allow_duplicates  # near duplicates are other versions (§4.3)
        self.bundle_dir = Path(bundle_dir) if bundle_dir is not None else None
        self.config = config
        self.tree = tree
        self.marker = marker or (tree.marker if tree is not None else None)
        self.season_node: SeasonNode | None = None
        if tree is not None:
            self.season_node = self._pick_season(tree, season)
            if season is None and self.season_node is not None:
                season = self.season_node.number
        self.starting_ordering = ordering
        self.ordering = ordering  # the ordering in use: may change after ordering detection
        self.season = season
        self.episodes = None if episodes is None else set(episodes)
        self.include_specials = include_specials or bool(
            self.marker is not None and self.marker.include_specials
        )
        self.use_blocks = use_blocks
        b = config.blocks
        self.block_params = block_params or BlockParams(
            b.unclaimed_penalty, b.gap_penalty, b.ordering_threshold,
            block_slack=b.block_slack, ordering_margin=b.ordering_margin,
        )
        self.cache = cache
        if ordering not in bundle.orderings:
            known = ", ".join(sorted(bundle.orderings))
            raise ValueError(f"unknown ordering {ordering!r} (the bundle has: {known})")
        self.candidates = self._candidates(ordering)
        if not self.candidates:
            raise ValueError(
                f"no candidate episodes in ordering {ordering!r}"
                + (f" for season {season}" if season is not None else "")
                + (" for the given --episodes" if self.episodes is not None else "")
            )
        self.scorers: list[Scorer] = [
            s for s in (list(scorers) if scorers is not None else default_scorers())
            if s.name != "llm"
        ]
        # an LLM scorer handed in up front is run once, by score(), on the full candidate list
        self._llm_initial = llm or next(
            (s for s in (scorers or ()) if s.name == "llm"), None
        )
        self.language = bundle.show.language
        self._vocab = vocabulary
        self.files: list[FileInfo] = []  # primary analyzed files: the rows the scorers see
        # duplicate path -> the primary row it mirrors (§14): exact and near duplicates, always
        # mapped straight to a row of the solve (an exact copy of a near duplicate included)
        self.duplicates: dict[Path, Path] = {}
        self.dup_infos: dict[Path, FileInfo] = {}
        self.near: dict[Path, tuple[Path, float]] = {}  # near duplicate -> (primary, overlap)
        self._exact: dict[Path, Path] = {}  # exact copy -> the file with its content key
        self._analyzed: list[FileInfo] = []  # every analyzed primary, near duplicates included
        self._quality: dict[Path, Quality] = {}
        self._shingles: dict[Path, Shingles] = {}
        self.errors: dict[Path, str] = {}  # unreadable files: path -> reason
        self.not_ready: dict[Path, FileInfo] = {}  # skipped, still being written (§5.2)
        self.named: dict[Path, _Named] = {}  # already named files (§5.5), in input order
        self._keys: dict[str, Path] = {}  # content key -> primary path
        self._node: dict[Path, tuple[DiscNode | None, FileNode]] = {}
        if self.season_node is not None:
            for d in self.season_node.discs:
                for f in d.files:
                    self._node[Path(f.path)] = (d, f)
            for f in self.season_node.loose:
                self._node[Path(f.path)] = (None, f)
        self.pins: dict[Path, str | None] = {}  # row path -> episode id (None = extra)
        self.solution: Solution | None = None
        self._evid: dict[str, _Evidence] = {}
        self._llm_rows: dict[int, tuple[dict[str, float], float, dict[str, dict]]] = {}
        self._choice: OrderingChoice | None = None
        self._oos: dict[str, tuple[list[Candidate], np.ndarray] | None] = {}
        self._window_failed: set[int] = set()
        self.notices: list[str] = []  # session-level (unresolved names, skipped files)

    # ---- scope -----------------------------------------------------------------------------

    @staticmethod
    def _pick_season(tree: ShowNode, season: int | None) -> SeasonNode | None:
        if not tree.seasons:
            return None
        if season is not None:
            hit = [s for s in tree.seasons if s.number == season]
            if not hit and len(tree.seasons) == 1:
                return tree.seasons[0]  # an explicit --season overrides the folder (§4.3)
            if not hit:
                raise ValueError(f"no season {season} in the tree under {tree.root}")
            return hit[0]
        if len(tree.seasons) > 1:
            nums = ", ".join(str(s.number) for s in tree.seasons)
            raise ValueError(f"the tree holds several seasons ({nums}): solve one at a time")
        return tree.seasons[0]

    def _candidates(self, ordering: str) -> list[Candidate]:
        """In-scope candidates in `ordering` (§11.3): the season (or --episodes) read in that
        ordering. An ordering without real seasons (one absolute list) keeps the starting
        ordering's season and only changes the episode order."""
        cands = build_candidates(
            self.bundle,
            ordering,
            season=self.season,
            episodes=self.episodes,
            include_specials=self.include_specials,
        )
        if (
            ordering != self.starting_ordering
            and self.season is not None
            and self.episodes is None
            and self._seasonless(ordering)
            and not self._seasonless(self.starting_ordering)
        ):
            ids = {c.id for c in self._candidates(self.starting_ordering)}
            by_id = self.bundle.episode_by_id()
            entries = sorted(
                self.bundle.orderings[ordering].entries, key=lambda e: (e.season, e.episode)
            )
            cands = []
            for ent in entries:
                if ent.episode_id in ids:
                    cands.append(Candidate(len(cands), by_id[ent.episode_id], ent.season,
                                           ent.episode))
        return cands

    def _seasonless(self, ordering: str) -> bool:
        seasons = {e.season for e in self.bundle.orderings[ordering].entries if e.season != 0}
        return len(seasons) <= 1

    def _out_of_scope_candidates(self, ordering: str) -> list[Candidate]:
        """Every episode of `ordering` (specials only when they would be candidates)."""
        return build_candidates(
            self.bundle,
            ordering,
            season=None,
            include_specials=self.include_specials or self.season == 0,
        )

    # ---- analysis ------------------------------------------------------------------------

    @property
    def vocabulary(self) -> set[str]:
        if self._vocab is None:
            self._vocab = bundle_vocabulary(self.bundle, self.bundle_dir)
        return self._vocab

    def _disc_name(self, path: Path) -> str | None:
        hit = self._node.get(path)
        if hit is not None:
            return disc_label(hit[0]) if hit[0] is not None else None
        return None if self.tree is not None else disc_of(path)

    def _named_order(self) -> str:
        """Names are read in the marker's naming_order (§5.5), however it is written (key,
        type or Jellyfin keyword), whatever --order says; without one, the starting ordering."""
        if self.marker is not None:
            key = ordering_key(self.bundle, self.marker.naming_order)
            if key is not None:
                return key
        return self.starting_ordering

    def analyze(
        self, paths: Iterable[Path | str] | None = None, progress: Progress | None = None
    ) -> list[FileInfo]:
        """Stage 1 + 2 for each path, concurrent and cached (§8.5). With a tree, `paths` may
        be None (every file of the season). Not-ready files (FileNode.ready False) and files
        whose names parse as an episode are not analyzed. Exact duplicates (same content key,
        §7.1) are recorded against the first copy; near duplicates (same dialogue, different
        bytes) against the best copy (see `_find_near`). Unreadable files land in `errors` and
        are skipped. Returns the primary analyzed files."""
        if paths is None:
            paths = list(self._node) if self.tree is not None else []
        paths = [Path(p) for p in paths]
        total = len(paths)
        done = 0
        todo: list[Path] = []
        named_nodes: list[FileNode] = []
        hist_hits: list[tuple[Path, str, str | None]] = []
        for path in paths:
            node = self._node.get(path)
            fn = node[1] if node is not None else None
            if fn is not None and not fn.ready:
                self.not_ready[path] = FileInfo(
                    path=path, fingerprint="", duration_s=0.0, disc=self._disc_name(path)
                )
            elif fn is not None and fn.named is not None:
                named_nodes.append(fn)
            elif (hit := self._history_hit(path)) is not None:
                hist_hits.append(hit)
            else:
                todo.append(path)
                continue
            done += 1
            if progress is not None:
                progress(done, total, path, None)
        if named_nodes:
            self._add_named(named_nodes)
        for path in self._add_confirmed(hist_hits):  # not usable as a pin: analyze it
            todo.append(path)
            done -= 1

        offset = done

        def relay(n: int, _t: int, path: Path, err: str | None) -> None:
            if progress is not None:
                progress(offset + n, total, path, err)

        results = analyze_files(
            todo,
            language=self.language,
            vocabulary=self.vocabulary if todo else set(),
            cfg=self.config.analysis,
            cache=self.cache,
            progress=relay,
            known_keys=self._keys,
        )
        for r in results:
            if r.error is not None:
                self.errors[r.path] = r.error
            elif r.duplicate_of is not None and r.duplicate_of != r.path:
                self._exact[r.path] = r.duplicate_of
                self.dup_infos[r.path] = FileInfo(
                    path=r.path,
                    fingerprint=r.fingerprint or "",
                    duration_s=0.0,
                    disc=self._disc_name(r.path),
                )
            elif r.duplicate_of is None and r.probe is not None:
                assert r.content_key is not None and r.fingerprint is not None
                self._keys[r.content_key] = r.path
                lines = sum(w.lines for w in r.dialogue.windows) if r.dialogue else 0
                self._quality[r.path] = Quality(
                    lines, r.dialogue.ocr_quality if r.dialogue else None, r.probe.size
                )
                self._analyzed.append(
                    FileInfo(
                        path=r.path,
                        fingerprint=r.fingerprint,
                        duration_s=r.probe.duration_s,
                        title_tag=r.probe.title_tag,
                        disc=self._disc_name(r.path),
                        dialogue=r.dialogue,
                        ocr_quality=r.dialogue.ocr_quality if r.dialogue else None,
                        card_reads=self._read_cards(r.path, r.fingerprint, r.probe.duration_s),
                    )
                )
        for dup, primary in list(self._exact.items()):
            if primary in self.errors:  # an exact copy of an unreadable file is unreadable too
                del self._exact[dup]
                self.dup_infos.pop(dup, None)
                self.errors[dup] = f"exact copy of {primary.name}: {self.errors[primary]}"
        self._find_near()
        self._evidence_changed(reset_llm=True)
        return list(self.files)

    def _read_cards(self, path: Path, fp: str, duration_s: float):
        """Title-card OCR reads (§8.4) when the show has `title_card`; None (no video touched)
        otherwise. Never fails an analysis. Files are read one at a time, so two reads of one
        storage root never overlap."""
        cfg = self.bundle.title_card
        if cfg is None:
            return None
        try:
            return read_cards(path, fp, duration_s, cfg, self.cache)
        except Exception:
            return []

    def _find_near(self) -> None:
        """Near duplicates (§14) among the analyzed files of this session (one season; named
        files and exact copies are never compared): groups by dialogue overlap
        (`analysis.duplicates`), each collapsed into its best copy, which stays the row. A
        pinned file is never collapsed: a group holding pinned files follows the best of them.
        Recomputed over every analyzed file on each call (shingles cached), so analyzing again
        is idempotent and a better copy analyzed later becomes the primary."""
        cfg = self.config.analysis
        sh: dict[Path, Shingles] = {}
        for f in self._analyzed:
            if f.path not in self._shingles:
                self._shingles[f.path] = shingle_set(f.dialogue)
            sh[f.path] = self._shingles[f.path]
        near: dict[Path, tuple[Path, float]] = {}
        for group in find_near_duplicates(sh, cfg.duplicate_overlap, cfg.duplicate_min_shingles):
            pinned = [p for p in group if p in self.pins]
            primary = choose_primary(pinned or group, self._quality)
            for p in group:
                if p != primary and p not in self.pins:
                    near[p] = (primary, overlap(sh[p], sh[primary]))
        for p in self.near.keys() - near.keys():
            if p not in self._exact:
                self.dup_infos.pop(p, None)
        self.near = near
        self.files = [f for f in self._analyzed if f.path not in near]
        for f in self._analyzed:
            if f.path in near:
                self.dup_infos[f.path] = f  # its own duration and dialogue, for the report
        self.duplicates = {
            d: near[p][0] if p in near else p for d, p in self._exact.items()
        } | {p: q for p, (q, _) in near.items()}

    def _add_named(self, nodes: list[FileNode]) -> None:
        rep = named_episodes(nodes, self.bundle, self._named_order())
        in_scope = {c.id for c in self.candidates}
        held = {n.episode_id for n in self.named.values() if n.row}
        for fn in nodes:
            path = Path(fn.path)
            try:
                fp = fingerprint(path)
            except OSError:
                fp = ""
            info = FileInfo(path=path, fingerprint=fp, duration_s=0.0,
                            disc=self._disc_name(path))
            eid = rep.resolved.get(fn.path)
            if eid is None:
                reason = rep.unresolved.get(fn.path, "name does not resolve to an episode")
                self.named[path] = _Named(info, None, reason)
                self.notices.append(f"{path.name}: {reason}")
            elif eid in held:
                self.named[path] = _Named(info, eid, "another named file holds this episode")
            elif eid not in in_scope:
                self.named[path] = _Named(info, eid, "named episode is outside this season")
            else:
                held.add(eid)
                self.named[path] = _Named(info, eid, None, row=True)

    def _history_hit(self, path: Path) -> tuple[Path, str, str | None] | None:
        """(path, fingerprint, episode id | None for an extra) when history holds a confirmed
        decision for this file in this show; None otherwise (or with --reanalyze)."""
        if self.history is None or self.reanalyze or path in self.pins:
            return None
        try:
            fp = fingerprint(path)
        except OSError:
            return None
        mine = [m for m in self.history.confirmed_for_file(fp) if m.show_slug == self.show_slug]
        if not mine:
            return None
        if any(m.episode_id is None for m in mine):
            return path, fp, None
        return path, fp, mine[-1].episode_id

    def _add_confirmed(self, hits: list[tuple[Path, str, str | None]]) -> list[Path]:
        """Files decided earlier: extras are reported as such, episodes become pinned rows.
        Returns the paths that can't be pinned (episode out of scope or already held), which
        are analyzed after all."""
        in_scope = {c.id for c in self.candidates}
        held = {n.episode_id for n in self.named.values() if n.row}
        back: list[Path] = []
        for path, fp, eid in hits:
            info = FileInfo(path=path, fingerprint=fp, duration_s=0.0,
                            disc=self._disc_name(path))
            if eid is None:
                self.confirmed_extras[path] = info
            elif eid in in_scope and eid not in held:
                held.add(eid)
                self.named[path] = _Named(info, eid, None, row=True, confirmed=True)
            else:
                back.append(path)
        return back

    def _history_penalty(self, cands: list[Candidate]) -> np.ndarray | None:
        if self.history is None or not self.files:
            return None
        m = np.asarray(self.history.history_penalty(
            self.show_slug, [f.fingerprint for f in self.files], [c.id for c in cands],
            self.config.history.penalty,
        ), dtype=float)
        return m if m.any() else None

    def _evidence_changed(self, reset_llm: bool = False) -> None:
        """New analysis (reset_llm): everything goes. New LLM evidence: the base scorers'
        results stay; the fused matrices, block caches and the ordering choice go."""
        if reset_llm:
            self._evid = {}
            self._oos = {}
            self._llm_rows = {}
            self._window_failed = set()
        else:
            for ev in self._evid.values():
                ev.results = ev.fused = ev.cache = None
        self._choice = None
        self.solution = None
        if self.ordering != self.starting_ordering:
            self._set_ordering(self.starting_ordering)

    def _set_ordering(self, ordering: str) -> None:
        if ordering == self.ordering:
            return
        self.ordering = ordering
        ev = self._evid.get(ordering)
        self.candidates = ev.candidates if ev is not None else self._candidates(ordering)

    # ---- rows, pins and discs --------------------------------------------------------------

    @property
    def named_rows(self) -> list[_Named]:
        return [n for n in self.named.values() if n.row]

    @property
    def rows(self) -> list[FileInfo]:
        return self.files + [n.info for n in self.named_rows]

    def _row_pins(self, candidates: list[Candidate]) -> dict[int, int | None]:
        """Row -> column in `candidates` (-1 = extra; None = pinned episode not a candidate)."""
        col = {c.id: c.index for c in candidates}
        out: dict[int, int | None] = {}
        F = len(self.files)
        for k, n in enumerate(self.named_rows):
            out[F + k] = col.get(n.episode_id)  # type: ignore[arg-type]
        for path, eid in self.pins.items():
            out[self._index(path)] = -1 if eid is None else col.get(eid)
        return out

    @staticmethod
    def _pinned_S(S0: np.ndarray, pins: dict[int, int | None]) -> np.ndarray:
        S = S0.copy()
        for c in pins.values():
            if c is not None and c >= 0:
                S[:, c] = NEG  # nobody else may take the episode
        for i, c in pins.items():
            S[i, :] = NEG  # extra (or an episode that is not a candidate): unmatched only
            if c is not None and c >= 0:
                S[i, c] = S0[i, c] + PIN_BONUS
        return S

    @staticmethod
    def _pin_forbidden(pins: dict[int, int | None], F: int, E: int) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for i, c in pins.items():
            out.update((i, j) for j in range(E) if j != c)
            if c is not None and c >= 0:
                out.add((i, -1))
                out.update((o, c) for o in range(F) if o != i)
        return out

    def _disc_specs(self) -> tuple[list[DiscSpec], list[int]]:
        """DiscSpecs for the season (§11.5): rows by disc in tree order, numbers from the
        DiscRef, free gaps where numbers skip, incomplete when a file is not ready. Discs with
        no file in this session are left out. Rows with no disc are loose."""
        rows = self.rows
        if self.season_node is None:
            return [], list(range(len(rows)))
        row_of = {f.path: r for r, f in enumerate(rows)}
        discs: list[DiscSpec] = []
        seen: set[int] = set()
        prev: int | None = None
        for d in self.season_node.discs:
            paths = [Path(f.path) for f in d.files]
            mine = [row_of[p] for p in paths if p in row_of]
            known = any(
                p in row_of or p in self.not_ready or p in self.duplicates or p in self.named
                for p in paths
            )
            if not known:
                continue
            incomplete = d.incomplete or any(p in self.not_ready for p in paths)
            skip = d.number is not None and prev is not None and d.number > prev + 1
            discs.append(
                DiscSpec(d.key, tuple(mine), d.number, incomplete, free_gap_before=skip)
            )
            if d.number is not None:
                prev = d.number
            seen.update(mine)
        loose = [r for r in range(len(rows)) if r not in seen]
        return discs, loose

    # ---- scoring ---------------------------------------------------------------------------

    def context(self, candidates: list[Candidate] | None = None) -> ScoreContext:
        return ScoreContext(
            bundle=self.bundle,
            candidates=self.candidates if candidates is None else candidates,
            config=self.config,
            bundle_dir=self.bundle_dir,
            language=self.language,
            vocabulary=self.vocabulary,
        )

    def _evidence(self, ordering: str) -> _Evidence:
        ev = self._evid.get(ordering)
        if ev is None:
            cands = self.candidates if ordering == self.ordering else self._candidates(ordering)
            ctx = self.context(cands)
            ev = _Evidence(cands, {s.name: s.score(self.files, ctx) for s in self.scorers})
            self._evid[ordering] = ev
        return ev

    def _llm_result(self, candidates: list[Candidate]) -> ScoreResult | None:
        """Stored LLM evidence (keyed by episode id) laid out in `candidates`' columns; an
        episode the LLM wasn't shown gets the floor. No API call: the LLM is never re-asked
        for another ordering."""
        if not self._llm_rows:
            return None
        out = ScoreResult.empty(len(self.files), len(candidates))
        for i, (by_id, floor, notes) in self._llm_rows.items():
            out.available[i] = True
            for c in candidates:
                out.llr[i, c.index] = by_id.get(c.id, floor)
                if c.id in notes:
                    out.notes[(i, c.index)] = notes[c.id]
        return out

    def _results(self, ordering: str) -> dict[str, ScoreResult]:
        ev = self._evidence(ordering)
        if ev.results is None:
            llm = self._llm_result(ev.candidates)
            ev.results = {**ev.base, **({"llm": llm} if llm is not None else {})}
        return ev.results

    def _fused(self, ordering: str) -> Fused:
        ev = self._evidence(ordering)
        if ev.fused is None:
            res = self._results(ordering)
            F, E = len(self.files), len(ev.candidates)
            hp = self._history_penalty(ev.candidates)
            if res:
                f = fuse(res, self.config.weights, history_penalty=hp)
            else:  # no scorers: prior only
                f = fuse({}, self.config.weights, history_penalty=np.zeros((F, E)))
            N = len(self.named_rows)
            if N:  # named rows: no evidence, S = 0 (their pins decide)
                pad = np.zeros((N, E))
                f = Fused(
                    S=np.vstack([f.S, pad]),
                    contributions={k: np.vstack([v, pad]) for k, v in f.contributions.items()},
                    regimes=[*f.regimes, *(["base"] * N)],
                    log_prior=f.log_prior,
                    available={k: np.concatenate([v, np.zeros(N, bool)])
                               for k, v in f.available.items()},
                )
            ev.fused = f
        return ev.fused

    @property
    def results(self) -> dict[str, ScoreResult] | None:
        ev = self._evid.get(self.ordering)
        return None if ev is None else self._results(self.ordering)

    def score(self) -> dict[str, ScoreResult]:
        """Raw evidence of every scorer in the ordering in use (cached until the evidence
        changes). An LLM scorer handed to the constructor is run here once."""
        if self._llm_initial is not None and not self._llm_rows and self.files:
            llm, self._llm_initial = self._llm_initial, None
            self._store_llm(llm, llm.score(self.files, self.context()), None)
        return self._results(self.ordering)

    # ---- solving ---------------------------------------------------------------------------

    def _build(self, ordering: str) -> tuple[np.ndarray, list[DiscSpec], list[int]]:
        """detect_ordering's build(): S in `ordering` (season filter recomputed, every pin
        applied by episode id), with the disc structure."""
        ev = self._evidence(ordering)
        S = self._pinned_S(self._fused(ordering).S, self._row_pins(ev.candidates))
        discs, loose = self._disc_specs()
        return S, discs, loose

    def _comparable_orderings(self) -> list[str]:
        return [n for n in self.bundle.orderings if self._candidates(n)]

    def solve(self) -> Solution:
        """Fusion (config weights, history prior off) + assignment + confidence, with pins as
        hard constraints. With disc structure and use_blocks: ordering detection (§11.6) and
        the block DP (§11.5); otherwise the basic assignment."""
        self.score()
        discs, loose = self._disc_specs()
        notices: list[str] = []
        if discs and self.use_blocks:
            if self._choice is None:
                self._set_ordering(self.starting_ordering)
                self._choice = detect_ordering(
                    self._build,
                    self._comparable_orderings(),
                    self.starting_ordering,
                    self.block_params,
                )
                self._set_ordering(self._choice.ordering)
            choice = self._choice
            if choice.switched:
                o = self.bundle.orderings[choice.ordering]
                notices.append(f"This release follows '{o.name or choice.ordering}'.")
            if choice.fits_nowhere:
                notices.append(
                    "The discs don't look sequential; fell back to the basic assignment."
                )
                sol = self._solve_basic(discs, loose)
            else:
                sol = self._solve_blocks(discs, loose)
                if sol is None:
                    notices.append(
                        "The pins can't be met with disc blocks; fell back to the basic "
                        "assignment."
                    )
                    sol = self._solve_basic(discs, loose)
            sol.switched, sol.fits_nowhere = choice.switched, choice.fits_nowhere
        else:
            sol = self._solve_basic(discs, loose)
        sol.out_of_scope = self._out_of_scope(sol)
        sol.notices = notices
        self.solution = sol
        return sol

    def _solution(self, S, assignment, confs, pins, discs, loose, **kw) -> Solution:
        return Solution(
            files=self.rows,
            candidates=self.candidates,
            results=self._results(self.ordering),
            fused=self._fused(self.ordering),
            S=S,
            assignment=assignment,
            confidences=confs,
            pins={i: (-1 if c is None else c) for i, c in pins.items()},
            ordering=self.ordering,
            discs=discs,
            loose=loose,
            **kw,
        )

    def _solve_basic(self, discs: list[DiscSpec], loose: list[int]) -> Solution:
        fused = self._fused(self.ordering)
        pins = self._row_pins(self.candidates)
        S = self._pinned_S(fused.S, pins)
        assignment = assign_basic(S)
        confs = confidence(S, assignment)
        for i in pins:
            confs[i] = FileConfidence(math.inf, "high", None, None)
        return self._solution(S, assignment, confs, pins, discs, loose, used_blocks=False)

    def _solve_blocks(self, discs: list[DiscSpec], loose: list[int]) -> Solution | None:
        ev = self._evidence(self.ordering)
        fused = self._fused(self.ordering)
        S0 = fused.S
        if ev.cache is None:
            ev.cache = _BlockCache(S0, self.block_params)
        F, E = S0.shape
        pins = self._row_pins(self.candidates)
        forb = self._pin_forbidden(pins, F, E)
        p = self.block_params
        sol = _solve(S0, discs, loose, p, frozenset(forb), ev.cache)
        if not math.isfinite(sol.total):
            return None
        in_solve = {f for d in discs for f in d.files} | set(loose)
        confs: list[FileConfidence] = []
        for i in range(F):
            if i in pins:
                confs.append(FileConfidence(math.inf, "high", None, None))
                continue
            if i not in in_solve:
                confs.append(FileConfidence(0.0, "low", None, None))
                continue
            alt = _solve(S0, discs, loose, p, frozenset(forb | {(i, sol.column[i])}), ev.cache)
            if not math.isfinite(alt.total):
                confs.append(FileConfidence(math.inf, "high", None, None))
                continue
            delta = max(0.0, sol.total - alt.total)
            confs.append(FileConfidence(delta, level_of(delta), alt.column[i], delta))
        return self._solution(
            self._pinned_S(S0, pins),
            Assignment(list(sol.column), sol.total),
            confs,
            pins,
            discs,
            loose,
            used_blocks=True,
            blocks=dict(sol.blocks),
        )

    def _out_of_scope(self, sol: Solution) -> dict[int, Candidate]:
        """§11.3: rows whose best out-of-scope score beats their assigned score by more than
        4.6 nats. Scored once per ordering over every season (base scorers, no LLM: the LLM
        was never shown those episodes), on one scale (one prior for the union)."""
        if self.ordering not in self._oos:
            self._oos[self.ordering] = None
            allc = self._out_of_scope_candidates(self.ordering)
            inside = {c.id for c in self.candidates}
            if self.files and any(c.id not in inside for c in allc):
                ctx = self.context(allc)
                res = {s.name: s.score(self.files, ctx) for s in self.scorers}
                if res:
                    self._oos[self.ordering] = (allc, fuse(res, self.config.weights).S)
        hit = self._oos[self.ordering]
        if hit is None:
            return {}
        allc, U = hit
        inside = {c.id for c in self.candidates}
        ucol = {c.id: c.index for c in allc}
        out_cols = [c.index for c in allc if c.id not in inside]
        out: dict[int, Candidate] = {}
        for i in range(len(self.files)):
            if i in sol.pins:
                continue
            col = sol.assignment.column[i]
            mine = 0.0 if col < 0 else float(U[i, ucol[self.candidates[col].id]])
            j = max(out_cols, key=lambda k: U[i, k])
            if U[i, j] - mine > OUT_OF_SCOPE_DELTA:
                out[i] = allc[j]
        return out

    # ---- LLM pass (§10.2) ------------------------------------------------------------------

    def llm_targets(self, mode: str) -> list[int]:
        """Indices of the primary files the LLM scorer is sent: `all` = every file; `auto` =
        files below high in the solve without the LLM (solving first if needed); `off` = none."""
        if mode == "off":
            return []
        if mode == "all":
            return list(range(len(self.files)))
        if mode != "auto":
            raise ValueError(f"unknown LLM mode {mode!r} (off, auto, all)")
        sol = self.solution or self.solve()
        return [i for i in range(len(self.files)) if sol.confidences[i].level != "high"]

    def llm_windows(self, sol: Solution | None = None) -> list[tuple[set[int], list[int] | None]]:
        """The LLM's candidate scope per group of files (§10.2): [(file indices, shown
        columns or None = the full season)]. Disc by disc when the season is split into
        discs: the next n_d episodes after the last episode of the previous disc, skipping
        positions pinned files already hold; after a missing disc number (or before a first
        disc numbered > 1) the window runs to the end of the season. n_d counts the disc's
        analyzed files that are not identified extras (named files and exact duplicates are
        not analyzed rows). Loose files, `[llm] scope = "season"` and flat lists get the full
        season."""
        sol = sol or self.solution or self.solve()
        F, E = len(self.files), len(self.candidates)
        if self.config.llm.scope == "season" or not sol.used_blocks or not sol.discs:
            return [(set(range(F)), None)]
        held = {c for c in sol.pins.values() if c >= 0}
        out: list[tuple[set[int], list[int] | None]] = []
        cursor, prev = 0, None
        for k, d in enumerate(_ordered(sol.discs)):
            rows = {r for r in d.files if r < F}
            if d.number is None:
                known = True
            elif k == 0:
                known = d.number == 1
            else:
                known = prev is not None and d.number == prev + 1
            if known:
                n = sum(
                    1 for r in rows
                    if not (sol.assignment.column[r] == -1 and sol.confidences[r].level == "high")
                )
                cols: list[int] = []
                j = cursor
                while len(cols) < n and j < E:
                    if j not in held:
                        cols.append(j)
                    j += 1
            else:
                cols = [j for j in range(cursor if k > 0 else 0, E) if j not in held]
            if rows:
                out.append((rows, cols))
            last = [sol.assignment.column[r] for r in d.files if sol.assignment.column[r] >= 0]
            if last:
                cursor = max(cursor, max(last) + 1)
            prev = d.number
        loose = {r for r in sol.loose if r < F}
        if loose:
            out.append((loose, None))
        return out

    def _store_llm(self, llm, res: ScoreResult, cands: list[Candidate] | None,
                   shown_n: dict[int, int] | None = None) -> None:
        cands = self.candidates if cands is None else cands
        for i in np.flatnonzero(res.available):
            i = int(i)
            n = (shown_n or {}).get(i, len(cands))
            by_id = {c.id: float(res.llr[i, c.index]) for c in cands}
            notes = {c.id: res.notes[(i, c.index)] for c in cands if (i, c.index) in res.notes}
            self._llm_rows[i] = (by_id, floor_evidence(n), notes)
        self._evidence_changed()

    def apply_llm(self, llm: Scorer) -> Solution:
        """Score with `llm` (its own `send` set decides which files) disc by disc (see
        `llm_windows`) and re-solve. The other scorers' results are kept, not recomputed. A
        windowed file is flagged `disc_window_failed` when the LLM rated "none of these"
        highest in both orders, or when it is still below high after the re-solve."""
        sol = self.solution or self.solve()
        groups = self.llm_windows(sol)
        send0, shown0 = getattr(llm, "send", None), getattr(llm, "shown", None)
        targets = set(range(len(self.files))) if send0 is None else set(send0)
        cands = list(self.candidates)
        F, E = len(self.files), len(cands)
        merged = ScoreResult.empty(F, E)
        shown_n: dict[int, int] = {}
        windowed: set[int] = set()
        try:
            for rows, cols in groups:
                send = targets & rows
                if not send:
                    continue
                llm.send = send  # type: ignore[attr-defined]
                llm.shown = shown0 if cols is None else set(cols)  # type: ignore[attr-defined]
                if cols is not None and not cols:
                    continue
                r = llm.score(self.files, self.context(cands))
                for i in send:
                    if r.available[i]:
                        merged.available[i] = True
                        merged.llr[i] = r.llr[i]
                        for j in range(E):
                            if (i, j) in r.notes:
                                merged.notes[(i, j)] = r.notes[(i, j)]
                        shown_n[i] = len(cols) if cols is not None else (
                            E if shown0 is None else len(shown0)
                        )
                if cols is not None:
                    windowed |= send
        finally:
            llm.send, llm.shown = send0, shown0  # type: ignore[attr-defined]
        self._store_llm(llm, merged, cands, shown_n)
        failed_fn = getattr(llm, "disc_window_failed", None)
        failed = {i for i in windowed if failed_fn is not None and failed_fn(i)}
        sol = self.solve()
        failed |= {i for i in windowed if sol.confidences[i].level != "high"}
        self._window_failed = (self._window_failed - windowed) | failed
        return sol

    # ---- pins ------------------------------------------------------------------------------

    def _index(self, file: Path | str | int) -> int:
        if isinstance(file, int):
            return file
        p = Path(file)
        p = self.duplicates.get(p, p)
        for i, f in enumerate(self.rows):
            if f.path == p:
                return i
        raise KeyError(f"not an analyzed file: {file}")

    def _column(self, episode: str | int | Candidate) -> int:
        if isinstance(episode, Candidate):
            return episode.index
        if isinstance(episode, int):
            if not -1 <= episode < len(self.candidates):
                raise ValueError(f"no candidate column {episode}")
            return episode
        if episode.strip().lower() == "extra":
            return -1
        m = _LABEL_RE.match(episode.strip())
        for c in self.candidates:
            if c.id == episode or (m and (c.season, c.number) == (int(m[1]), int(m[2]))):
                return c.index
        raise ValueError(f"{episode} is not a candidate in this scope")

    def pin(self, file: Path | str | int, episode: str | int | Candidate) -> Solution:
        """Fix `file` to an episode (id, `S01E05` label in the ordering in use, candidate index
        or Candidate) or to "extra", then re-solve. An episode can be pinned to one file only;
        named files are already pins."""
        i = self._index(file)
        if i >= len(self.files):
            raise ValueError(f"{self.rows[i].path.name} is named as an episode: already a pin")
        col = self._column(episode)
        path = self.files[i].path
        eid = None if col < 0 else self.candidates[col].id
        if eid is not None:
            for other, e in self.pins.items():
                if e == eid and other != path:
                    raise ValueError(f"{self.candidates[col].label} is already pinned to {other}")
            for n in self.named_rows:
                if n.episode_id == eid:
                    raise ValueError(
                        f"{self.candidates[col].label} is held by the named file {n.info.path}"
                    )
        self.pins[path] = eid
        return self.solve()

    def unpin(self, file: Path | str | int) -> Solution:
        i = self._index(file)
        if i < len(self.files):
            self.pins.pop(self.files[i].path, None)
        return self.solve()

    def overridden(self) -> dict[Path, Decision | None]:
        """For each pinned file whose pin differs from what a solve without that one pin picks,
        that decision (§12.4). Pins that agree with the tool are left out. Session state (pins,
        last solution, ordering choice and caches) is unchanged afterwards."""
        if not self.pins:
            return {}
        saved = (dict(self.pins), self.solution, self._choice, self.ordering, self.candidates,
                 dict(self._evid), dict(self._oos), set(self._window_failed))
        out: dict[Path, Decision | None] = {}
        try:
            for path, eid in saved[0].items():
                self.pins = {p: e for p, e in saved[0].items() if p != path}
                sol = self.solve()
                i = self._index(path)
                got = _decision(sol.assignment.column[i], sol.candidates)
                same = (got.kind == "extra") if eid is None else (
                    got.episode is not None and got.episode.episode_id == eid)
                if not same:
                    out[path] = got
        finally:
            (self.pins, self.solution, self._choice, self.ordering, self.candidates,
             self._evid, self._oos, self._window_failed) = saved
        return out

    def explain(self, file: Path | str | int, n: int = 3) -> FileDetail:
        """The evidence pane for one file (§12.4): its top `n` candidates by fused score with
        odds against the best, each one's signal contributions, notes and runtime. Reads the
        last solution (solving first when there is none); no new scoring. Named and side rows
        have no evidence: ValueError, as for `pin`."""
        try:
            i = self._index(file)
        except KeyError:
            p = Path(file) if not isinstance(file, int) else None
            if p is not None and (p in self.named or p in self.not_ready or p in self.errors
                                  or p in self.confirmed_extras):
                raise ValueError(f"{p.name} has no evidence to explain") from None
            raise
        if i >= len(self.files):
            raise ValueError(f"{self.rows[i].path.name} is named as an episode: no evidence")
        sol = self.solution or self.solve()
        f = sol.files[i]
        regime = sol.fused.regimes[i]
        row = sol.fused.S[i]
        top = sorted(range(len(sol.candidates)), key=lambda j: (-row[j], j))[:n]
        best = float(row[top[0]]) if top else 0.0
        cands = []
        for j in top:
            c = sol.candidates[j]
            notes: list[str] = []
            for name in REGIME_SCORERS[regime]:
                res = sol.results.get(name)
                if res is not None and (i, j) in res.notes:
                    notes += note_lines(name, res.notes[(i, j)])
            syn = next((s.text for s in c.episode.synopses if s.text), None)
            delta = best - float(row[j])
            cands.append(CandidateDetail(
                episode_id=c.id, label=c.label, title=c.episode.title, score=float(row[j]),
                delta=delta, odds=format_odds(delta),
                signals=_signals(i, regime, sol.fused, j, None, False), notes=notes,
                synopsis=syn, runtime_min=c.episode.runtime_min,
            ))
        return FileDetail(path=str(f.path), duration_s=f.duration_s, regime=regime,
                          candidates=cands)

    # ---- report ----------------------------------------------------------------------------

    @property
    def show_slug(self) -> str:
        if self.bundle_dir is not None:
            return self.bundle_dir.name
        return slugify(self.bundle.show.title, self.bundle.show.year)

    def flags(self, i: int) -> list[str]:
        sol = self.solution
        out: list[str] = []
        if i >= len(self.files):
            k = i - len(self.files)
            rows = self.named_rows
            return [FLAG_CONFIRMED if k < len(rows) and rows[k].confirmed else FLAG_NAMED]
        f = self.files[i]
        if f.dialogue is None or f.dialogue.track is None:
            out.append(FLAG_NO_DIALOGUE)
        if sol is not None and i in sol.pins:
            out.append(FLAG_PINNED)
        if sol is not None and i in sol.out_of_scope:
            out.append(FLAG_OUT_OF_SCOPE)
        if i in self._window_failed:
            out.append(FLAG_DISC_WINDOW_FAILED)
        if sol is not None and i < len(sol.files) and i not in sol.pins:
            col = sol.assignment.column[i]
            if play_all(i, f.duration_s, sol.candidates, sol.results):
                out.append(FLAG_PLAY_ALL)
            if possible_two_parter(i, col, f.duration_s, sol.candidates, sol.fused):
                out.append(FLAG_POSSIBLE_TWO_PARTER)
            if i in self._history_dups(sol):
                out.append(FLAG_HISTORY_DUPLICATE)
        return out

    def _history_dups(self, sol: Solution) -> dict[int, tuple[str, Match]]:
        """Row -> (episode label, a match) for rows whose chosen episode is already confirmed
        for a different fingerprint (§11.4 duplicate warning)."""
        out: dict[int, tuple[str, Match]] = {}
        if self.history is None:
            return out
        for i, f in enumerate(sol.files[: len(self.files)]):
            col = sol.assignment.column[i]
            if col < 0 or i in sol.pins:
                continue
            cand = sol.candidates[col]
            other = [m for m in self.history.confirmed_for_episode(self.show_slug, cand.id)
                     if m.fingerprint != f.fingerprint]
            if other:
                out[i] = (cand.label, other[-1])
        return out

    def _history_notices(self, sol: Solution | None) -> list[str]:
        out = []
        if sol is None or self.history is None:
            return out
        for i, (label, m) in self._history_dups(sol).items():
            mf = self.history.get_file(m.fingerprint)
            where = mf.last_path if mf is not None else "a file no longer known"
            out.append(
                f"{sol.files[i].path.name}: {label} is already confirmed for {where} "
                "(duplicate, or a re-rip to review)"
            )
        return out

    def report(self, solution: Solution | None = None) -> Report:
        """Report over every file the session saw: solved rows, duplicates (repeating their
        primary's decision, flagged `duplicate`, §14), named files (`named`), and files not
        ready (`not_ready`, reported as low-confidence extras: not analyzed) and unreadable
        files (`unreadable`, low confidence with the reason in a notice: they need review).
        With no row to solve (every file named, not ready or unreadable) nothing is solved."""
        if solution is None and self.solution is None and not self.rows:
            return self._unsolved_report()
        sol = solution or self.solution or self.solve()
        R = len(sol.files)
        files = list(sol.files)
        rows = list(range(R))
        flags = {i: self.flags(i) for i in range(R)}
        own: set[int] = set()  # near duplicates: their own duration and dialogue
        for dup, primary in self.duplicates.items():
            i = self._index(primary)
            flags[len(files)] = [
                *self._dup_flags(dup), *(fl for fl in flags[i] if fl != FLAG_PINNED)
            ]
            if dup in self.near:
                own.add(len(files))
            files.append(self.dup_infos[dup])
            rows.append(i)
        # exact duplicates keep their own path/fingerprint/disc, but duration comes from the
        # primary (they weren't probed)
        for k in range(R, len(files)):
            if k in own:
                continue
            p = sol.files[rows[k]]
            files[k] = FileInfo(
                path=files[k].path,
                fingerprint=files[k].fingerprint,
                duration_s=p.duration_s,
                title_tag=p.title_tag,
                disc=files[k].disc,
                dialogue=p.dialogue,
                ocr_quality=p.ocr_quality,
            )
        idx = np.asarray(rows, dtype=int)
        fused = Fused(
            S=sol.fused.S[idx],
            contributions={k: v[idx] for k, v in sol.fused.contributions.items()},
            regimes=[sol.fused.regimes[r] for r in rows],
            log_prior=sol.fused.log_prior,
            available={k: v[idx] for k, v in sol.fused.available.items()},
        )
        assignment = Assignment([sol.assignment.column[r] for r in rows], sol.assignment.total)
        confs = [sol.confidences[r] for r in rows]
        rep = build_report(
            files,
            sol.candidates,
            fused,
            assignment,
            confs,
            show_slug=self.show_slug,
            bundle_revision=self.bundle.bundle_revision,
            ordering=sol.ordering,
            season=self.season,
            flags=flags,
        )
        return self._finish(rep, [*rep.notices, *self._notices(sol)])

    def _dup_flags(self, dup: Path) -> list[str]:
        """`duplicate` for an exact copy; a near duplicate adds `near_duplicate`, and with
        --allow-duplicates its `duplicate` becomes `duplicate_allowed` (another version, §4.3).
        An exact copy of a near duplicate is a plain duplicate of that version."""
        if dup in self.near:
            first = FLAG_DUPLICATE_ALLOWED if self.allow_duplicates else FLAG_DUPLICATE
            return [first, FLAG_NEAR_DUPLICATE]
        if self._exact.get(dup) in self.near:
            return [FLAG_DUPLICATE, FLAG_NEAR_DUPLICATE]
        return [FLAG_DUPLICATE]

    def _near_notices(self, sol: Solution | None) -> list[str]:
        out = []
        for dup, (primary, ov) in self.near.items():
            head = (f"{dup.name} looks like a copy of {primary.name} "
                    f"(dialogue overlap {round(100 * ov)}%)")
            what = "the same extra"
            if sol is not None:
                col = sol.assignment.column[self._index(primary)]
                if col >= 0:
                    what = sol.candidates[col].label
            if self.allow_duplicates:
                out.append(f"{head}: kept as another version of {what} (--allow-duplicates)")
            else:
                out.append(
                    f"{head}: a duplicate of {what} to review; a plan leaves it in its folder and "
                    "renames it <episode>_dupe_N (keep it as a version with --allow-duplicates)"
                )
        return out

    def _unsolved_report(self) -> Report:
        rep = Report(
            tool_version=report_version(),
            show_slug=self.show_slug,
            bundle_revision=self.bundle.bundle_revision,
            ordering=self.ordering,
            season=self.season,
            summary=Summary(files=0, high=0, medium=0, low=0, extras=0),
        )
        return self._finish(rep, self._notices(None))

    def _finish(self, rep: Report, notices: list[str]) -> Report:
        """Adds the rows that are not part of the solve (named side rows, not-ready and
        unreadable files) and recounts the summary."""
        extra_rows = [self._side_row(n) for n in self.named.values() if not n.row]
        extra_rows += [
            self._plain_row(info, "extra", None, [FLAG_NOT_READY])
            for info in self.not_ready.values()
        ]
        extra_rows += [
            self._plain_row(
                FileInfo(path=path, fingerprint="", duration_s=0.0, disc=self._disc_name(path)),
                "extra", None, [FLAG_UNREADABLE],
            )
            for path in self.errors
        ]
        extra_rows += [
            self._plain_row(info, "extra", None, [FLAG_CONFIRMED], high=True)
            for info in self.confirmed_extras.values()
        ]
        dup = self._duplicate_ranks()
        explicit = {p for p, (_, n) in self._node.items() if n.explicit}
        rep = rep.model_copy(update={"files": [
            r.model_copy(update={"explicit": True}) if Path(r.path) in explicit else r
            for r in rep.files]})
        all_rows = [
            r.model_copy(update={"duplicate_of": str(dup[r.path][0]),
                                 "duplicate_rank": dup[r.path][1]})
            if r.path in dup else r
            for r in [*rep.files, *extra_rows]
        ]
        all_rows.sort(key=lambda r: (r.disc is None, r.disc or "", r.path))
        return rep.model_copy(
            update={
                "files": all_rows,
                "summary": Summary(
                    files=len(all_rows),
                    high=sum(r.confidence.level == "high" for r in all_rows),
                    medium=sum(r.confidence.level == "medium" for r in all_rows),
                    low=sum(r.confidence.level == "low" for r in all_rows),
                    extras=sum(r.decision.kind == "extra" for r in all_rows),
                ),
                "notices": notices,
            }
        )

    def _duplicate_ranks(self) -> dict[str, tuple[Path, int]]:
        """Report path -> (primary, 0-based rank among that primary's copies), exact and near
        duplicates (decision 56). Ranked like `choose_primary`: most usable dialogue, then the
        larger file, then path. An exact copy ranks with the quality of the file it copies."""
        groups: dict[Path, list[Path]] = {}
        for d, p in self.duplicates.items():
            groups.setdefault(p, []).append(d)

        def key(d: Path):
            q = self._quality.get(d) or self._quality.get(self._exact.get(d, d)) or Quality()
            return (-q.usable_lines, -q.size, d)

        out: dict[str, tuple[Path, int]] = {}
        for p, copies in groups.items():
            for rank, d in enumerate(sorted(copies, key=key)):
                out[str(d)] = (p, rank)
        return out

    def _notices(self, sol: Solution | None) -> list[str]:
        out = [] if sol is None else list(sol.notices)
        for path, err in self.errors.items():
            out.append(f"{path.name}: unreadable ({err}); it needs review")
        for i, c in ({} if sol is None else sol.out_of_scope).items():
            out.append(
                f"{sol.files[i].path.name}: evidence points to {c.label} ({c.episode.title}), "
                "outside this season; it was not moved there (pin it to change season)"
            )
        out += self._near_notices(sol)
        out += self._history_notices(sol)
        waiting: dict[str, int] = {}
        for info in self.not_ready.values():
            key = info.disc or "(no disc)"
            waiting[key] = waiting.get(key, 0) + 1
        for disc, n in waiting.items():
            out.append(f"{disc}: {n} file{'s' if n > 1 else ''} still being written")
        out += self.notices
        return out

    def _plain_row(
        self, info: FileInfo, kind: str, ref: EpisodeRef | None, flags: list[str],
        high: bool = False,
    ) -> FileReport:
        return FileReport(
            path=str(info.path),
            fingerprint=info.fingerprint,
            disc=info.disc,
            duration_s=info.duration_s,
            regime="base",
            decision=Decision(kind=kind, episode=ref),  # type: ignore[arg-type]
            confidence=Confidence(
                level="high" if high else "low",
                delta=None if high else 0.0,
                odds=format_odds(None if high else 0.0),
            ),
            flags=flags,
        )

    def _side_row(self, n: _Named) -> FileReport:
        """A named file that is not a row: unresolved, out of season, or a second copy."""
        flags = [FLAG_NAMED]
        if n.episode_id is None:
            return self._plain_row(n.info, "extra", None, flags)
        order = self.bundle.orderings[self._named_order()]
        ent = next(e for e in order.entries if e.episode_id == n.episode_id)
        ep = self.bundle.episode_by_id()[n.episode_id]
        if n.reason and "outside" in n.reason:
            flags.append(FLAG_OUT_OF_SCOPE)
        elif n.reason:
            flags.append(FLAG_DUPLICATE)
        ref = EpisodeRef(episode_id=ep.id, season=ent.season, episode=ent.episode,
                         title=ep.title)
        return self._plain_row(n.info, "episode", ref, flags, high=True)


def record_overrides(hist, sess: Session, run_id: int | None, show_slug: str) -> int:
    """Record what the tool would have chosen for each pin that overrides it as a `rejected`
    match in `hist` (§7.2, decision 61); returns how many. A pin equal to the tool's own top
    choice records nothing. Nothing is recorded when the session has no pins."""
    n = 0
    files = {f.path: f for f in sess.files}
    for path, d in sess.overridden().items():
        if d is None or path not in files:
            continue
        ep = d.episode
        f = files[path]
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        hist.touch_file(f.fingerprint, size=size, path=str(path))  # the match row links to it
        hist.reject(
            f.fingerprint, show_slug, episode_id=ep.episode_id if ep else None, run_id=run_id,
            season=ep.season if ep else None, episode=ep.episode if ep else None,
            title=ep.title if ep else None,
        )
        n += 1
    return n


__all__ = [
    "record_overrides",
    "FLAG_CONFIRMED",
    "FLAG_DISC_WINDOW_FAILED",
    "FLAG_DUPLICATE",
    "FLAG_DUPLICATE_ALLOWED",
    "FLAG_HISTORY_DUPLICATE",
    "FLAG_NAMED",
    "FLAG_NEAR_DUPLICATE",
    "FLAG_NOT_READY",
    "FLAG_NO_DIALOGUE",
    "FLAG_PLAY_ALL",
    "FLAG_POSSIBLE_TWO_PARTER",
    "FLAG_OUT_OF_SCOPE",
    "FLAG_PINNED",
    "FLAG_UNREADABLE",
    "Session",
    "Solution",
    "bundle_vocabulary",
    "default_scorers",
    "disc_label",
    "disc_of",
    "ordering_key",
]
