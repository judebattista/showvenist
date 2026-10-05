"""Engine API used by the TUI and report mode (spec §12.1). Flat-list scope for now (§11.5 basic
assignment; no disc blocks, no ordering detection, no history prior).

    s = Session(bundle, bundle_dir, config, ordering="aired", season=1, cache=cache)
    s.analyze(paths)          # stage 1 + 2, cached; exact duplicates set aside (§14)
    s.score()                 # raw evidence per scorer
    sol = s.solve()           # fusion + basic assignment + confidence
    rep = s.report()          # report_schema.Report
    s.pin(path, "S01E03")     # or "extra"; re-solves
    s.unpin(path)

No matching logic lives outside this module and the modules it calls.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from showvenist.analysis.probe import probe
from showvenist.analysis.subtitles import normalize
from showvenist.analysis.types import AnalysisCache
from showvenist.analysis.windows import extract_dialogue
from showvenist.assign.basic import NEG, assign_basic
from showvenist.assign.confidence import confidence
from showvenist.bundle.schema import Bundle, lang_codes
from showvenist.bundle.store import slugify
from showvenist.config import Config
from showvenist.fusion import fuse
from showvenist.report import build_report
from showvenist.report_schema import Report
from showvenist.scoring.candidates import build_candidates
from showvenist.scoring.characters import GuestScorer
from showvenist.scoring.cues import CuesScorer
from showvenist.scoring.duration import DurationScorer
from showvenist.scoring.refsubs import RefsubsScorer
from showvenist.scoring.synopsis import SynopsisScorer
from showvenist.scoring.title import TitleScorer
from showvenist.scoring.types import (
    Assignment,
    Candidate,
    FileConfidence,
    FileInfo,
    Fused,
    Regime,
    ScoreContext,
    Scorer,
    ScoreResult,
)
from showvenist.state.fingerprint import content_key, fingerprint

# progress(done, total, path, error): called once per input path after it is analyzed
Progress = Callable[[int, int, Path, str | None], None]

PIN_BONUS = 1e6  # added to a pinned (file, episode) cell; far above any evidence, far below NEG
_DISC_RE = re.compile(r"(^|[\s._-])(disc|disk|d)[\s._-]*\d+$", re.IGNORECASE)
_LABEL_RE = re.compile(r"^s(\d{1,3})e(\d{1,4})$", re.IGNORECASE)

FLAG_DUPLICATE = "duplicate"
FLAG_NO_DIALOGUE = "no_dialogue"  # no usable subtitle track: dialogue scorers not available
FLAG_PINNED = "pinned"


def default_scorers() -> list[Scorer]:
    return [
        DurationScorer(),
        TitleScorer(),
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
    """One solve over the session's primary (non-duplicate) files, in `files` order.

    `S` is the matrix actually solved: `fused.S` with pins applied. `results` are the raw
    per-scorer results (unclamped, unweighted), so eval and the weight fit can read them.
    """

    files: list[FileInfo]
    candidates: list[Candidate]
    results: dict[str, ScoreResult]
    fused: Fused
    S: np.ndarray
    assignment: Assignment
    confidences: list[FileConfidence]
    pins: dict[int, int] = field(default_factory=dict)  # file index -> column (-1 = extra)

    @property
    def regimes(self) -> list[Regime]:
        return self.fused.regimes


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
    ):
        self.bundle = bundle
        self.bundle_dir = Path(bundle_dir) if bundle_dir is not None else None
        self.config = config
        self.ordering = ordering
        self.season = season
        self.cache = cache
        self.candidates = build_candidates(
            bundle,
            ordering,
            season=season,
            episodes=None if episodes is None else set(episodes),
            include_specials=include_specials,
        )
        if not self.candidates:
            raise ValueError(
                f"no candidate episodes in ordering {ordering!r}"
                + (f" for season {season}" if season is not None else "")
            )
        self.scorers: list[Scorer] = list(scorers) if scorers is not None else default_scorers()
        if llm is not None:
            self.scorers.append(llm)  # the LLM scorer only runs when a client is injected
        self.language = bundle.show.language
        self._vocab = vocabulary
        self.files: list[FileInfo] = []  # primary files, scored and solved
        self.duplicates: dict[Path, Path] = {}  # duplicate path -> primary path (§14)
        self.dup_infos: dict[Path, FileInfo] = {}
        self.errors: dict[Path, str] = {}  # unreadable files: path -> reason
        self._keys: dict[str, Path] = {}  # content key -> primary path
        self.results: dict[str, ScoreResult] | None = None
        self.pins: dict[Path, int] = {}  # primary path -> column (-1 = extra)
        self.solution: Solution | None = None

    # ---- analysis ------------------------------------------------------------------------

    @property
    def vocabulary(self) -> set[str]:
        if self._vocab is None:
            self._vocab = bundle_vocabulary(self.bundle, self.bundle_dir)
        return self._vocab

    def _analyze_one(self, path: Path) -> FileInfo:
        info = probe(path, self.cache)
        d = extract_dialogue(
            path,
            self.language,
            self.vocabulary,
            self.config.analysis,
            self.cache,
            probe_result=info,
        )
        return FileInfo(
            path=path,
            fingerprint=fingerprint(path),
            duration_s=info.duration_s,
            title_tag=info.title_tag,
            disc=disc_of(path),
            dialogue=d,
            ocr_quality=d.ocr_quality,
        )

    def analyze(
        self, paths: Iterable[Path | str], progress: Progress | None = None
    ) -> list[FileInfo]:
        """Stage 1 + 2 for each path (cached through the AnalysisCache). Exact duplicates (same
        content key, §7.1) are recorded against the first copy and not analyzed again.
        Unreadable files land in `errors` and are skipped. Returns the primary files."""
        paths = [Path(p) for p in paths]
        by_key = self._keys
        for n, path in enumerate(paths, 1):
            err: str | None = None
            try:
                ckey = content_key(path)
                if ckey in by_key and by_key[ckey] != path:
                    self.duplicates[path] = by_key[ckey]
                    self.dup_infos[path] = FileInfo(
                        path=path,
                        fingerprint=fingerprint(path),
                        duration_s=0.0,
                        disc=disc_of(path),
                    )
                elif ckey not in by_key:
                    self.files.append(self._analyze_one(path))
                    by_key[ckey] = path
            except Exception as e:  # unreadable/corrupt media: record, keep going
                err = f"{type(e).__name__}: {e}"
                self.errors[path] = err
            if progress is not None:
                progress(n, len(paths), path, err)
        self.results = None
        self.solution = None
        return list(self.files)

    # ---- scoring and solving ---------------------------------------------------------------

    def context(self) -> ScoreContext:
        return ScoreContext(
            bundle=self.bundle,
            candidates=self.candidates,
            config=self.config,
            bundle_dir=self.bundle_dir,
            language=self.language,
            vocabulary=self.vocabulary,
        )

    def score(self) -> dict[str, ScoreResult]:
        ctx = self.context()
        self.results = {s.name: s.score(self.files, ctx) for s in self.scorers}
        return self.results

    def _index(self, file: Path | str | int) -> int:
        if isinstance(file, int):
            return file
        p = Path(file)
        p = self.duplicates.get(p, p)
        for i, f in enumerate(self.files):
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

    def solve(self) -> Solution:
        """Fusion (config weights, history prior off) + basic assignment + confidence, with
        pins applied as hard constraints."""
        if self.results is None:
            self.score()
        assert self.results is not None
        E = len(self.candidates)
        if self.results:
            fused = fuse(self.results, self.config.weights)
        else:  # no scorers: prior only
            fused = fuse({}, self.config.weights, history_penalty=np.zeros((len(self.files), E)))
        S = fused.S.copy()
        pins: dict[int, int] = {}
        for path, col in self.pins.items():
            i = self._index(path)
            pins[i] = col
            if col == -1:
                S[i, :] = NEG  # only its unmatched column is left
            else:
                S[:, col] = NEG  # nobody else may take the episode
                S[i, :] = NEG
                S[i, col] = fused.S[i, col] + PIN_BONUS
        assignment = assign_basic(S)
        confs = confidence(S, assignment)
        for i in pins:
            confs[i] = FileConfidence(math.inf, "high", None, None)
        self.solution = Solution(
            files=list(self.files),
            candidates=self.candidates,
            results=self.results,
            fused=fused,
            S=S,
            assignment=assignment,
            confidences=confs,
            pins=pins,
        )
        return self.solution

    def pin(self, file: Path | str | int, episode: str | int | Candidate) -> Solution:
        """Fix `file` to an episode (id, `S01E05` label in this ordering, candidate index or
        Candidate) or to "extra", then re-solve. An episode can be pinned to one file only."""
        i = self._index(file)
        col = self._column(episode)
        path = self.files[i].path
        if col >= 0:
            for other, c in self.pins.items():
                if c == col and other != path:
                    raise ValueError(f"{self.candidates[col].label} is already pinned to {other}")
        self.pins[path] = col
        return self.solve()

    def unpin(self, file: Path | str | int) -> Solution:
        self.pins.pop(self.files[self._index(file)].path, None)
        return self.solve()

    # ---- report ----------------------------------------------------------------------------

    @property
    def show_slug(self) -> str:
        if self.bundle_dir is not None:
            return self.bundle_dir.name
        return slugify(self.bundle.show.title, self.bundle.show.year)

    def flags(self, i: int) -> list[str]:
        f = self.files[i]
        out: list[str] = []
        if f.dialogue is None or f.dialogue.track is None:
            out.append(FLAG_NO_DIALOGUE)
        if self.solution is not None and i in self.solution.pins:
            out.append(FLAG_PINNED)
        return out

    def report(self, solution: Solution | None = None) -> Report:
        """Report over every analyzed file; duplicates repeat their primary's decision and
        carry the `duplicate` flag (§14). Unreadable files are not in the report (see
        `errors`)."""
        sol = solution or self.solution or self.solve()
        F = len(sol.files)
        files = list(sol.files)
        rows = list(range(F))
        flags = {i: self.flags(i) for i in range(F)}
        for dup, primary in self.duplicates.items():
            i = self._index(primary)
            flags[len(files)] = [FLAG_DUPLICATE, *(fl for fl in flags[i] if fl != FLAG_PINNED)]
            files.append(self.dup_infos[dup])
            rows.append(i)
        # duplicates keep their own path/fingerprint, but duration comes from the primary
        for k in range(F, len(files)):
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
        )
        assignment = Assignment([sol.assignment.column[r] for r in rows], sol.assignment.total)
        confs = [sol.confidences[r] for r in rows]
        return build_report(
            files,
            sol.candidates,
            fused,
            assignment,
            confs,
            show_slug=self.show_slug,
            bundle_revision=self.bundle.bundle_revision,
            ordering=self.ordering,
            season=self.season,
            flags=flags,
        )
