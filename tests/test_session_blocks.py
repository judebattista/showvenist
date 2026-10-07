"""Session with a Show/Season/Disc tree: disc blocks, ordering detection, scope, named and
not-ready files, the per-disc LLM window, and pin/unpin speed (spec §5, §10.2, §11.3-11.7).

Tiny synthetic MKVs only (no real media, no network; the LLM is a fake client)."""

from __future__ import annotations

import random
import re
import shutil
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from showvenist.analysis.types import AnalysisCache
from showvenist.bundle.schema import (
    Bundle,
    Episode,
    Ordering,
    OrderingEntry,
    ShowInfo,
    SubtitleRef,
    Synopsis,
)
from showvenist.config import Config, HierarchyConfig
from showvenist.hierarchy.parse import build_tree
from showvenist.report import exit_code
from showvenist.scoring.types import ScoreResult
from showvenist.session import (
    FLAG_DISC_WINDOW_FAILED,
    FLAG_DUPLICATE,
    FLAG_NAMED,
    FLAG_NOT_READY,
    FLAG_OUT_OF_SCOPE,
    Session,
)
from tests.fixtures.textsubs import make_srt
from tests.test_session import FILLER, dialogue_for, make_episode_mkv, synth_bundle, write_refs

# ---- helpers ----------------------------------------------------------------------------------


class TableScorer:
    """Controlled evidence: file name -> {episode id: LLR}. Named like a real scorer so fusion
    sums it; files missing from the table have no observation."""

    name = "refsubs"
    can_be_negative = True

    def __init__(self, table: dict[str, dict[str, float]]):
        self.table = table
        self.calls = 0

    def score(self, files, ctx) -> ScoreResult:
        self.calls += 1
        out = ScoreResult.empty(len(files), len(ctx.candidates))
        for i, f in enumerate(files):
            row = self.table.get(f.path.name)
            if row is None:
                continue
            out.available[i] = True
            for c in ctx.candidates:
                out.llr[i, c.index] = row.get(c.id, 0.0)
        return out


def plain_bundle(n: int, orderings: dict[str, list[int]] | None = None,
                 names: dict[str, str] | None = None) -> Bundle:
    """Episodes e1..en; `orderings` maps an ordering name to episode numbers in its order
    (all in season 1). 'aired' is e1..en."""
    eps = [Episode(id=f"e{k}", title=f"Story {k}") for k in range(1, n + 1)]
    orders = {"aired": list(range(1, n + 1)), **(orderings or {})}
    return Bundle(
        show=ShowInfo(title="Synth Show", year=2000),
        episodes=eps,
        orderings={
            name: Ordering(
                source="tmdb" if name == "aired" else "custom",
                type="aired" if name == "aired" else name,
                name=(names or {}).get(name),
                entries=[OrderingEntry(season=1, episode=p + 1, episode_id=f"e{k}")
                         for p, k in enumerate(seq)],
            )
            for name, seq in orders.items()
        },
    ).with_revision()


def make_season(root: Path, layout: dict[str, list[str]]) -> Path:
    """root/Synth Show/Season 1/<disc>/<file>: each file a tiny MKV whose dialogue names it
    (so the fake LLM can recognize it). Returns the season folder."""
    season = root / "Synth Show" / "Season 1"
    for disc, names in layout.items():
        for name in names:
            tok = name.removesuffix(".mkv")
            lines = [f"token {tok} speaks", *(" ".join(random.Random(tok).choices(FILLER, k=6))
                                               for _ in range(5))]
            make_episode_mkv(season / disc / name, 2, lines)
    return season


def tree_of(season: Path):
    return build_tree(season, config=HierarchyConfig(), tag_of=lambda p: None)


def session_for(season: Path, bundle: Bundle, table, tmp_path: Path, **kw) -> Session:
    s = Session(bundle, None, kw.pop("config", Config()), season=1,
                cache=AnalysisCache(tmp_path / "analysis.db"), scorers=[TableScorer(table)],
                vocabulary=set(), tree=tree_of(season), **kw)
    s.analyze()
    return s


def assigned(sol, name: str) -> str:
    i = next(k for k, f in enumerate(sol.files) if f.path.name == name)
    col = sol.assignment.column[i]
    return "extra" if col < 0 else sol.candidates[col].id


def by_name(rep):
    return {Path(r.path).name: r for r in rep.files}


STRONG = 8.0
# 3 discs, 9 episodes in release order; e7 was not ripped (MakeMKV minimum length) and c3 is
# an extra. a3 is really e3, but its evidence leans to e7 (weak and wrong).
LAYOUT = {
    "Disc 1": ["a1.mkv", "a2.mkv", "a3.mkv"],
    "Disc 2": ["b1.mkv", "b2.mkv", "b3.mkv"],
    "Disc 3": ["c1.mkv", "c2.mkv", "c3.mkv"],
}
TABLE = {
    "a1.mkv": {"e1": STRONG}, "a2.mkv": {"e2": STRONG}, "a3.mkv": {"e3": 3.0, "e7": 3.5},
    "b1.mkv": {"e4": STRONG}, "b2.mkv": {"e5": STRONG}, "b3.mkv": {"e6": STRONG},
    "c1.mkv": {"e8": STRONG}, "c2.mkv": {"e9": STRONG},
    # an extra: rejected by every episode (as the duration scorer does for short titles). With
    # no evidence at all, S = prior ≈ −0.46 at E = 9 is less than u = 1, and the block DP
    # would rather fill the unclaimed e7 slot with it.
    "c3.mkv": {f"e{k}": -3.0 for k in range(1, 10)},
}


@pytest.fixture(scope="module")
def season3(tmp_path_factory) -> Path:
    return make_season(tmp_path_factory.mktemp("rips"), LAYOUT)


# ---- block structure ----------------------------------------------------------------------------


def test_blocks_fix_weak_evidence_where_basic_fails(season3, tmp_path):
    blocks = session_for(season3, plain_bundle(9), TABLE, tmp_path)
    flat = session_for(season3, plain_bundle(9), TABLE, tmp_path, use_blocks=False)
    sb, sf = blocks.solve(), flat.solve()
    assert sf.used_blocks is False and sb.used_blocks is True
    assert assigned(sf, "a3.mkv") == "e7"  # basic: the weak, wrong evidence wins
    assert assigned(sb, "a3.mkv") == "e3"  # blocks: disc 1 can only hold e1..e3
    assert sb.blocks["Disc 1"] == (0, 3) and sb.blocks["Disc 2"] == (3, 6)
    assert assigned(sb, "c3.mkv") == "extra"
    for name, eid in [("a1.mkv", "e1"), ("b2.mkv", "e5"), ("c2.mkv", "e9")]:
        assert assigned(sb, name) == eid == assigned(sf, name)
    assert sb.ordering == "aired" and not sb.switched and not sb.fits_nowhere
    assert [d.key for d in sb.discs] == ["Disc 1", "Disc 2", "Disc 3"]
    rep = blocks.report()
    assert by_name(rep)["a3.mkv"].disc == "Disc 1"
    assert by_name(rep)["a3.mkv"].decision.episode.episode == 3


def test_extras_and_duplicate_inflate_disc_count(tmp_path):
    layout = {
        "Disc 1": ["a1.mkv", "a2.mkv", "a3.mkv"],
        "Disc 2": ["b1.mkv", "b2.mkv", "b3.mkv", "bx.mkv", "by.mkv"],  # 2 extras
        "Disc 3": ["c1.mkv", "c2.mkv", "c3.mkv"],
    }
    season = make_season(tmp_path / "rips", layout)
    shutil.copyfile(season / "Disc 2" / "b2.mkv", season / "Disc 2" / "b2copy.mkv")
    table = {**TABLE, "a3.mkv": {"e3": STRONG}, "c1.mkv": {"e7": STRONG}}  # e8 not ripped
    s = session_for(season, plain_bundle(9), table, tmp_path)
    sol = s.solve()
    # six files on disc 2, three episodes: the block is found from content, not counts
    assert sol.blocks["Disc 2"] == (3, 6)
    assert sol.blocks["Disc 3"] == (6, 9)
    assert {assigned(sol, n) for n in ("bx.mkv", "by.mkv")} == {"extra"}
    assert [assigned(sol, n) for n in ("c1.mkv", "c2.mkv", "c3.mkv")] == ["e7", "e9", "extra"]
    rep = by_name(s.report())
    assert FLAG_DUPLICATE in rep["b2copy.mkv"].flags
    assert rep["b2copy.mkv"].decision.episode.episode_id == "e5"
    assert rep["b2copy.mkv"].disc == "Disc 2"


def test_not_ready_file_skipped_and_disc_incomplete(season3, tmp_path):
    tree = tree_of(season3)
    node = next(f for f in tree.all_files() if f.path.name == "c2.mkv")
    node.ready, node.not_ready_reason = False, "still growing"
    s = Session(plain_bundle(9), None, Config(), season=1, scorers=[TableScorer(TABLE)],
                vocabulary=set(), tree=tree)
    seen = []
    s.analyze(None, progress=lambda n, t, p, e: seen.append((n, t, p.name)))
    assert "c2.mkv" not in {f.path.name for f in s.files}
    assert len(seen) == 9 and seen[-1][0] == 9 and {x[1] for x in seen} == {9}
    sol = s.solve()
    disc3 = next(d for d in sol.discs if d.key == "Disc 3")
    assert disc3.incomplete and len(disc3.files) == 2
    assert assigned(sol, "c1.mkv") == "e8"
    rep = s.report()
    row = by_name(rep)["c2.mkv"]
    assert FLAG_NOT_READY in row.flags and row.confidence.level == "low"
    assert any("Disc 3: 1 file still being written" in n for n in rep.notices)
    assert exit_code(rep) == 2
    assert rep.summary.files == 9


def test_named_file_is_a_pin_and_never_analyzed(tmp_path):
    layout = {k: list(v) for k, v in LAYOUT.items()}
    layout["Disc 2"].remove("b2.mkv")
    season = make_season(tmp_path / "rips", layout)
    named = season / "Disc 2" / "synth_s01e05.mkv"
    named.write_bytes(b"not a matroska file at all" * 50)  # analysis would fail on it
    # without the pin, b1's weak evidence would take e5
    table = {**TABLE, "b1.mkv": {"e4": 2.0, "e5": 2.6}}
    s = session_for(season, plain_bundle(9), table, tmp_path)
    assert named not in s.errors and named.name not in {f.path.name for f in s.files}
    sol = s.solve()
    assert assigned(sol, "synth_s01e05.mkv") == "e5"
    assert assigned(sol, "b1.mkv") == "e4"
    assert sol.blocks["Disc 2"] == (3, 6)
    row = by_name(s.report())["synth_s01e05.mkv"]
    assert FLAG_NAMED in row.flags and row.confidence.level == "high"
    assert row.decision.episode.episode == 5
    with pytest.raises(ValueError):
        s.pin(season / "Disc 2" / "b1.mkv", "S01E05")  # held by the named file
    with pytest.raises(ValueError):
        s.pin(named, "S01E06")  # a named file is already a pin


# ---- ordering detection (§11.6) ---------------------------------------------------------------

DVD = [1, 4, 2, 5, 3, 6]  # episode numbers in DVD order


@pytest.fixture(scope="module")
def dvd_season(tmp_path_factory) -> Path:
    return make_season(tmp_path_factory.mktemp("dvd"), {
        "Disc 1": ["x1.mkv", "x2.mkv", "x3.mkv"],
        "Disc 2": ["x4.mkv", "x5.mkv", "x6.mkv"],
    })


def test_ordering_switch_to_the_release_order(dvd_season, tmp_path):
    # the discs follow DVD order: disc 1 = e1, e4, e2
    table = {f"x{p + 1}.mkv": {f"e{k}": STRONG} for p, k in enumerate(DVD)}
    bundle = plain_bundle(6, {"dvd": DVD}, names={"dvd": "DVD Order"})
    s = session_for(dvd_season, bundle, table, tmp_path)
    sol = s.solve()
    assert sol.switched and sol.ordering == "dvd" and sol.used_blocks
    assert s.ordering == "dvd" and s.starting_ordering == "aired"
    assert sol.blocks == {"Disc 1": (0, 3), "Disc 2": (3, 6)}
    for p, k in enumerate(DVD):
        assert assigned(sol, f"x{p + 1}.mkv") == f"e{k}"
    rep = s.report()
    assert rep.ordering == "dvd"
    assert "This release follows 'DVD Order'." in rep.notices
    # pin/unpin keep the chosen ordering
    sol = s.pin(dvd_season / "Disc 1" / "x2.mkv", "e4")
    assert sol.ordering == "dvd" and sol.switched


def test_no_ordering_fits_falls_back_to_basic(dvd_season, tmp_path):
    table = {f"x{p + 1}.mkv": {f"e{k}": STRONG} for p, k in enumerate([1, 6, 3, 4, 2, 5])}
    s = session_for(dvd_season, plain_bundle(6, {"dvd": DVD}), table, tmp_path)
    sol = s.solve()
    assert sol.fits_nowhere and not sol.used_blocks and not sol.switched
    assert assigned(sol, "x2.mkv") == "e6"
    assert any("don't look sequential" in n for n in s.report().notices)


# ---- scope (§11.3) and --episodes ---------------------------------------------------------------


def test_out_of_scope_warning(tmp_path):
    bdir = tmp_path / "synth-show-2000"
    bundle = write_refs(synth_bundle(seasons=2), bdir)
    season = tmp_path / "rips" / "Synth Show" / "Season 1"
    make_episode_mkv(season / "Disc 1" / "t00.mkv", 8, dialogue_for(0))
    make_episode_mkv(season / "Disc 1" / "t01.mkv", 11, dialogue_for(1))
    make_episode_mkv(season / "Disc 1" / "t02.mkv", 14, dialogue_for(2, season=2))  # S02E03
    s = Session(bundle, bdir, Config(), season=1, cache=AnalysisCache(tmp_path / "a.db"),
                tree=tree_of(season))
    s.analyze()
    sol = s.solve()
    assert {c.season for c in sol.candidates} == {1}  # the folder's season is a hard filter
    assert assigned(sol, "t00.mkv") == "tmdb:100"
    i = next(k for k, f in enumerate(sol.files) if f.path.name == "t02.mkv")
    assert sol.out_of_scope[i].label == "S02E03"
    assert sol.assignment.column[i] == -1 or sol.candidates[sol.assignment.column[i]].season == 1
    rep = s.report()
    assert FLAG_OUT_OF_SCOPE in by_name(rep)["t02.mkv"].flags
    assert FLAG_OUT_OF_SCOPE not in by_name(rep)["t00.mkv"].flags
    assert any("evidence points to S02E03" in n for n in rep.notices)


def test_episodes_scope_and_pins(season3, tmp_path):
    with pytest.raises(ValueError):
        Session(plain_bundle(9), None, Config(), episodes=set())
    with pytest.raises(ValueError):
        Session(plain_bundle(9), None, Config(), season=1, episodes={(4, 1)})
    with pytest.raises(ValueError):
        Session(plain_bundle(9), None, Config(), ordering="nonesuch")
    s = Session(plain_bundle(9), None, Config(), episodes={(1, 1), (1, 2), (1, 3)},
                scorers=[TableScorer(TABLE)], vocabulary=set())
    s.analyze([season3 / "Disc 1" / n for n in LAYOUT["Disc 1"]])
    assert [c.id for c in s.candidates] == ["e1", "e2", "e3"]
    sol = s.solve()
    assert assigned(sol, "a3.mkv") == "e3"  # e7 is out of scope
    with pytest.raises(ValueError):
        s.pin(season3 / "Disc 1" / "a1.mkv", "S01E04")  # not in --episodes
    sol = s.pin(season3 / "Disc 1" / "a1.mkv", "S01E02")
    assert assigned(sol, "a1.mkv") == "e2" and assigned(sol, "a2.mkv") != "e2"
    sol = s.unpin(season3 / "Disc 1" / "a1.mkv")
    assert assigned(sol, "a1.mkv") == "e1"


# ---- per-disc LLM window (§10.2), fake client ---------------------------------------------------


class WindowFake:
    """Fake anthropic client: recognizes the file by the `token <name>` line of its excerpt and
    rates the episode in `truth`; when that episode isn't shown, "none of these" wins."""

    def __init__(self, truth: dict[str, int]):
        self.truth = truth
        self.calls: list[tuple[str, list[str]]] = []
        self.messages = self

    def parse(self, **kw):
        from showvenist.scoring.llm import CandidateScore, LLMResult

        block, excerpt = (c["text"] for c in kw["messages"][0]["content"])
        labels = re.findall(r"\[(S\d+E\d+)\]", block)
        tok = re.search(r"token (\w+)", excerpt)[1]
        self.calls.append((tok, labels))
        want = f"S01E{self.truth[tok]:02d}"
        hit = want in labels
        scores = [
            CandidateScore(episode=lb, probability=0.97 if lb == want else 0.01,
                           quotes=[f"token {tok}"] if lb == want else [])
            for lb in labels
        ]
        usage = NS(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0,
                   cache_read_input_tokens=0)
        return NS(parsed_output=LLMResult(scores=scores, none_probability=0.02 if hit else 0.95),
                  stop_reason="end_turn", usage=usage)


def test_llm_asks_disc_by_disc(season3, tmp_path):
    from showvenist.scoring.llm import LlmScorer

    table = {**TABLE, "a3.mkv": {"e3": STRONG}, "b2.mkv": {"e5": 2.0}}  # b2: weak
    s = session_for(season3, plain_bundle(9), table, tmp_path)
    sol = s.solve()
    b2 = next(i for i, f in enumerate(s.files) if f.path.name == "b2.mkv")
    c3 = next(i for i, f in enumerate(s.files) if f.path.name == "c3.mkv")
    assert sol.confidences[b2].level != "high"
    windows = s.llm_windows()
    assert [sorted(c) for _, c in windows] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    fake = WindowFake({"b2": 5, "c3": 2})  # c3 claims an episode of disc 1: window fails
    llm = LlmScorer(fake)
    llm.send = {b2, c3}
    sol = s.apply_llm(llm)
    shown = {tok: set(labels) for tok, labels in fake.calls}
    assert shown["b2"] == {"S01E04", "S01E05", "S01E06"}
    assert shown["c3"] == {"S01E07", "S01E08", "S01E09"}
    assert len(fake.calls) == 4  # two files, both orders
    assert llm.send == {b2, c3} and llm.shown is None  # restored
    assert assigned(sol, "b2.mkv") == "e5" and sol.regimes[b2] == "llm"
    # out-of-window columns got the floor, never 0
    assert sol.results["llm"].llr[b2, 0] < 0
    rep = by_name(s.report())
    assert FLAG_DISC_WINDOW_FAILED in rep["c3.mkv"].flags
    assert FLAG_DISC_WINDOW_FAILED not in rep["b2.mkv"].flags
    assert rep["b2.mkv"].regime == "llm"


def test_llm_season_scope_and_flat_list_show_full_season(season3, tmp_path):
    from showvenist.scoring.llm import LlmScorer

    cfg = Config()
    cfg.llm.scope = "season"
    s = session_for(season3, plain_bundle(9), TABLE, tmp_path, config=cfg)
    s.solve()
    assert s.llm_windows() == [(set(range(9)), None)]
    fake = WindowFake({"b2": 5})
    llm = LlmScorer(fake, send_both_orders=False)
    llm.send = {next(i for i, f in enumerate(s.files) if f.path.name == "b2.mkv")}
    s.apply_llm(llm)
    assert len(fake.calls[0][1]) == 9


# ---- performance: 24 files on 6 discs, real scorers --------------------------------------------

SYL = "ka lo mi ru te zan bo vi qua shel dor fen gri hap jum nix pel sor tuv wex yob".split()


def topics(n: int) -> list[list[str]]:
    rng, seen, out = random.Random(7), set(), []
    while len(out) < n:
        words = []
        while len(words) < 5:
            w = "".join(rng.choices(SYL, k=3))
            if w not in seen:
                seen.add(w)
                words.append(w)
        out.append(words)
    return out


def script(words: list[str], k: int) -> list[str]:
    rng = random.Random(k)
    topical = [f"Look out, that {words[0]} is coming!", f"Back to the {words[1]} tonight.",
               f"Nobody believes the {words[2]} is real.", f"Meet me at the {words[3]}.",
               f"That {words[4]} nearly had us."]
    return [topical[n % 5] if n % 4 == 0 else " ".join(rng.choices(FILLER, k=8))
            for n in range(240)]


def test_24_files_6_discs_end_to_end_and_fast_resolve(tmp_path, capsys):
    n, per = 24, 4
    tps = topics(n)
    bdir = tmp_path / "synth-show-2000"
    (bdir / "subs").mkdir(parents=True)
    eps = []
    for k in range(n):
        rel = f"subs/e{k + 1}.en.srt"
        make_srt(bdir / rel, [(5.0 * m, 5.0 * m + 2, t) for m, t in enumerate(script(tps[k], k))])
        w = tps[k]
        eps.append(Episode(
            id=f"e{k + 1}", title=f"Story {k + 1}", runtime_min=20 + k % 5,
            synopses=[Synopsis(source="tmdb", text=f"A {w[0]} near the {w[1]}, a {w[2]} by "
                                                   f"the {w[3]} and a {w[4]}.")],
            subtitles=[SubtitleRef(lang="en", path=rel, source="localsubs")],
        ))
    bundle = Bundle(
        show=ShowInfo(title="Synth Show", year=2000), episodes=eps,
        orderings={"aired": Ordering(source="tmdb", type="aired", entries=[
            OrderingEntry(season=1, episode=k + 1, episode_id=f"e{k + 1}") for k in range(n)])},
    ).with_revision(bdir)
    season = tmp_path / "rips" / "Synth Show" / "Season 1"
    rng = random.Random(3)
    for d in range(n // per):
        order = list(range(d * per, (d + 1) * per))
        rng.shuffle(order)  # title order within a disc is random
        for t, k in enumerate(order):
            make_episode_mkv(season / f"Disc {d + 1}" / f"t{t:02d}_{k}.mkv", 20 + k % 5,
                             script(tps[k], k))

    t0 = time.perf_counter()
    s = Session(bundle, bdir, Config(), season=1, cache=AnalysisCache(tmp_path / "a.db"),
                tree=tree_of(season))
    s.analyze()
    t1 = time.perf_counter()
    sol = s.solve()
    t2 = time.perf_counter()
    assert len(s.files) == n and sol.used_blocks and len(sol.discs) == 6
    for i, f in enumerate(sol.files):
        truth = f"e{int(f.path.stem.split('_')[1]) + 1}"
        assert sol.candidates[sol.assignment.column[i]].id == truth
    assert all(c.level == "high" for c in sol.confidences)
    S_before = s._fused(s.ordering).S
    cache = s._evidence(s.ordering).cache
    cached = len(cache.store)
    first = sol.files[0].path
    t3 = time.perf_counter()
    sol = s.pin(first, "extra")
    sol = s.unpin(first)
    t4 = time.perf_counter()
    assert s._fused(s.ordering).S is S_before  # S is not rebuilt by pin/unpin
    assert s._evidence(s.ordering).cache is cache  # block scores are reused...
    assert len(cache.store) - cached < cached  # ...only blocks the pin touches are new
    rep = s.report()
    assert exit_code(rep) == 0
    with capsys.disabled():
        print(f"\n24 files / 6 discs: analyze {t1 - t0:.2f}s, first solve {t2 - t1:.2f}s, "
              f"pin+unpin re-solves {t4 - t3:.3f}s")
    assert (t4 - t3) < (t2 - t1)


# ---- review fixes: Δ with pins is an exact re-solve; LLM window after a missing disc ----------


def _brute_blocks(S, discs, params, forbidden) -> float:
    """The §11.5 objective by exhaustion: every one-to-one assignment (or unmatched) of the
    disc files, every placement of ordered, non-overlapping blocks [a_d, b_d) of length at most
    files + slack holding each disc's matched files; − u per unclaimed position in a block,
    − g per skipped position before and between blocks (discs numbered 1, 2, ...; complete)."""
    import itertools
    import math

    import showvenist.assign.blocks as blocks_mod

    F, E = S.shape
    u, g = params.unclaimed_penalty, params.gap_penalty
    slack = getattr(params, "block_slack", getattr(blocks_mod, "BLOCK_SLACK", 0))
    spans = [[(a, b) for a in range(E + 1) for b in range(a, min(E, a + len(d.files) + slack) + 1)]
             for d in discs]
    best = -math.inf
    for cols in itertools.product(range(-1, E), repeat=F):
        used = [c for c in cols if c >= 0]
        if len(set(used)) != len(used) or any((i, c) in forbidden for i, c in enumerate(cols)):
            continue
        base = sum(float(S[i, c]) for i, c in enumerate(cols) if c >= 0)
        for placement in itertools.product(*spans):
            prev, pen, ok = 0, 0.0, True
            for d, (a, b) in zip(discs, placement, strict=True):
                if a < prev or any(not (cols[r] < 0 or a <= cols[r] < b) for r in d.files):
                    ok = False
                    break
                claimed = sum(1 for r in d.files if cols[r] >= 0)
                pen += u * (b - a - claimed) + g * (a - prev)
                prev = b
            if ok:
                best = max(best, base - pen)
    return best


def test_pinned_delta_loop_equals_exact_resolve(tmp_path):
    layout = {"Disc 1": ["a1.mkv", "a2.mkv"], "Disc 2": ["b1.mkv", "b2.mkv"]}
    season = make_season(tmp_path / "rips", layout)
    table = {
        "a1.mkv": {"e1": 3.0, "e2": 1.0}, "a2.mkv": {"e2": 2.0, "e3": 1.5},
        "b1.mkv": {"e3": 2.5, "e4": 1.0}, "b2.mkv": {"e4": 0.5, "e5": 2.0},
    }
    s = session_for(season, plain_bundle(5), table, tmp_path)
    sol = s.pin(season / "Disc 2" / "b1.mkv", "e4")  # against b1's best evidence (e3)
    assert sol.used_blocks and not sol.switched
    S0 = s._fused(s.ordering).S
    F, E = S0.shape
    pins = s._row_pins(s.candidates)
    forb = s._pin_forbidden(pins, F, E)
    discs = sorted(sol.discs, key=lambda d: d.number)
    total = _brute_blocks(S0, discs, s.block_params, forb)
    assert sol.assignment.total == pytest.approx(total)
    b1 = next(i for i, f in enumerate(sol.files) if f.path.name == "b1.mkv")
    assert sol.assignment.column[b1] == 3 and sol.confidences[b1].level == "high"
    checked = 0
    for i in range(F):
        if i in pins:
            continue
        alt = _brute_blocks(S0, discs, s.block_params, forb | {(i, sol.assignment.column[i])})
        assert sol.confidences[i].delta == pytest.approx(max(0.0, total - alt)), sol.files[i]
        checked += 1
    assert checked == 3


def test_llm_window_after_a_missing_disc_runs_to_the_season_end(tmp_path):
    from showvenist.scoring.llm import LlmScorer

    # disc 2 was not ripped: disc 3's window can't be counted from disc 1's end
    layout = {"Disc 1": ["a1.mkv", "a2.mkv", "a3.mkv"], "Disc 3": ["c1.mkv", "c2.mkv", "c3.mkv"]}
    season = make_season(tmp_path / "rips", layout)
    table = {"a1.mkv": {"e1": STRONG}, "a2.mkv": {"e2": STRONG}, "a3.mkv": {"e3": STRONG},
             "c1.mkv": {"e7": STRONG}, "c2.mkv": {"e8": 1.0}, "c3.mkv": {"e9": STRONG}}
    s = session_for(season, plain_bundle(9), table, tmp_path)
    sol = s.solve()
    assert [d.number for d in sol.discs] == [1, 3] and sol.discs[1].free_gap_before
    c2 = next(i for i, f in enumerate(s.files) if f.path.name == "c2.mkv")
    windows = s.llm_windows()
    assert [sorted(c) for _, c in windows] == [[0, 1, 2], [3, 4, 5, 6, 7, 8]]
    fake = WindowFake({"c2": 8})
    llm = LlmScorer(fake, send_both_orders=False)
    llm.send = {c2}
    sol = s.apply_llm(llm)
    ((tok, labels),) = fake.calls
    assert tok == "c2" and labels == [f"S01E{k:02d}" for k in range(4, 10)]
    assert assigned(sol, "c2.mkv") == "e8"


def test_named_files_read_in_the_marker_naming_order_by_type(tmp_path):
    from showvenist.hierarchy.types import Marker

    layout = {"Disc 1": ["a1.mkv", "a2.mkv"]}
    season = make_season(tmp_path / "rips", layout)
    named = season / "Disc 1" / "synth_s01e02.mkv"
    named.write_bytes(b"never analyzed" * 50)
    b = plain_bundle(4, {"chrono": [1, 3, 2, 4]})
    b = b.model_copy(update={"orderings": {
        **b.orderings, "chrono": b.orderings["chrono"].model_copy(update={"type": "dvd"})}})
    # naming_order given as an ordering type ('dvd', as a Jellyfin keyword) while --order is
    # aired: names are read in the naming order (S01E02 there is e3), not in --order
    s = session_for(season, b, {"a1.mkv": {"e1": STRONG}, "a2.mkv": {"e2": STRONG}}, tmp_path,
                    marker=Marker(show="Synth Show", naming_order="dvd"))
    assert s.starting_ordering == "aired" and s._named_order() == "chrono"
    sol = s.solve()
    assert assigned(sol, "synth_s01e02.mkv") == "e3"
