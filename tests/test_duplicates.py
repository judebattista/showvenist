"""Near-duplicate rips by dialogue overlap (spec §14): the module on synthetic dialogue, then
Session and eval on tiny synthetic MKVs (no real media, no network)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from showvenist.analysis.duplicates import (
    Quality,
    choose_primary,
    find_near_duplicates,
    overlap,
    shingle_set,
)
from showvenist.analysis.types import AnalysisCache, Cue, Dialogue, SubtitleTrack, Window
from showvenist.config import Config, HierarchyConfig
from showvenist.eval.runner import run_eval
from showvenist.eval.types import ManifestRow
from showvenist.hierarchy.parse import build_tree
from showvenist.session import (
    FLAG_DUPLICATE,
    FLAG_DUPLICATE_ALLOWED,
    FLAG_NAMED,
    FLAG_NEAR_DUPLICATE,
    Session,
)
from tests.fixtures.textsubs import make_mkv_with_text_subs
from tests.test_session import FILLER, dialogue_for, make_episode_mkv, synth_bundle, write_refs

TRACK = SubtitleTrack(index=1, codec="ass", lang="eng")


def lines(seed: int, n: int = 120) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choices(FILLER, k=8)) for _ in range(n)]


def dialogue(*windows: list[str]) -> Dialogue:
    return Dialogue(TRACK, [Window(0.0, 0.0, [Cue(k, k + 1, t) for k, t in enumerate(w)])
                            for w in windows])


# ---- module -----------------------------------------------------------------------------------


def test_shingles_normalized_and_per_window():
    d = dialogue(["JOHN: Hello, there <i>my</i>", "old friend!"], ["one two three four"])
    assert shingle_set(d) == {"hello there my old friend"}  # no shingle spans two windows
    assert shingle_set(None) == frozenset() == shingle_set(Dialogue(None))


def test_overlap_is_containment():
    a = shingle_set(dialogue(lines(1)))
    recap = shingle_set(dialogue(lines(9, 30) + lines(1)))  # a copy with an extra recap
    other = shingle_set(dialogue(lines(2)))
    assert overlap(a, a) == 1.0
    assert overlap(a, recap) == overlap(recap, a) == 1.0
    assert overlap(a, other) < 0.05
    assert overlap(a, frozenset()) == 0.0


def test_groups_threshold_min_shingles_and_order():
    s1, s2 = shingle_set(dialogue(lines(1))), shingle_set(dialogue(lines(2)))
    small = shingle_set(dialogue(lines(3, 5)))
    assert 0 < len(small) < 100
    sh = {
        Path("d/c.mkv"): s1, Path("d/a.mkv"): s1 | {"x y z w v"}, Path("d/b.mkv"): s2,
        Path("d/e.mkv"): s2, Path("d/f.mkv"): small, Path("d/g.mkv"): small,
    }
    groups = find_near_duplicates(sh, 0.5, 100)
    assert groups == [[Path("d/a.mkv"), Path("d/c.mkv")], [Path("d/b.mkv"), Path("d/e.mkv")]]
    assert find_near_duplicates(sh, 0.5, 1)[-1] == [Path("d/f.mkv"), Path("d/g.mkv")]


def test_choose_primary():
    a, b, c = Path("a"), Path("b"), Path("c")
    q = {a: Quality(100, 0.6, 10), b: Quality(90, None, 5), c: Quality(90, None, 9)}
    assert choose_primary([a, b, c], q) == c  # 90 text lines beat 100 lines at OCR 0.6; size
    assert choose_primary([a, b], {}) == a  # path order last


# ---- session ----------------------------------------------------------------------------------


def lossy_copy(src_lines: list[str], path: Path, minutes: float, drop_every: int = 7,
               prefix: list[str] | None = None) -> Path:
    """The same dialogue on the same clock, every `drop_every`-th cue missing (a worse
    subtitle track: fewer lines read), after an optional recap (lines before the episode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cues, t, n = [], 2.0, 0
    for line in prefix or []:
        cues.append((t, t + 2.0, line))
        t += 5.0
    dur = minutes * 60 + t - 2.0
    while t + 2 < dur:
        if drop_every == 0 or (n + 1) % drop_every:
            cues.append((t, t + 2.0, src_lines[n % len(src_lines)]))
        t += 5.0
        n += 1
    make_mkv_with_text_subs(path, cues, duration_s=dur)
    return path


@pytest.fixture(scope="module")
def season(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("rips") / "Synth Show" / "Season 1" / "Disc 1"
    return {
        "e3": make_episode_mkv(d / "t00.mkv", 14, dialogue_for(2)),
        "e1": make_episode_mkv(d / "t01.mkv", 8, dialogue_for(0)),
        "e3b": lossy_copy(dialogue_for(2), d / "t02.mkv", 14),  # same episode, other bytes
        "e2": make_episode_mkv(d / "t03.mkv", 11, dialogue_for(1)),
    }


def new_session(tmp_path: Path, **kw) -> Session:
    bdir = tmp_path / "synth-show-2000"
    return Session(write_refs(synth_bundle(), bdir), bdir, Config(), season=1,
                   cache=AnalysisCache(tmp_path / "analysis.db"), **kw)


def rows(rep) -> dict[str, object]:
    return {Path(r.path).name: r for r in rep.files}


def test_near_duplicate_collapsed_into_the_better_copy(season, tmp_path):
    s = new_session(tmp_path)
    s.analyze([season["e3b"], season["e3"], season["e1"], season["e2"]])
    assert s.duplicates == {season["e3b"]: season["e3"]}  # fewer lines: not the primary
    assert [f.path for f in s.files] == [season["e3"], season["e1"], season["e2"]]
    primary, ov = s.near[season["e3b"]]
    assert primary == season["e3"] and ov > 0.9
    rep = s.report()
    assert len(rep.files) == 4
    r = rows(rep)
    assert r["t02.mkv"].flags[:2] == [FLAG_DUPLICATE, FLAG_NEAR_DUPLICATE]
    assert r["t02.mkv"].decision.episode.episode == 3
    assert r["t02.mkv"].confidence.level == r["t00.mkv"].confidence.level
    assert FLAG_NEAR_DUPLICATE not in r["t00.mkv"].flags
    assert {Path(x.path).name for x in rep.files if FLAG_DUPLICATE in x.flags} == {"t02.mkv"}
    note = [n for n in rep.notices if "looks like a copy" in n]
    assert len(note) == 1 and note[0].startswith("t02.mkv looks like a copy of t00.mkv")
    assert "dialogue overlap" in note[0] and "S01E03" in note[0]
    assert "_dupe_N" in note[0]

    # idempotent: analyzing again changes nothing, and the near duplicate isn't re-read
    seen = []
    s.analyze([season["e3"], season["e3b"], season["e1"], season["e2"]],
              progress=lambda n, t, p, e: seen.append(p))
    assert s.duplicates == {season["e3b"]: season["e3"]}
    assert [f.path for f in s.files] == [season["e3"], season["e1"], season["e2"]]
    assert len(seen) == 4
    assert rows(s.report())["t02.mkv"].decision.episode.episode == 3

    # pinning via the near duplicate pins its primary
    sol = s.pin(season["e3b"], "S01E04")
    i = next(k for k, f in enumerate(sol.files) if f.path == season["e3"])
    assert sol.candidates[sol.assignment.column[i]].label == "S01E04"


def test_allow_duplicates_keeps_it_as_another_version(season, tmp_path):
    s = new_session(tmp_path, allow_duplicates=True)
    s.analyze([season["e3"], season["e3b"], season["e1"]])
    rep = s.report()
    r = rows(rep)["t02.mkv"]
    assert r.flags[:2] == [FLAG_DUPLICATE_ALLOWED, FLAG_NEAR_DUPLICATE]
    assert FLAG_DUPLICATE not in r.flags
    assert r.decision.episode.episode == 3
    assert any(n.endswith("kept as another version of S01E03 (--allow-duplicates)")
               for n in rep.notices)


def test_different_episodes_same_runtime_not_grouped(tmp_path):
    d = tmp_path / "rips" / "Season 1" / "Disc 1"
    a = make_episode_mkv(d / "a.mkv", 11, dialogue_for(1))
    b = make_episode_mkv(d / "b.mkv", 11, dialogue_for(3))  # equal runtime, other dialogue
    s = new_session(tmp_path)
    s.analyze([a, b])
    assert not s.duplicates and not s.near and len(s.files) == 2


def test_copy_with_an_extra_recap_grouped_by_containment(tmp_path):
    d = tmp_path / "rips" / "Season 1" / "Disc 1"
    a = make_episode_mkv(d / "a.mkv", 14, dialogue_for(2))
    recap = dialogue_for(0)[:12]  # one minute of "previously on" before the episode
    b = lossy_copy(dialogue_for(2), d / "b.mkv", 14, drop_every=0, prefix=recap)
    s = new_session(tmp_path)
    s.analyze([a, b])
    assert len(s.near) == 1 and len(s.files) == 1
    dup, (primary, ov) = next(iter(s.near.items()))
    assert {dup, primary} == {a, b} and ov >= 0.5


def test_little_dialogue_is_never_grouped(tmp_path):
    d = tmp_path / "rips" / "Season 1" / "Disc 1"
    few = ["the quick fox jumps over the lazy dog today"] * 3
    a = make_episode_mkv(d / "a.mkv", 3, few)
    b = lossy_copy(few, d / "b.mkv", 3, drop_every=0)
    s = new_session(tmp_path)
    s.analyze([a, b])
    assert not s.near and len(s.files) == 2


def test_named_file_never_collapsed(tmp_path):
    season = tmp_path / "rips" / "Synth Show" / "Season 1"
    make_episode_mkv(season / "Disc 1" / "t00.mkv", 14, dialogue_for(2))
    named = lossy_copy(dialogue_for(2), season / "Disc 1" / "synth_s01e03.mkv", 14)
    tree = build_tree(season, config=HierarchyConfig(), tag_of=lambda p: None)
    s = new_session(tmp_path, tree=tree)
    s.analyze()
    assert not s.near and not s.duplicates
    assert named in s.named and [f.path.name for f in s.files] == ["t00.mkv"]
    r = rows(s.report())
    assert r["synth_s01e03.mkv"].flags == [FLAG_NAMED]
    assert FLAG_NEAR_DUPLICATE not in r["t00.mkv"].flags


def test_pinned_file_is_never_collapsed(season, tmp_path):
    s = new_session(tmp_path)
    s.analyze([season["e3"], season["e3b"], season["e1"]])
    s.unpin(season["e3"])  # solve once
    s.pins[season["e3b"]] = None  # a pin from an earlier pass, set before re-analysis
    s.analyze([season["e3"], season["e3b"], season["e1"]])
    # the pinned copy is never collapsed: the group follows it, even from a better copy
    assert s.duplicates == {season["e3"]: season["e3b"]}
    assert {f.path for f in s.files} == {season["e3b"], season["e1"]}
    sol = s.solve()
    assert season["e3b"] in s.pins and sol.assignment.column[s._index(season["e3b"])] == -1


def test_detection_tuned_by_config(season, tmp_path):
    cfg = Config()
    cfg.analysis.duplicate_min_shingles = 10**6
    bdir = tmp_path / "synth-show-2000"
    s = Session(write_refs(synth_bundle(), bdir), bdir, cfg, season=1,
                cache=AnalysisCache(tmp_path / "analysis.db"))
    s.analyze([season["e3"], season["e3b"]])
    assert not s.near


# ---- eval -------------------------------------------------------------------------------------


def test_eval_near_duplicate_flagged_and_out_of_precision(season, tmp_path):
    slug = "synth-show-2000"
    bdir = tmp_path / slug
    bundle = write_refs(synth_bundle(), bdir)
    manifest = [
        ManifestRow(path=str(season["e3"]), show=slug, label="S01E03"),
        ManifestRow(path=str(season["e1"]), show=slug, label="S01E01"),
        ManifestRow(path=str(season["e3b"]), show=slug, label="S01E03"),
    ]
    decisions, result = run_eval(manifest, {slug: bundle}, Config(), {slug: bdir},
                                 AnalysisCache(tmp_path / "a.db"))
    by = {Path(d.path).name: d for d in decisions}
    assert by["t02.mkv"].duplicate and by["t02.mkv"].episode_id == "tmdb:102"
    assert not by["t00.mkv"].duplicate and by["t00.mkv"].episode_id == "tmdb:102"
    ov = result.overall
    assert ov.top1 == 1.0
    # the duplicate set (t00, t02) is kept out of precision at high: only t01 counts
    assert ov.precision_high is not None and ov.precision_high.n == 1
    assert ov.precision_high.errors == 0
