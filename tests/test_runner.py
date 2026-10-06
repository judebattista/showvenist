"""run_eval on a two-season synthetic manifest (tiny MKVs, no real media, no network)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from showvenist.analysis.types import AnalysisCache
from showvenist.config import Config
from showvenist.eval.runner import evidence_table, path_season, run_eval
from showvenist.eval.types import ManifestRow
from showvenist.scoring.duration import DurationScorer
from tests.test_session import dialogue_for, make_episode_mkv, synth_bundle, write_refs

SLUG = "synth-show-2000"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("lib")
    bdir = root / "bundles" / SLUG
    bundle = write_refs(synth_bundle(seasons=2), bdir)
    show = root / "rips" / "Synth Show"
    rows: list[ManifestRow] = []

    def add(rel: str, minutes, lines, label, lang="eng"):
        p = show / rel
        make_episode_mkv(p, minutes, lines, lang=lang)
        rows.append(ManifestRow(path=str(p), show=SLUG, label=label))
        return p

    # season 1: E03, E01 and an extra; season 2: E02, E04, a duplicate of E02, an `episode?`
    add("Season 1/Disc 1/t00.mkv", 14, dialogue_for(2), "S01E03")
    add("Season 1/Disc 1/t01.mkv", 8, dialogue_for(0), "S01E01")
    add("Season 1/Disc 1/t02.mkv", 2, ["behind the scenes with the cast"] * 10, "extra")
    e2 = add("Season_02/D1/t00.mkv", 11, dialogue_for(1, 2), "S02E02")
    add("Season_02/D1/t01.mkv", 17, dialogue_for(3, 2), "S02E04")
    dup = show / "Season_02/D1/t02.mkv"
    shutil.copyfile(e2, dup)
    rows.append(ManifestRow(path=str(dup), show=SLUG, label="S02E02"))
    add("Season_02/D2/t00.mkv", 8, dialogue_for(0, 2), "episode?")
    bad = show / "Season 1/Disc 2/broken.mkv"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\x00garbage" * 500)
    rows.append(ManifestRow(path=str(bad), show=SLUG, label="S01E02"))
    rows.append(ManifestRow(path=str(show / "x.mkv"), show=SLUG, label="S09E09"))  # unresolvable
    return rows, {SLUG: bundle}, {SLUG: bdir}, root


def test_run_eval_two_seasons(corpus, tmp_path):
    rows, bundles, bdirs, _ = corpus
    seen, groups = [], []
    decisions, result = run_eval(
        rows, bundles, Config(), bdirs, AnalysisCache(tmp_path / "a.db"),
        progress=lambda n, t, p, e: seen.append((n, t)), on_group=groups.append,
    )
    by_path = {d.path: d for d in decisions}
    assert len(decisions) == 7
    assert [n for n, _ in seen] == list(range(1, 9)) and seen[-1][1] == 8

    ov = result.overall
    assert ov.files == 7
    assert ov.top1 == 1.0
    assert ov.precision_high is not None and ov.precision_high.errors == 0
    assert ov.precision_high.n >= 4 and ov.coverage_high >= 0.5
    assert ov.unmatched_recall == 1.0
    assert ov.episode_unknown_files == 1 and ov.episode_unknown_provable_errors == 0
    assert set(result.per_season) == {f"{SLUG}-S01", f"{SLUG}-S02"}

    dup = [d for d in decisions if d.duplicate]
    assert len(dup) == 1 and dup[0].episode_id == "tmdb:201"
    q = by_path[rows[6].path]
    assert q.kind == "episode" and q.episode_id == "tmdb:200"

    reasons = {p.row.label: p.reason for p in result.problems}
    assert "unreadable" in reasons["S01E02"]
    assert "S09E09" in reasons
    assert not any(p.reason == "no decision" for p in result.problems)

    # evidence for the fit
    assert [(g.season, g.ordering) for g in groups] == [(1, "aired"), (2, "aired")]
    table = evidence_table(groups[1].solution, groups[1].labels)
    assert len(table) == 3  # the duplicate is not a separate row
    r0 = table[0]
    assert r0.candidates == ["tmdb:200", "tmdb:201", "tmdb:202", "tmdb:203"]
    assert set(r0.llr) == {"duration", "title", "refsubs", "synopsis", "guests", "cues"}
    assert all(v.shape == (4,) for v in r0.llr.values())
    assert r0.target == 1 and r0.label_kind == "episode"
    assert int(np.argmax(r0.llr["refsubs"])) == 1
    unknown = next(r for r in table if r.label_kind == "episode?")
    assert unknown.target is None and unknown.target_set == [0, 2]
    s1 = evidence_table(groups[0].solution, groups[0].labels)
    assert [r.target for r in s1] == [2, 0, -1]


def test_run_eval_ablation_and_whole_show_group(corpus, tmp_path):
    rows, bundles, bdirs, root = corpus
    # a file with no season folder: a whole-show group when nothing else is seasoned
    p = make_episode_mkv(root / "loose" / "t.mkv", 14, dialogue_for(2))
    alone = [ManifestRow(path=str(p), show=SLUG, label="S01E03")]
    decisions, result = run_eval(alone, bundles, Config(), bdirs, None)
    assert decisions[0].episode_id == "tmdb:102"
    # ...and a Problem when seasoned groups exist
    extra = ManifestRow(path=str(p), show=SLUG, label="extra")
    decisions, result = run_eval(
        [*rows[:2], extra], bundles, Config(), bdirs, None, scorers=[DurationScorer()]
    )
    assert len(decisions) == 2
    assert any("no season" in pr.reason for pr in result.problems)
    assert result.overall.files == 2


def test_path_season():
    assert path_season("/a/Season 3/D1/t.mkv") == 3
    assert path_season("/a/Season02/t.mkv") == 2
    assert path_season("/a/Season_11/t.mkv") == 11
    assert path_season("/a/Specials/t.mkv") is None


def test_season_from_folder_wins_and_mismatch_is_reported(corpus, tmp_path):
    _, bundles, bdirs, root = corpus
    # labeled in season 2's numbering, but it sits in the Season 1 folder (chronology labels)
    p = make_episode_mkv(root / "chrono" / "Season 1" / "t.mkv", 14, dialogue_for(2))
    ok = make_episode_mkv(root / "chrono" / "Season 1" / "u.mkv", 8, dialogue_for(0))
    rows = [ManifestRow(path=str(p), show=SLUG, label="S02E03"),
            ManifestRow(path=str(ok), show=SLUG, label="S01E01")]
    groups = []
    decisions, result = run_eval(rows, bundles, Config(), bdirs, None, on_group=groups.append)
    assert [g.season for g in groups] == [1]  # one group: the folder's season
    notes = [pr.reason for pr in result.problems if pr.row.path == str(p)]
    assert len(notes) == 1 and "season mismatch" in notes[0] and "Season 1" in notes[0]
    assert len(decisions) == 2  # kept and scored, not left out
    # its labeled episode (S02E03) isn't a candidate of season 1: no fit target
    table = evidence_table(groups[0].solution, groups[0].labels)
    assert table[0].target is None and table[1].target == 0


def test_group_timing_and_resolve_from_stored_evidence(corpus, tmp_path):
    from showvenist.eval.runner import group_decisions, resolve_metrics

    rows, bundles, bdirs, _ = corpus
    groups = []
    decisions, result = run_eval(rows, bundles, Config(), bdirs,
                                 AnalysisCache(tmp_path / "a.db"), on_group=groups.append)
    for g in groups:
        assert set(g.seconds) == {"analyze", "score", "solve", "total"}
        assert g.seconds["total"] >= g.seconds["analyze"] >= 0
        assert len(g.first_levels) == len(g.solution.files) and not g.llm_sent
    assert result.overall.cost_usd == 0.0
    # re-solving the stored evidence reproduces the run exactly
    again = {}
    for g in groups:
        again.update(group_decisions(g))
    assert again == {d.path: d for d in decisions}
    same = resolve_metrics(groups, bundles)
    assert same.overall == result.overall
    # ablation: without any scorer only the prior is left, nothing reaches high
    names = set(groups[0].solution.results)
    none = resolve_metrics(groups, bundles, drop=names)
    assert none.overall.coverage_high == 0.0
    no_refsubs = resolve_metrics(groups, bundles, drop={"refsubs"})
    assert no_refsubs.overall.files == result.overall.files


def test_run_eval_llm_needs_approval_and_records_cost(corpus, tmp_path):
    from showvenist.scoring.llm import LlmScorer
    from tests.test_session import FakeLlmClient

    rows, bundles, bdirs, _ = corpus
    clients = []

    def factory():
        clients.append(FakeLlmClient())
        return LlmScorer(clients[-1])

    estimates = []
    groups = []
    _, result = run_eval(rows, bundles, Config(), bdirs, AnalysisCache(tmp_path / "a.db"),
                         llm_mode="all", llm_factory=factory,
                         approve=lambda est: estimates.append(est) or False,
                         on_group=groups.append)
    assert all(e > 0 for e in estimates) and len(estimates) == 2
    assert all(not c.calls for c in clients)  # refused: no call was made
    assert all(g.llm_refused and g.llm_cost_usd == 0 for g in groups)
    assert result.overall.cost_usd == 0.0

    groups.clear()
    _, result = run_eval(rows, bundles, Config(), bdirs, AnalysisCache(tmp_path / "a.db"),
                         llm_mode="all", llm_factory=factory, approve=lambda est: True,
                         on_group=groups.append)
    assert sum(len(c.calls) for c in clients) > 0
    assert result.overall.cost_usd > 0
    assert result.overall.cost_usd == pytest.approx(sum(g.llm_cost_usd for g in groups))
    assert result.per_season[f"{SLUG}-S01"].cost_usd == pytest.approx(groups[0].llm_cost_usd)
    assert "llm" in groups[0].solution.regimes
    assert result.overall.top1 == 1.0
    with pytest.raises(ValueError):
        run_eval(rows, bundles, Config(), bdirs, None, llm_mode="sometimes")


# ---- disc blocks in eval (M4): discs from title tags, named files are unlabeled input --------


def tagged_mkv(path, minutes, lines, title):
    """make_episode_mkv with an MKV title tag (MakeMKV sets it to the disc name, §5.1)."""
    from tests.fixtures.textsubs import make_mkv_with_text_subs

    path.parent.mkdir(parents=True, exist_ok=True)
    dur = minutes * 60
    cues, t, n = [], 2.0, 0
    while t + 2 < dur:
        cues.append((t, t + 2.0, lines[n % len(lines)]))
        t, n = t + 5.0, n + 1
    make_mkv_with_text_subs(path, cues, duration_s=dur, title=title)
    return path


@pytest.fixture(scope="module")
def flattened(tmp_path_factory):
    """A flattened season (no disc folders) of files named like episodes, whose title tags
    give the disc: E1, E2 on disc 1; E3, E4 on disc 2."""
    root = tmp_path_factory.mktemp("flat")
    bdir = root / "bundles" / SLUG
    bundle = write_refs(synth_bundle(), bdir)
    rows = []
    for k, minutes in enumerate((8, 11, 14, 17)):
        disc = 1 if k < 2 else 2
        p = tagged_mkv(root / "Synth Show" / f"synth_s01e0{k + 1}.mkv", minutes,
                       dialogue_for(k), f"SYNTH SHOW SEASON 1 DISC {disc}")
        rows.append(ManifestRow(path=str(p), show=SLUG, label=f"S01E0{k + 1}"))
    return rows, {SLUG: bundle}, {SLUG: bdir}


def test_eval_discs_from_title_tags_reach_the_session(flattened, tmp_path):
    from showvenist.session import FLAG_NAMED

    rows, bundles, bdirs = flattened
    groups = []
    decisions, result = run_eval(rows, bundles, Config(), bdirs,
                                 AnalysisCache(tmp_path / "a.db"), on_group=groups.append)
    (g,) = groups
    sn = g.session.season_node
    assert [d.key for d in sn.discs] == ["tag:disc 1", "tag:disc 2"]
    assert [d.number for d in sn.discs] == [1, 2] and not sn.loose
    assert all(f.tag_disc is not None and f.tag_disc.source == "tag"
               for d in sn.discs for f in d.files)
    # named files are analyzed as unlabeled input, never pins
    assert all(f.named is None for d in sn.discs for f in d.files)
    assert len(g.session.files) == 4 and not g.session.named and not g.solution.pins
    assert all(FLAG_NAMED not in g.session.flags(i) for i in range(4))
    assert g.solution.used_blocks and g.n_discs == 2
    assert set(g.solution.blocks.values()) == {(0, 2), (2, 4)}
    assert result.overall.top1 == 1.0
    # re-solving the stored evidence goes through the block model as well
    from showvenist.eval.runner import group_decisions

    assert group_decisions(g) == {d.path: d for d in decisions}

    groups.clear()
    _, plain = run_eval(rows, bundles, Config(), bdirs, AnalysisCache(tmp_path / "a.db"),
                        on_group=groups.append, use_blocks=False)
    assert not groups[0].solution.used_blocks and plain.overall.top1 == 1.0


def test_group_tree_disc_sources():
    from showvenist.eval.runner import group_tree
    from showvenist.eval.types import ResolvedLabel

    def lab(path, disc=""):
        return ResolvedLabel(ManifestRow(path=path, show=SLUG, label="extra", disc=disc), "extra")

    tags = {"/x/a.mkv": "SHOW S1 DISC 2", "/x/Disc 1/b.mkv": "SHOW S1 DISC 3",
            "/x/c.mkv": "SHOW S1 DISC 2", "/x/d.mkv": None, "/x/e.mkv": "SHOW DISC 4"}
    labels = [lab("/x/a.mkv"), lab("/x/Disc 1/b.mkv"), lab("/x/c.mkv"), lab("/x/d.mkv"),
              lab("/x/e.mkv", disc="3")]
    tree = group_tree(labels, 1, lambda p: tags[str(p)])
    (s,) = tree.seasons
    assert [(d.number, [f.path.name for f in d.files]) for d in s.discs] == [
        (1, ["b.mkv"]),  # folder wins over the tag
        (2, ["a.mkv", "c.mkv"]),  # one disc per tag label
        (3, ["e.mkv"]),  # the manifest's disc column wins over everything
    ]
    assert [f.path.name for f in s.loose] == ["d.mkv"]
