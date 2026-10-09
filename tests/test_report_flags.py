"""Report `available` bits and the advisory flags play_all / possible_two_parter (spec §14)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from showvenist.analysis.types import AnalysisCache
from showvenist.config import Config
from showvenist.edgeflags import PLAY_ALL_WINDOW_MARGIN, play_all, possible_two_parter
from showvenist.report import build_report, render_tables, to_json
from showvenist.report_schema import Report
from showvenist.scoring.types import Assignment, FileConfidence, Fused, ScoreResult
from showvenist.session import (
    FLAG_DISC_WINDOW_FAILED,
    FLAG_PLAY_ALL,
    FLAG_POSSIBLE_TWO_PARTER,
    Session,
)
from tests.synth import make_bundle, make_ctx, make_file
from tests.test_session import dialogue_for, make_episode_mkv, synth_bundle, write_refs

# ---- available -------------------------------------------------------------------------------


def _report(avail):
    ctx = make_ctx()
    files = [make_file("a.mkv")]
    E = len(ctx.candidates)
    contrib = {"duration": np.zeros((1, E)), "title": np.zeros((1, E))}
    contrib["duration"][0, 0] = 1.0  # title contributes nothing
    fused = Fused(S=np.zeros((1, E)), contributions=contrib, regimes=["base"], log_prior=0.0,
                  available=avail)
    return build_report(files, ctx.candidates, fused, Assignment([0], 0.0),
                        [FileConfidence(math.inf, "high", None, None)], show_slug="s",
                        bundle_revision="r", ordering="aired")


def test_available_comes_from_the_scorer():
    r = _report({"duration": np.array([True]), "title": np.array([True])})
    sig = {s.name: s for s in r.files[0].signals}
    assert sig["title"].available and sig["title"].winner == 0.0  # available, contributes 0
    r = _report({"duration": np.array([True]), "title": np.array([False])})
    sig = {s.name: s for s in r.files[0].signals}
    assert not sig["title"].available and sig["duration"].available


def test_available_flows_through_fuse():
    from showvenist.config import WeightsConfig
    from showvenist.fusion import fuse

    res = {"duration": ScoreResult.empty(2, 3), "title": ScoreResult.empty(2, 3)}
    res["duration"].available[:] = [True, False]
    f = fuse(res, WeightsConfig())
    assert f.available["duration"].tolist() == [True, False]
    assert f.available["title"].tolist() == [False, False]


# ---- flag predicates ---------------------------------------------------------------------------


def _win_results(rows, E=6):
    """refsubs window scores for one file: rows = [(best_col, best, other)] per window."""
    win = np.zeros((1, len(rows), E))
    for w, (col, hi, other) in enumerate(rows):
        win[0, w, :] = other
        win[0, w, col] = hi
    r = ScoreResult.empty(1, E)
    r.window_llr = win
    return {"refsubs": r}


def test_play_all_predicate():
    cands = make_ctx().candidates  # six 44-minute episodes
    hi = PLAY_ALL_WINDOW_MARGIN + 3
    differ = _win_results([(0, hi, 0.0), (3, hi, 0.0), (3, hi, 0.0)])
    assert play_all(0, 90 * 60, cands, differ)
    assert not play_all(0, 60 * 60, cands, differ)  # not long enough (< 1.8 x 44)
    agree = _win_results([(2, hi, 0.0), (2, hi, 0.0), (2, hi, 0.0)])
    assert not play_all(0, 90 * 60, cands, agree)
    weak = _win_results([(0, 1.0, 0.0), (3, 1.0, 0.0)])  # margin below the constant
    assert not play_all(0, 90 * 60, cands, weak)
    one = _win_results([(0, hi, 0.0), (3, 0.0, 0.0)])  # second window has no dialogue
    assert not play_all(0, 90 * 60, cands, one)


def _fused(E, pos):
    c = np.zeros((1, E))
    for j in pos:
        c[0, j] = 2.0
    return Fused(S=np.zeros((1, E)), contributions={"refsubs": c}, regimes=["base"],
                 log_prior=0.0)


def test_two_parter_predicate():
    cands = make_ctx(make_bundle(6, runtimes=[44, 44, 44, 44, 44, 44])).candidates
    assert possible_two_parter(0, 2, 88 * 60, cands, _fused(6, [2, 3]))
    assert not possible_two_parter(0, 2, 88 * 60, cands, _fused(6, [2]))  # next has no evidence
    assert not possible_two_parter(0, 3, 88 * 60, cands, _fused(6, [2, 3]))  # not the first
    assert not possible_two_parter(0, 2, 44 * 60, cands, _fused(6, [2, 3]))  # single length
    assert not possible_two_parter(0, -1, 88 * 60, cands, _fused(6, [2, 3]))  # extra
    assert not possible_two_parter(0, 5, 88 * 60, cands, _fused(6, [5]))  # no next episode


# ---- session -----------------------------------------------------------------------------------


def _session(tmp_path, specs, pins=()):
    """specs: {name: (minutes, lines)} -> analyzed Session on the synth bundle."""
    bdir = tmp_path / "synth-show-2000"
    root = tmp_path / "rips" / "Synth Show" / "Season 1" / "Disc 1"
    paths = {n: make_episode_mkv(root / n, m, lines) for n, (m, lines) in specs.items()}
    s = Session(write_refs(synth_bundle(), bdir), bdir, Config(), season=1,
                cache=AnalysisCache(tmp_path / "a.db"))
    s.analyze(list(paths.values()))
    return s, paths


def _two_episode_lines(a, b, n=180):
    return dialogue_for(a)[:n] + dialogue_for(b)[:n]


@pytest.fixture
def play_all_session(tmp_path):
    specs = {
        "long.mkv": (30, _two_episode_lines(0, 2)),  # median runtime 12.5 -> 30 >= 22.5
        "same.mkv": (30, dialogue_for(1)),
        "short.mkv": (8, dialogue_for(0)),
    }
    return _session(tmp_path, specs)


def _flags(s, paths, name):
    sol = s.solve()
    i = next(k for k, f in enumerate(sol.files) if f.path == paths[name])
    return s.flags(i)


def test_play_all_in_session(play_all_session):
    s, paths = play_all_session
    assert FLAG_PLAY_ALL in _flags(s, paths, "long.mkv")
    assert FLAG_PLAY_ALL not in _flags(s, paths, "same.mkv")  # windows agree
    assert FLAG_PLAY_ALL not in _flags(s, paths, "short.mkv")
    rep = s.report()
    by = {r.path.rsplit("/", 1)[-1]: r for r in rep.files}
    assert FLAG_PLAY_ALL in by["long.mkv"].flags
    assert FLAG_PLAY_ALL in render_tables(rep)
    data = Report.model_validate_json(to_json(rep))
    assert FLAG_PLAY_ALL in {f for r in data.files for f in r.flags}


def test_play_all_not_flagged_when_pinned(play_all_session):
    s, paths = play_all_session
    s.solve()
    s.pin(paths["long.mkv"], "extra")
    assert FLAG_PLAY_ALL not in _flags(s, paths, "long.mkv")


def test_flags_do_not_change_assignment(play_all_session, monkeypatch):
    import showvenist.session as sess

    s, _ = play_all_session
    with_flags = s.solve()
    a1 = (list(with_flags.assignment.column), [(c.level, c.delta) for c in with_flags.confidences])
    s2, _ = play_all_session  # same Session object: re-solve with the detectors disabled
    monkeypatch.setattr(sess, "play_all", lambda *a, **k: False)
    monkeypatch.setattr(sess, "possible_two_parter", lambda *a, **k: False)
    s2._evid.clear()
    s2.solution = None
    sol = s2.solve()
    assert (list(sol.assignment.column), [(c.level, c.delta) for c in sol.confidences]) == a1
    assert all(FLAG_PLAY_ALL not in s2.flags(i) for i in range(len(sol.files)))


def test_two_parter_in_session(tmp_path):
    # S01E01 (8 min) + S01E02 (11 min) = 19 min: one file holding both scripts
    specs = {
        "two.mkv": (19, _two_episode_lines(0, 1)),
        "e3.mkv": (14, dialogue_for(2)),
        "e4.mkv": (17, dialogue_for(3)),
    }
    s, paths = _session(tmp_path, specs)
    sol = s.solve()
    i = next(k for k, f in enumerate(sol.files) if f.path == paths["two.mkv"])
    assert sol.candidates[sol.assignment.column[i]].label == "S01E01"
    assert FLAG_POSSIBLE_TWO_PARTER in s.flags(i)
    j = next(k for k, f in enumerate(sol.files) if f.path == paths["e3.mkv"])
    assert FLAG_POSSIBLE_TWO_PARTER not in s.flags(j)
    assert FLAG_POSSIBLE_TWO_PARTER in render_tables(s.report())


# ---- out_of_scope / disc_window_failed in the report -------------------------------------------


def test_disc_window_failed_is_reported(play_all_session):
    s, paths = play_all_session
    sol = s.solve()
    i = next(k for k, f in enumerate(sol.files) if f.path == paths["short.mkv"])
    s._window_failed = {i}  # what apply_llm records for a failed disc window
    rep = s.report()
    by = {r.path: r for r in rep.files}
    assert FLAG_DISC_WINDOW_FAILED in by[str(paths["short.mkv"])].flags
    assert FLAG_DISC_WINDOW_FAILED not in by[str(paths["same.mkv"])].flags
    assert FLAG_DISC_WINDOW_FAILED in to_json(rep)
