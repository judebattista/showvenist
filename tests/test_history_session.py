"""The history prior and confirmed files in Session (spec §4.3, §11.4); synthetic MKVs."""

import numpy as np
import pytest

from showvenist.analysis.types import AnalysisCache
from showvenist.config import Config
from showvenist.session import (
    FLAG_CONFIRMED,
    FLAG_HISTORY_DUPLICATE,
    Session,
)
from showvenist.state.db import StateDb
from showvenist.state.fingerprint import fingerprint
from showvenist.state.history import History
from tests.test_session import rips, synth_bundle, write_refs  # noqa: F401  (rips fixture)

SHOW = "synth-show-2000"
FILES = ("e3", "e1", "nosubs")


@pytest.fixture
def bdir(tmp_path):
    d = tmp_path / SHOW
    write_refs(synth_bundle(), d)
    return d


def make(bdir, tmp_path, rips, history=None, **kw):  # noqa: F811
    s = Session(write_refs(synth_bundle(), bdir), bdir, Config(), season=1,
                cache=AnalysisCache(tmp_path / "analysis.db"), history=history, **kw)
    s.analyze([rips[k] for k in FILES])
    return s


def hist():
    return History(StateDb(":memory:"))


def confirm(h, path, eid, n, source="auto"):
    fp = fingerprint(path)
    h.touch_file(fp, size=path.stat().st_size, path=str(path))
    h.confirm(fp, SHOW, episode_id=eid, season=1 if eid else None, episode=n, title=None,
              source=source, run_id=None)


def other_copy(h, eid, n, path="/old/rip/copy.mkv"):
    h.touch_file("9:other", size=9, path=path)
    h.confirm("9:other", SHOW, episode_id=eid, season=1, episode=n, title=None,
              source="manual", run_id=None)


def test_empty_history_changes_nothing(bdir, tmp_path, rips):  # noqa: F811
    off = make(bdir, tmp_path, rips)
    on = make(bdir, tmp_path, rips, history=hist())
    a, b = off.report(), on.report()
    assert a.model_dump() == b.model_dump()
    assert np.array_equal(off.solve().fused.S, on.solve().fused.S)


def test_prior_lowers_a_taken_episode_softly_and_warns(bdir, tmp_path, rips):  # noqa: F811
    off = make(bdir, tmp_path, rips)
    h = hist()
    other_copy(h, "tmdb:100", 1)  # S01E01 is already confirmed for another file
    on = make(bdir, tmp_path, rips, history=h)
    s_off, s_on = off.solve(), on.solve()
    i = next(k for k, f in enumerate(s_on.files) if f.path == rips["e1"])
    assert s_off.fused.S[i, 0] - s_on.fused.S[i, 0] == pytest.approx(np.log(10))
    assert np.allclose(s_off.fused.S[i, 1:], s_on.fused.S[i, 1:])
    assert s_on.fused.contributions["history"][i, 0] == pytest.approx(-np.log(10))
    rep = on.report()
    row = next(r for r in rep.files if r.path == str(rips["e1"]))
    assert row.decision.episode.episode == 1  # soft: the strong evidence still wins
    assert FLAG_HISTORY_DUPLICATE in row.flags
    assert any("copy.mkv" in n and "t01.mkv" in n for n in rep.notices)
    assert all(FLAG_HISTORY_DUPLICATE not in r.flags for r in rep.files if r is not row)


def test_own_confirmation_is_not_a_duplicate_when_reanalyzed(bdir, tmp_path, rips):  # noqa: F811
    h = hist()
    confirm(h, rips["e1"], "tmdb:100", 1)
    on = make(bdir, tmp_path, rips, history=h, reanalyze=True)
    rep = on.report()
    row = next(r for r in rep.files if r.path == str(rips["e1"]))
    assert FLAG_HISTORY_DUPLICATE not in row.flags and FLAG_CONFIRMED not in row.flags
    assert rips["e1"] in [f.path for f in on.files]


def test_confirmed_extra_is_not_analyzed(bdir, tmp_path, rips):  # noqa: F811
    h = hist()
    confirm(h, rips["e1"], None, None)
    on = make(bdir, tmp_path, rips, history=h)
    assert rips["e1"] not in [f.path for f in on.files]
    rep = on.report()
    row = next(r for r in rep.files if r.path == str(rips["e1"]))
    assert row.decision.kind == "extra" and row.flags == [FLAG_CONFIRMED]
    assert row.confidence.level == "high"
    assert len(rep.files) == 3 and {r.decision.episode.episode for r in rep.files
                                    if r.decision.episode} == {3, 4}


def test_confirmed_episode_is_reported_from_history_and_pins_the_solve(bdir, tmp_path, rips):  # noqa: F811
    h = hist()
    confirm(h, rips["e1"], "tmdb:102", 3)  # contradicts the evidence on purpose: history wins
    on = make(bdir, tmp_path, rips, history=h)
    assert rips["e1"] not in [f.path for f in on.files]  # not analyzed
    rep = on.report()
    row = next(r for r in rep.files if r.path == str(rips["e1"]))
    assert row.decision.episode.episode == 3 and row.flags == [FLAG_CONFIRMED]
    assert row.confidence.level == "high"
    other = next(r for r in rep.files if r.path == str(rips["e3"]))
    assert other.decision.episode is None or other.decision.episode.episode != 3  # held
    # --reanalyze ignores the stored decision for this file
    again = make(bdir, tmp_path, rips, history=h, reanalyze=True)
    assert rips["e1"] in [f.path for f in again.files]


def test_other_show_history_is_ignored(bdir, tmp_path, rips):  # noqa: F811
    h = hist()
    fp = fingerprint(rips["e1"])
    h.touch_file(fp, size=1, path=str(rips["e1"]))
    h.confirm(fp, "another-show", episode_id=None, season=None, episode=None, title=None,
              source="auto", run_id=None)
    on = make(bdir, tmp_path, rips, history=h)
    assert rips["e1"] in [f.path for f in on.files]
