"""M6 engine additions (decision 61, spec §12.4): Session.explain, Session.overridden,
FileNode.explicit through the planner, and recording overrides as rejected matches."""

from __future__ import annotations

import pytest

from showvenist.hierarchy.parse import build_tree
from showvenist.rename.plan import build_plan
from showvenist.rename.types import PlanOptions
from showvenist.report import format_odds
from showvenist.session import Session, record_overrides
from showvenist.state.db import StateDb
from showvenist.state.history import History
from tests.synth import make_bundle
from tests.test_rename_plan import SLUG, plan_for, row, season, touch
from tests.test_session import rips, session  # noqa: F401  (fixtures)

# ---- explain ----------------------------------------------------------------------------------


def test_explain_top_candidates(session, rips):  # noqa: F811
    sol = session.solve()
    d = session.explain(rips["e3"])
    assert d.path == str(rips["e3"]) and d.duration_s == sol.files[0].duration_s
    assert len(d.candidates) == 3
    best = d.candidates[0]
    assert best.label == "S01E03" and best.delta == 0.0
    scores = [c.score for c in d.candidates]
    assert scores == sorted(scores, reverse=True)
    for c in d.candidates:
        assert c.delta == pytest.approx(best.score - c.score)
        assert c.odds == format_odds(c.delta)
        assert {s.name for s in c.signals} >= {"duration", "refsubs"}
    rs = next(s for s in best.signals if s.name == "refsubs")
    assert rs.available and rs.winner > 0 and rs.runner_up is None
    assert any(n.startswith("refsubs") for n in best.notes)
    assert best.runtime_min == next(
        c.episode.runtime_min for c in session.candidates if c.id == best.episode_id)
    assert len(d.signals) == len(d.notes) == len(d.runtimes) == 3
    assert len(session.explain(rips["e3"], n=2).candidates) == 2
    # a duplicate's path explains its primary
    assert session.explain(rips["dup"]).path == str(rips["e3"])


def test_explain_rejects_non_rows(session, rips, tmp_path):  # noqa: F811
    session.solve()
    session.errors[tmp_path / "bad.mkv"] = "unreadable"
    with pytest.raises(ValueError):
        session.explain(tmp_path / "bad.mkv")
    with pytest.raises(KeyError):
        session.explain(tmp_path / "unknown.mkv")


def test_explain_does_not_change_state(session, rips):  # noqa: F811
    sol = session.solve()
    session.explain(rips["e1"])
    assert session.solution is sol


# ---- overridden -------------------------------------------------------------------------------


def _state(s: Session):
    return (dict(s.pins), s.solution, s._choice, s.ordering, list(s.candidates),
            set(s._evid), set(s._oos), set(s._window_failed))


def test_overridden_lists_only_real_overrides_and_keeps_state(session, rips):  # noqa: F811
    session.solve()
    assert session.overridden() == {}
    session.pin(rips["e1"], "S01E02")  # the tool says S01E01
    session.pin(rips["e3"], "S01E03")  # the tool agrees
    before = _state(session)
    sol = session.solution
    out = session.overridden()
    assert list(out) == [rips["e1"]]
    ep = out[rips["e1"]].episode
    assert ep is not None and ep.episode == 1
    after = _state(session)
    assert after == before
    assert session.solution is sol
    assert session.report().model_dump() == session.report().model_dump()
    assert len(session.pins) == 2


def test_overridden_extra_pin(session, rips):  # noqa: F811
    session.solve()
    session.pin(rips["e1"], "extra")
    d = session.overridden()[rips["e1"]]
    assert d.kind == "episode" and d.episode.episode == 1


def test_overridden_swapped_pins_report_the_unaided_choices(session, rips):  # noqa: F811
    """The tool picks E03 for t00 and E01 for t01; pinning them the other way round lists both,
    each with the tool's own episode (a solve per pin keeping the other pin gave 'extra')."""
    sol = session.solve()
    assert _assigned_label(sol, rips["e3"]) == "S01E03"
    assert _assigned_label(sol, rips["e1"]) == "S01E01"
    session.pin(rips["e3"], "S01E01")
    session.pin(rips["e1"], "S01E03")
    before = _state(session)
    out = session.overridden()
    assert set(out) == {rips["e3"], rips["e1"]}
    assert out[rips["e3"]].kind == "episode" and out[rips["e3"]].episode.episode == 3
    assert out[rips["e1"]].kind == "episode" and out[rips["e1"]].episode.episode == 1
    assert _state(session) == before


def _assigned_label(sol, path):
    i = next(k for k, f in enumerate(sol.files) if f.path == path)
    col = sol.assignment.column[i]
    return "extra" if col < 0 else sol.candidates[col].label


def test_record_overrides_writes_rejected_rows(session, rips):  # noqa: F811
    h = History(StateDb(":memory:"))
    session.solve()
    session.pin(rips["e1"], "S01E02")
    session.pin(rips["e3"], "S01E03")
    assert record_overrides(h, session, None, "synth-show-2000") == 1
    rows = h.rejected("synth-show-2000")
    assert len(rows) == 1
    assert rows[0].fingerprint == session.files[1].fingerprint
    assert (rows[0].season, rows[0].episode) == (1, 1) and rows[0].status == "rejected"


def test_record_overrides_nothing_when_pin_equals_choice(session, rips):  # noqa: F811
    h = History(StateDb(":memory:"))
    session.solve()
    session.pin(rips["e3"], "S01E03")
    assert record_overrides(h, session, None, "synth-show-2000") == 0
    assert h.rejected("synth-show-2000") == []
    session.unpin(rips["e3"])
    assert record_overrides(h, session, None, "synth-show-2000") == 0


# ---- explicit files ---------------------------------------------------------------------------


def test_explicit_file_is_renamed_in_place_walked_file_is_flattened(tmp_path):
    s = season(tmp_path)
    a = touch(s / "Disc 1" / "t00.mkv")
    b = touch(s / "Disc 1" / "t01.mkv")
    ra = row(a, 1).model_copy(update={"explicit": True})
    plan = plan_for(ra, row(b, 2))
    assert plan.ok, plan.errors
    got = {m.old.name: m.new for m in plan.moves}
    assert got["t00.mkv"] == s / "Disc 1" / "synth_show_s01e01.mkv"
    assert got["t01.mkv"] == s / "synth_show_s01e02.mkv"


def test_explicit_is_not_in_the_json_schema(tmp_path):
    r = row(touch(tmp_path / "Disc 1" / "t00.mkv"), 1).model_copy(update={"explicit": True})
    assert '"explicit"' not in r.model_dump_json()


def test_files_tree_marks_explicit_and_session_threads_it(rips):  # noqa: F811
    from showvenist.cli import _files_tree
    from showvenist.config import HierarchyConfig

    hcfg = HierarchyConfig()
    tree = _files_tree([rips["e3"]], hcfg, lambda p: None)
    assert [f.explicit for s in tree.seasons for d in s.discs for f in d.files] == [True]
    walked = build_tree(rips["e3"].parents[2], config=hcfg, tag_of=lambda p: None)
    assert not any(f.explicit for s in walked.seasons for d in s.discs for f in d.files)
    # Session.report carries the flag onto the rows (the file is a disc file in the tree)
    from showvenist.config import Config
    from tests.test_session import synth_bundle, write_refs

    bdir = rips["e3"].parents[3] / "synth-show-2000"
    sess = Session(write_refs(synth_bundle(), bdir), bdir, Config(), season=1, tree=tree)
    assert sess._node[rips["e3"]][1].explicit
    sess.analyze([rips["e3"], rips["e1"]])
    got = {r.path: r.explicit for r in sess.report().files}
    assert got == {str(rips["e3"]): True, str(rips["e1"]): False}


def test_build_plan_uses_report_explicit(tmp_path):
    s = season(tmp_path)
    a = touch(s / "Disc 1" / "t00.mkv")
    b = make_bundle()
    from tests.test_rename_plan import report

    rep = report(row(a, 1).model_copy(update={"explicit": True}), bundle=b)
    plan = build_plan(rep, b, PlanOptions())
    assert plan.moves[0].new.parent == s / "Disc 1"
    assert SLUG
