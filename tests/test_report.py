import math
from pathlib import Path

import numpy as np

from showvenist.report import build_report, exit_code, format_odds, render_tables, to_json
from showvenist.report_schema import Report
from showvenist.scoring.types import Assignment, FileConfidence, Fused
from tests.synth import make_ctx, make_file


def _build(flags=None, levels=("high", "medium", "high"), deltas=(8.0, 3.0, math.inf)):
    ctx = make_ctx()
    files = [make_file("b.mkv"), make_file("a.mkv"), make_file("x.mkv")]
    files[0].disc, files[1].disc, files[2].disc = "disc2", "disc1", "disc1"
    F, E = 3, len(ctx.candidates)
    dur = np.zeros((F, E))
    dur[0, 0], dur[0, 1] = 1.5, -2.0
    title = np.zeros((F, E))
    title[1, 1] = 3.0
    fused = Fused(S=np.zeros((F, E)), contributions={"duration": dur, "title": title,
                  "llm": np.zeros((F, E))}, regimes=["base", "base", "base"], log_prior=0.0)
    asg = Assignment(column=[0, 1, -1], total=0.0)
    confs = [FileConfidence(deltas[0], levels[0], 1, 2.0),
             FileConfidence(deltas[1], levels[1], -1, 1.0),
             FileConfidence(deltas[2], levels[2], None, None)]
    return build_report(files, ctx.candidates, fused, asg, confs, show_slug="synth",
                        bundle_revision="r1", ordering="aired", flags=flags)


def test_odds_formatting():
    assert format_odds(math.log(120)) == "120:1"
    assert format_odds(0.0) == "1.0:1"
    assert format_odds(math.log(3.04)) == "3.0:1"
    assert format_odds(math.log(10.2)) == "10:1"
    assert format_odds(math.log(5000)) == ">999:1"
    assert format_odds(math.inf) == ">999:1"
    assert format_odds(None) == ">999:1"


def test_sorting_and_decisions():
    r = _build()
    assert [Path(f.path).name for f in r.files] == ["a.mkv", "x.mkv", "b.mkv"]
    a, x, b = r.files
    assert a.decision.episode.episode == 2 and a.decision.episode.title == "Episode Title 2"
    assert x.decision.kind == "extra" and x.decision.episode is None
    assert x.confidence.delta is None and x.confidence.odds == ">999:1"
    assert x.runner_up is None and all(s.runner_up is None for s in x.signals)
    assert r.summary.model_dump() == {"files": 3, "high": 2, "medium": 1, "low": 0, "extras": 1}


def test_runner_up_and_signals():
    r = _build()
    b = r.files[2]
    assert b.runner_up.episode.episode == 2
    names = [s.name for s in b.signals]
    assert "llm" not in names and "duration" in names  # base regime only
    dur = next(s for s in b.signals if s.name == "duration")
    assert dur.available and dur.winner == 1.5 and dur.runner_up == -2.0
    a = r.files[0]  # runner-up is extra -> 0.0
    assert a.runner_up.kind == "extra"
    t = next(s for s in a.signals if s.name == "title")
    assert t.winner == 3.0 and t.runner_up == 0.0 and t.available


def test_json_roundtrip():
    r = _build(flags={0: ["duplicate"]})
    j = to_json(r)
    assert j.lstrip().startswith("{\n  \"report_version\": 1")
    assert Report.model_validate_json(j) == r
    assert to_json(r) == j


def test_exit_codes():
    assert exit_code(_build(levels=("high",) * 3)) == 0
    assert exit_code(_build()) == 2
    assert exit_code(_build(levels=("high", "low", "high"))) == 2
    # a medium extra needs review too
    assert exit_code(_build(levels=("high", "high", "medium"))) == 2
    # confirmed medium file is fine; not_ready high file is not
    assert exit_code(_build(levels=("high", "medium", "high"),
                            flags={1: ["already_confirmed"]})) == 0
    assert exit_code(_build(levels=("high",) * 3, flags={0: ["not_ready"]})) == 2


def test_render_tables():
    out = render_tables(_build(flags={0: ["duplicate"]}))
    assert "Season 1" in out and "Extras" in out
    assert "S01E01 Episode Title 1" in out and "extra" in out
    assert "duplicate" in out and "3 files" in out
