import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from showvenist.analysis.types import AnalysisCache
from showvenist.scoring.llm import (
    CandidateScore,
    LLMResult,
    LlmScorer,
    cost_usd,
    evidence,
    floor_evidence,
)
from tests.synth import make_bundle, make_ctx, make_file

LINES = ["The purple walrus sings softly", "Bring the green lantern home"]
QUOTE = "purple walrus sings softly"
USAGE = NS(input_tokens=100, output_tokens=50, cache_creation_input_tokens=1000,
           cache_read_input_tokens=0)


class Fake:
    def __init__(self, responder, usage=USAGE):
        self.calls, self.responder, self.usage = [], responder, usage
        self.messages = self

    def parse(self, **kw):
        self.calls.append(kw)
        r = self.responder(kw)
        return NS(parsed_output=r, stop_reason="refusal" if r is None else "end_turn",
                  usage=self.usage)


def res(probs: dict[int, float], none: float, quote_for=()):
    """probs: candidate number (1-based) -> p; others get a tiny share."""
    n = max(probs)
    return LLMResult(
        scores=[CandidateScore(episode=f"S01E{k:02d}", probability=probs.get(k, 0.001),
                               quotes=[QUOTE] if k in quote_for else [])
                for k in range(1, n + 1)],
        none_probability=none)


def first_listed(kw):
    t = kw["messages"][0]["content"][0]["text"]
    return t.index("[S01E01]") < t.index("[S01E02]")


def setup(n=22):
    return make_ctx(make_bundle(n)), [make_file(lines=LINES)]


def full(ctx, top, p, none, quote=True):
    probs = {c.number: (1 - p - none) / (ctx.n_candidates - 1) for c in ctx.candidates}
    probs[top] = p
    return res(probs, none, quote_for=(top,) if quote else ())


def test_evidence_values():
    assert evidence(0.97, 22) == pytest.approx(math.log(0.97 / 0.03) + math.log(21))
    assert 6.4 < evidence(0.97, 22) < 6.6
    assert evidence(0.9999, 22) == evidence(0.99, 22)
    assert evidence(1e-9, 22) == floor_evidence(22)
    assert evidence(1 / 22, 22) == pytest.approx(0.0)


def test_high_confidence_file():
    ctx, files = setup()
    f = Fake(lambda kw: full(ctx, 3, 0.97, 0.005))
    r = LlmScorer(f, send_both_orders=False).score(files, ctx)
    assert r.available[0]
    p = r.notes[(0, 2)]["p"]
    assert r.llr[0, 2] == pytest.approx(evidence(p, 22)) and 6.0 < r.llr[0, 2] < 7.0
    assert r.llr[0].argmax() == 2 and r.llr[0, 0] < 0


def test_kwargs_shape():
    ctx, files = setup(3)
    f = Fake(lambda kw: full(ctx, 1, 0.8, 0.05))
    LlmScorer(f, send_both_orders=False).score(files, ctx)
    kw = f.calls[0]
    c = kw["messages"][0]["content"]
    assert c[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in c[1]
    assert "purple walrus" in c[1]["text"] and "duration" not in c[1]["text"].lower()
    assert kw["model"] == "claude-sonnet-5-5" and kw["output_format"] is LLMResult


def test_quote_check_drops_unsupported_candidate():
    ctx, files = setup(5)
    bad = lambda kw: LLMResult(scores=[CandidateScore(  # noqa: E731
        episode="S01E02", probability=0.8, quotes=["totally invented dragons line"])] +
        [CandidateScore(episode=f"S01E0{k}", probability=0.04, quotes=[]) for k in (1, 3, 4, 5)],
        none_probability=0.04)
    r = LlmScorer(Fake(bad), send_both_orders=False).score(files, ctx)
    assert r.notes[(0, 1)]["quote_failed"]
    assert r.llr[0, 1] == pytest.approx(0.0, abs=1e-9)
    ok = LlmScorer(Fake(lambda kw: full(ctx, 2, 0.8, 0.04)), send_both_orders=False)
    r2 = ok.score(files, ctx)
    assert not r2.notes[(0, 1)]["quote_failed"] and r2.llr[0, 1] > 2.5


def test_both_orders_averaged_one_says_none():
    ctx, files = setup(22)

    def r(kw):
        if first_listed(kw):
            return full(ctx, 3, 0.97, 0.01)
        return full(ctx, 3, 0.01, 0.97, quote=False)

    s = LlmScorer(Fake(r))
    out = s.score(files, ctx)
    assert out.notes[(0, 2)]["p"] == pytest.approx(0.49, abs=0.01)
    assert 2.5 < out.llr[0, 2] < 3.5
    assert not s.disc_window_failed(0)  # none highest in only one order


def test_averaging_cancels_position_bias():
    ctx, files = setup(3)

    def r(kw):
        top = 1 if first_listed(kw) else 3
        return res({1: 0.1, 2: 0.1, 3: 0.1, top: 0.6}, 0.1, quote_for=(top,))

    out = LlmScorer(Fake(r)).score(files, ctx)
    assert out.llr[0, 0] == pytest.approx(out.llr[0, 2])


def test_refusal_unavailable():
    ctx, files = setup(3)
    s = LlmScorer(Fake(lambda kw: None))
    r = s.score(files, ctx)
    assert not r.available[0] and np.all(r.llr == 0)


def test_failed_call_unavailable():
    ctx, files = setup(3)

    def boom(kw):
        raise RuntimeError("down")

    r = LlmScorer(Fake(boom)).score(files, ctx)
    assert not r.available[0]


def test_no_dialogue_unavailable_and_no_call():
    ctx = make_ctx(make_bundle(3))
    f = Fake(lambda kw: pytest.fail("called"))
    r = LlmScorer(f).score([make_file()], ctx)
    assert not r.available[0] and f.calls == []


def test_send_subset():
    ctx = make_ctx(make_bundle(3))
    files = [make_file("a.mkv", lines=LINES), make_file("b.mkv", lines=LINES)]
    f = Fake(lambda kw: full(ctx, 1, 0.8, 0.05))
    r = LlmScorer(f, send_both_orders=False, send=[1]).score(files, ctx)
    assert list(r.available) == [False, True] and len(f.calls) == 1
    assert np.all(r.llr[0] == 0)


def test_cache_hit_zero_calls(tmp_path):
    ctx, files = setup(3)
    cache = AnalysisCache(tmp_path / "a.db")
    f = Fake(lambda kw: full(ctx, 1, 0.8, 0.05))
    a = LlmScorer(f, cache=cache).score(files, ctx)
    assert len(f.calls) == 2
    f2 = Fake(lambda kw: pytest.fail("API called"))
    s2 = LlmScorer(f2, cache=cache)
    b = s2.score(files, ctx)
    assert f2.calls == [] and np.array_equal(a.llr, b.llr) and s2.total_cost_usd == 0
    assert s2.estimate_cost(files, ctx) == 0



def test_floor_for_non_shown_columns():
    ctx, files = setup(6)
    f = Fake(lambda kw: res({1: 0.9, 2: 0.02, 3: 0.02}, 0.02, quote_for=(1,)))
    # shown = first 3 candidate indexes
    r = LlmScorer(f, send_both_orders=False, shown=[0, 1, 2]).score(files, ctx)
    fl = floor_evidence(3)
    assert np.all(r.llr[0, 3:] == fl) and fl < 0
    assert r.llr[0, 0] > 0 and (0, 3) not in r.notes
    assert "S01E04" not in f.calls[0]["messages"][0]["content"][0]["text"]


def test_disc_window_failed_requires_both_orders():
    ctx, files = setup(4)
    none_top = lambda kw: res({1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}, 0.6)  # noqa: E731
    s = LlmScorer(Fake(none_top))
    s.score(files, ctx)
    assert s.disc_window_failed(0)
    # only one order answers (second refuses)
    n = {"i": 0}

    def half(kw):
        n["i"] += 1
        return none_top(kw) if n["i"] == 1 else None

    s2 = LlmScorer(Fake(half))
    r = s2.score(files, ctx)
    assert r.available[0] and not s2.disc_window_failed(0)
    # single-order mode: one order suffices
    s3 = LlmScorer(Fake(none_top), send_both_orders=False)
    s3.score(files, ctx)
    assert s3.disc_window_failed(0)
    # a candidate wins: not failed
    s4 = LlmScorer(Fake(lambda kw: full(ctx, 1, 0.9, 0.05)))
    s4.score(files, ctx)
    assert not s4.disc_window_failed(0)


def test_cost_from_usage():
    u = dict(input_tokens=1_000_000, output_tokens=1_000_000,
             cache_creation_input_tokens=0, cache_read_input_tokens=1_000_000)
    assert cost_usd(u, "claude-sonnet-5-5") == pytest.approx(2 + 10 + 0.2)
    ctx, files = setup(3)
    s = LlmScorer(Fake(lambda kw: full(ctx, 1, 0.8, 0.05)))
    s.score(files, ctx)
    per = (100 * 2.0 + 1000 * 2.5 + 50 * 10.0) / 1e6
    assert s.total_cost_usd == pytest.approx(2 * per)
    assert s.file_info[0]["cost_usd"] == pytest.approx(2 * per)


def test_estimate_cost_no_api():
    ctx, files = setup(10)
    f = Fake(lambda kw: pytest.fail("called"))
    s = LlmScorer(f)
    est = s.estimate_cost(files, ctx)
    assert est > 0 and f.calls == []
    assert LlmScorer(f, send=[]).estimate_cost(files, ctx) == 0


def test_lazy_anthropic_import(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)  # import would fail
    LlmScorer()  # constructing must not import it
