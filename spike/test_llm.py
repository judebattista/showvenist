# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic", "rapidfuzz", "pytest"]
# ///
"""Run: uv run spike/test_llm.py   (fake client, synthetic text, no network)"""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location("llm", Path(__file__).parent / "llm.py")
llm = importlib.util.module_from_spec(spec)
sys.modules["llm"] = llm
spec.loader.exec_module(llm)

TMDB = {"show_title": "Fake Show", "episodes": [
    {"season": 1, "episode": n, "title": f"T{n}", "overview": f"Synopsis {n}.",
     "guest_stars": [{"actor": "A", "character": f"Char{n}"}]} for n in (1, 2, 3)]}
DIAL = {"path": "/x/a.mkv", "basename": "a.mkv", "windows": [
    {"lines": [{"t": 1, "text": "The purple walrus\nsings softly", "norm": "x"},
               {"t": 2, "text": "[NOISE]", "norm": ""},
               {"t": 3, "text": "Bring the green lantern home", "norm": "x"}]}]}
USAGE = NS(input_tokens=100, output_tokens=50, cache_creation_input_tokens=1000, cache_read_input_tokens=0)


class Fake:
    """Fake client: responder(kwargs) -> LLMResult | None (None = refusal)."""
    def __init__(self, responder):
        self.calls, self.responder = [], responder
        self.messages = self

    def parse(self, **kw):
        self.calls.append(kw)
        r = self.responder(kw)
        return NS(parsed_output=r, stop_reason="refusal" if r is None else "end_turn", usage=USAGE)


def res(p, none, quotes=None):
    q = quotes or {}
    return llm.LLMResult(
        scores=[llm.CandidateScore(episode=f"S01E0{i + 1}", probability=v, quotes=q.get(i, [])) for i, v in enumerate(p)],
        none_probability=none)


def run_file(client, tmp, budget=None):
    return llm.score_file(client, tmp, TMDB, DIAL, budget or llm.Budget(10), {})


def test_excerpt_skips_empty_and_flattens():
    ex = llm.build_excerpt(DIAL)
    assert "NOISE" not in ex and "\n" not in ex.split("\n[...]")[0]


def test_parse_call_shape_and_cache_breakpoint(tmp_path):
    f = Fake(lambda kw: res([0.6, 0.2, 0.1], 0.1, {0: ["purple walrus sings"]}))
    run_file(f, tmp_path)
    kw = f.calls[0]
    assert kw["model"] == "claude-sonnet-5-5" and kw["output_format"] is llm.LLMResult
    c = kw["messages"][0]["content"]
    assert c[0]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in c[1]
    assert "purple walrus" in c[1]["text"] and "Synopsis 1" in c[0]["text"]


def test_renormalize_and_logodds(tmp_path):
    f = Fake(lambda kw: res([2.0, 1.0, 1.0], 1.0, {0: ["purple walrus sings"]}))  # sums to 5
    out = run_file(f, tmp_path)
    e = out["episodes"]
    assert e["S01E01"]["p"] == pytest.approx(0.4)
    assert e["S01E01"]["logodds"] == pytest.approx(math.log(0.4) - math.log(0.2))
    assert e["S01E02"]["logodds"] == pytest.approx(0.0)


def test_quote_check_drops_positive_evidence(tmp_path):
    f = Fake(lambda kw: res([0.7, 0.1, 0.1], 0.1, {0: ["completely invented sentence about dragons"]}))
    out = run_file(f, tmp_path)
    e = out["episodes"]["S01E01"]
    assert e["quote_failed"] and e["logodds"] <= 0 and e["quotes"] == []


def test_quote_fuzzy_match_ok(tmp_path):
    f = Fake(lambda kw: res([0.7, 0.1, 0.1], 0.1, {0: ["Bring the green lantern hom"]}))
    e = run_file(f, tmp_path)["episodes"]["S01E01"]
    assert not e["quote_failed"] and e["logodds"] > 0 and e["quotes"]


def test_refusal_gives_no_evidence(tmp_path):
    out = run_file(Fake(lambda kw: None), tmp_path)
    assert out["refusal"] and out["no_evidence"] and out["episodes"] == {}


def test_one_order_refused_uses_other(tmp_path):
    n = {"i": 0}
    def r(kw):
        n["i"] += 1
        return None if n["i"] == 1 else res([0.7, 0.1, 0.1], 0.1, {0: ["purple walrus sings"]})
    out = run_file(Fake(r), tmp_path)
    assert out["refusal"] and not out["no_evidence"] and out["episodes"]["S01E01"]["logodds"] > 0


def test_cache_hit_costs_nothing(tmp_path):
    f = Fake(lambda kw: res([0.7, 0.1, 0.1], 0.1, {0: ["purple walrus sings"]}))
    a = run_file(f, tmp_path)
    assert len(f.calls) == 2 and a["cost_new_usd"] > 0
    f2 = Fake(lambda kw: pytest.fail("API called"))
    b = run_file(f2, tmp_path)
    assert b["cost_new_usd"] == 0 and b["episodes"] == a["episodes"]


def test_both_orders_averaged_and_reversed_list(tmp_path):
    # position-biased fake: favours whichever candidate is listed first
    def r(kw):
        first = kw["messages"][0]["content"][0]["text"]
        top = 0 if first.index("[S01E01]") < first.index("[S01E03]") else 2
        p = [0.1, 0.1, 0.1]
        p[top] = 0.6
        return res(p, 0.1, {top: ["purple walrus sings"]})
    out = run_file(Fake(r), tmp_path)
    e = out["episodes"]
    assert e["S01E01"]["logodds"] == pytest.approx(e["S01E03"]["logodds"])
    assert e["S01E01"]["logodds"] == pytest.approx((math.log(6) + math.log(1)) / 2)


def test_ceiling_abort(tmp_path):
    f = Fake(lambda kw: res([0.7, 0.1, 0.1], 0.1, {0: ["purple walrus sings"]}))
    with pytest.raises(llm.CeilingExceeded):
        run_file(f, tmp_path, llm.Budget(0.0))
    assert f.calls == []


def test_cost_from_usage():
    u = dict(input_tokens=1_000_000, output_tokens=1_000_000, cache_creation_input_tokens=0, cache_read_input_tokens=1_000_000)
    assert llm.cost_usd(u) == pytest.approx(2 + 10 + 0.2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
