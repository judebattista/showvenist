import json
from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest

from showvenist.adapters import llm_cues as lc
from tests.synth import make_bundle

USAGE = NS(input_tokens=1000, output_tokens=500, cache_creation_input_tokens=0,
           cache_read_input_tokens=0)
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def msg(payload, stop="end_turn"):
    return NS(content=[NS(type="text", text=payload)], stop_reason=stop, usage=USAGE)


class FakeBatches:
    def __init__(self, outcomes, polls_to_end=2):
        self.outcomes, self.polls, self.polls_to_end = outcomes, 0, polls_to_end
        self.created = None

    def create(self, requests):
        self.created = requests
        return NS(id="batch_1")

    def retrieve(self, bid):
        self.polls += 1
        return NS(processing_status="ended" if self.polls >= self.polls_to_end else "in_progress")

    def results(self, bid):
        # reversed: results arrive in any order
        for r in reversed(self.created):
            kind = self.outcomes.get(r["custom_id"])
            if kind is None:
                continue
            if isinstance(kind, str):
                yield NS(custom_id=r["custom_id"], result=NS(type=kind))
            else:
                yield NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=kind))


def client(outcomes, **kw):
    b = FakeBatches(outcomes, **kw)
    return NS(messages=NS(batches=b)), b


GOOD = json.dumps({"names": ["Bob", "bob", " Alice "], "terms": ["flux"],
                   "lines": ["a", "b", "c", "d"]})


def test_requests_shape_and_skips():
    b = make_bundle(3)
    b.episodes[1].excluded = True
    reqs = lc.build_requests(b, "claude-sonnet-5-5")
    assert len(reqs) == 2
    ids = [r["custom_id"] for r in reqs]
    assert ids == ["tmdb_100", "tmdb_102"]
    p = reqs[0]["params"]
    assert "thinking" not in p and p["output_config"]["format"]["type"] == "json_schema"
    assert "confident" in p["system"]
    from showvenist.bundle.schema import LlmCues
    b.episodes[0].llm_cues = LlmCues(model="m", generated_at="x")
    assert len(lc.build_requests(b, "m")) == 1
    assert len(lc.build_requests(b, "m", overwrite=True)) == 2


def test_generate_cues_end_to_end():
    b = make_bundle(3)
    c, batches = client({"tmdb_100": msg(GOOD), "tmdb_101": "errored",
                         "tmdb_102": msg("not json")})
    slept, seen = [], []
    rep = lc.generate_cues(b, c, "claude-sonnet-5-5", sleep=slept.append, on_submitted=seen.append)
    assert seen == ["batch_1"] and slept == [30.0] and batches.polls == 2
    cues = b.episodes[0].llm_cues
    assert cues.names == ["Bob", "Alice"] and cues.lines == ["a", "b", "c"]
    assert cues.model == "claude-sonnet-5-5" and cues.generated_at
    assert b.episodes[1].llm_cues is None and b.episodes[2].llm_cues is None
    assert rep.stored == ["tmdb:100"]
    assert rep.failed == {"tmdb:101": "errored", "tmdb:102": "invalid"}
    # two results carry usage (one stored, one invalid); the errored one has none
    expect = 2 * (1000 * 2.0 + 500 * 10.0) / 1e6 * 0.5
    assert rep.cost_usd == pytest.approx(expect)


def test_collect_stamp_and_refusal():
    b = make_bundle(2)
    c, _ = client({"tmdb_100": msg(GOOD), "tmdb_101": msg("{}", stop="refusal")})
    c.messages.batches.create(requests=lc.build_requests(b, "m"))
    rep = lc.collect_cues(b, c, "batch_1", "m", now=lambda: NOW)
    assert b.episodes[0].llm_cues.generated_at == "2026-10-05T12:00:00Z"
    assert rep.failed == {"tmdb:101": "refusal"}


def test_submit_poll_separate_and_timeout():
    b = make_bundle(2)
    c, batches = client({}, polls_to_end=99)
    assert lc.submit_cues(b, c, "m") == "batch_1"
    assert lc.poll_cues(c, "batch_1") == "in_progress"
    with pytest.raises(TimeoutError):
        lc.generate_cues(b, c, "m", poll_interval=10, timeout=20, sleep=lambda s: None)


def test_nothing_to_ask_makes_no_call():
    b = make_bundle(1)
    b.episodes[0].excluded = True
    c, batches = client({})
    assert lc.generate_cues(b, c, "m").stored == [] and batches.created is None


def test_estimate_cost():
    assert lc.estimate_cost(make_bundle(10), "claude-sonnet-5-5") > 0
