import numpy as np

from showvenist.bundle.schema import LlmCues
from showvenist.scoring.cues import CuesScorer
from tests.synth import make_bundle, make_ctx, make_file


def _cues(**kw):
    return LlmCues(model="m", generated_at="2026-01-01", **kw)


def test_unavailable_without_cues():
    ctx = make_ctx(make_bundle(6))
    res = CuesScorer().score([make_file(lines=["a submarine mutiny"])], ctx)
    assert not res.available.any() and not res.llr.any() and res.llr.shape == (1, 6)


def test_cues_rank_true_episode():
    terms = [["submarine", "mutiny", "harbor"], ["vineyard", "poison"], ["satellite", "orbit"],
             ["lighthouse", "tunnel"], ["violin", "orchestra"], ["forger", "carnival"]]
    b = make_bundle(6)
    for ep, t in zip(b.episodes, terms, strict=True):
        ep.llm_cues = _cues(terms=t, lines=[f"the {t[0]} is near"])
    ctx = make_ctx(b)
    res = CuesScorer().score(
        [make_file(lines=["The submarine mutiny in the harbor!"]), make_file("b.mkv")], ctx)
    assert list(res.available) == [True, False]
    assert int(np.argmax(res.llr[0])) == 0 and res.llr[0, 0] > 0 and (res.llr >= 0).all()
    assert not res.llr[1].any()
    assert "mutiny" in res.notes[(0, 0)]["terms"]


def test_cast_name_cues_ignored():
    b = make_bundle(6, guests=[["Zorblax"], [], [], [], [], []])
    for k, ep in enumerate(b.episodes):
        ep.llm_cues = _cues(names=["Zorblax"] if k == 0 else [], terms=[f"term{'x' * k}"])
    res = CuesScorer().score([make_file(lines=["zorblax zorblax"])], make_ctx(b))
    assert not res.llr.any()


def test_series_cast_cue_names_get_no_credit():
    from showvenist.bundle.schema import CastEntry

    b = make_bundle(6)
    b.episodes[0].llm_cues = _cues(names=["Zorblax"], terms=["submarine"])
    f = make_file(lines=["Zorblax and the submarine"])
    assert "zorblax" in CuesScorer().score([f], make_ctx(b)).notes[(0, 0)]["terms"]
    b.series_cast = [CastEntry(actor="Some Body", character="Zorblax")]
    notes = CuesScorer().score([f], make_ctx(b)).notes.get((0, 0), {})
    assert "zorblax" not in notes.get("terms", [])
