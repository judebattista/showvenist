import importlib.util
from pathlib import Path

import numpy as np
import pytest

from showvenist.scoring.synopsis import SynopsisScorer
from showvenist.scoring.text import Bm25, bm25_llr
from tests.synth import make_bundle, make_ctx, make_file

SYN = [
    "Submarine captain mutiny aboard frozen harbor",
    "Vineyard owner poisoning scandal during harvest festival",
    "Astronaut stranded orbiting broken satellite",
    "Lighthouse keeper discovers smuggling tunnel beneath cliffs",
    "Orchestra conductor loses violin before concert",
    "Detective chases forger through carnival crowds",
]
LINES = ["The captain faces a mutiny.", "Nobody leaves this submarine.",
         "The harbor is frozen solid.", "Mutiny, captain, mutiny!"]


def test_paraphrase_ranks_true_episode_first():
    ctx = make_ctx(make_bundle(6, synopses=SYN))
    res = SynopsisScorer().score([make_file(lines=LINES)], ctx)
    assert res.available[0]
    assert int(res.llr[0].argmax()) == 0 and res.llr[0, 0] > 0
    assert (res.llr >= 0).all()
    assert res.window_llr.shape == (1, 1, 6)
    assert "mutiny" in res.notes[(0, 0)]["terms"]


def test_noise_gets_no_credit():
    ctx = make_ctx(make_bundle(6, synopses=SYN))
    f = make_file(lines=["completely unrelated words zebra quartz"])
    res = SynopsisScorer().score([f], ctx)
    assert not res.llr.any()


def test_no_dialogue_unavailable_zeros():
    ctx = make_ctx(make_bundle(6, synopses=SYN))
    res = SynopsisScorer().score([make_file(), make_file()], ctx)
    assert not res.available.any() and not res.llr.any()


def test_cast_only_names_do_not_count():
    syn = list(SYN)
    syn[1] = "Vineyard owner poisoning scandal"
    guests = [[], ["Zorblax"], [], [], [], []]
    b = make_bundle(6, synopses=syn, guests=guests)
    b.episodes[2].synopses[0].text += " zorblax zorblax zorblax"  # name appears in other synopsis
    ctx = make_ctx(b)
    res = SynopsisScorer().score([make_file(lines=["Zorblax zorblax zorblax Zorblax"])], ctx)
    assert not res.llr.any()


def _spike():
    pytest.importorskip("scipy")
    spec = importlib.util.spec_from_file_location(
        "spike_score", Path(__file__).parents[1] / "spike" / "score.py")
    if spec is None or spec.loader is None or not spec.origin or not Path(spec.origin).exists():
        pytest.skip("spike not available")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_matches_spike_numerically():
    sp = _spike()
    docs = [["mutiny", "captain", "harbor"], ["wine", "poison", "harvest"], ["orbit", "satellite"],
            ["tunnel", "cliff", "smuggler"], ["violin", "concert"], ["forger", "carnival", "crowd"],
            ["ship", "storm", "rescue"]]
    q = ["mutiny", "captain", "mutiny", "harbor", "orbit", "violin", "captain", "mutiny"]
    assert np.allclose(Bm25(docs).scores(q), sp.bm25_scores(docs, q))
    for query in (q, ["mutiny", "captain", "harbor", "ship", "storm"]):
        want = np.minimum(sp.synopsis_llr(docs, query), 4.0)
        got = np.minimum(bm25_llr(Bm25(docs), query), 4.0)
        assert np.allclose(got, want)
