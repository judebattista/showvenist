import math

from showvenist.scoring.characters import GuestScorer
from tests.synth import make_bundle, make_ctx, make_file

GUESTS = [["Zorblax"], ["Quenta / Zorblax"], ["Doctor"], [], [], ["Wilhelmina (voice)"]]


def _run(lines_per_file):
    ctx = make_ctx(make_bundle(6, guests=GUESTS))
    files = [make_file(f"f{i}.mkv", lines=ls) for i, ls in enumerate(lines_per_file)]
    return GuestScorer().score(files, ctx)


def test_idf_weighted_and_counted_once():
    res = _run([["Zorblax! Zorblax is here.", "Where is zorblax?"], ["Quenta, hello."]])
    assert res.available.all()
    # zorblax is in 2 of 6 episodes: idf = ln 3, counted once despite 3 mentions
    assert math.isclose(res.llr[0, 0], math.log(3)) and math.isclose(res.llr[0, 1], math.log(3))
    assert math.isclose(res.llr[1, 1], math.log(6)) and res.llr[1, 0] == 0
    assert res.notes[(1, 1)] == {"names": ["quenta"]}
    assert (res.llr >= 0).all() and res.window_llr.shape == (2, 1, 6)


def test_dictionary_word_names_filtered():
    res = _run([["The doctor will see you now."]])
    assert not res.llr.any()


def test_parenthetical_ignored_and_no_dialogue():
    res = _run([["Wilhelmina!"], None])
    assert res.llr[0, 5] > 0
    assert list(res.available) == [True, False] and not res.llr[1].any()


def test_series_cast_characters_ignored():
    from showvenist.bundle.schema import CastEntry

    b = make_bundle(6, guests=GUESTS)
    b.series_cast = [CastEntry(actor="Main Star", character="Zorblax")]
    ctx = make_ctx(b)
    f = make_file("f.mkv", lines=["Zorblax and Quenta"])
    res = GuestScorer().score([f], ctx)
    assert math.isclose(res.llr[0, 1], math.log(6))  # only quenta (unique to ep 1) counts
    assert res.llr[0, 0] == 0 and res.notes[(0, 1)] == {"names": ["quenta"]}
