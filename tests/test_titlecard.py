"""TitleCardScorer (spec §9.2, decision 54, task sv-2k9): pure unit tests, hand-built objects."""

import numpy as np

from showvenist.analysis.types import CardRead
from showvenist.bundle.schema import Bundle, Episode, ShowInfo, TitleCardConfig
from showvenist.config import load_config
from showvenist.fusion import fuse
from showvenist.scoring.titlecard import MATCH_LLR, TitleCardScorer
from showvenist.scoring.types import Candidate, FileInfo, ScoreContext

JACK = ["X", "XI", "XVIII", "XXIII", "XXV", "XXVIII", "XXXIV", "XLVIII", "LII", "XCIII", "LI"]


def cfg(kind="roman", pattern=r"EPISODE\s+(?P<ref>[IVXLC]+)"):
    return TitleCardConfig.model_validate({
        "windows": [{"from": "end", "length_seconds": 60}], "pattern": pattern, "kind": kind})


def ctx(titles, card="default", kind="roman", pattern=None, alt=None):
    eps = [Episode(id=f"tmdb:{k}", title=t, alt_titles=(alt or {}).get(k, []))
           for k, t in enumerate(titles)]
    c = cfg(kind, pattern) if pattern else cfg(kind)
    b = Bundle(show=ShowInfo(title="Show"), episodes=eps).with_revision()
    if card is not None:
        b = b.model_copy(update={"title_card": c})
    cands = [Candidate(k, e, 1, k + 1) for k, e in enumerate(eps)]
    return ScoreContext(b, cands, load_config(None, env={}))


def fi(*texts, none=False):
    reads = None if none else [CardRead(float(k), t) for k, t in enumerate(texts)]
    return FileInfo(path="a", fingerprint="f", duration_s=1.0, card_reads=reads)


def run(titles, *files, **kw):
    return TitleCardScorer().score(list(files), ctx(titles, **kw))


def hits(res, i=0):
    return [j for j in range(res.llr.shape[1]) if res.llr[i, j] > 0]


def test_two_exact_reads():
    r = run(JACK, fi("EPISODE X", "EPISODE X"))
    assert hits(r) == [0] and r.llr[0, 0] == MATCH_LLR and r.available[0]


def test_x_not_confused_with_xi():
    assert hits(run(JACK, fi("EPISODE XI"))) == [1]
    assert hits(run(JACK, fi("EPISODE X"))) == [0]


def test_single_exact_read_counts():
    assert hits(run(JACK, fi("EPISODE XVIII"))) == [2]


def test_lookalike_normalisation_single_read():
    assert hits(run(JACK, fi("EPISODE XClll"))) == [9]  # XCIII


def test_unnormalisable_ocr_is_ignored():
    # 'Xxiill' / 'XXiIll' normalise to XXIIII: not a numeral, so no evidence at all
    r = run(JACK, fi("EPISODE Xxiill", "EPISODE XXiIll"))
    assert r.available[0] and not hits(r)


def test_noise_does_not_break_consensus():
    r = run(JACK, fi("EPISODE XXV", "Directed by Genndy", "EPISODE XXV", "EPISODE QQ"))
    assert hits(r) == [4] and r.notes[(0, 4)]["readings"] == ["XXV", "XXV"]


def test_noisy_wrong_numeral_blocks():
    assert not hits(run(JACK, fi("EPISODE XXV", "EPISODE XXV", "EPISODE XXVIII")))


def test_fade_li_to_lii_gives_zero():
    assert not hits(run(JACK, fi("EPISODE LI", "EPISODE LII")))


def test_reading_matching_no_candidate_gives_nothing():
    r = run(JACK, fi("EPISODE CCC"))
    assert r.available[0] and not hits(r)


def test_available_flags():
    r = run(JACK, fi("EPISODE X"), fi(none=True), fi())
    assert list(r.available) == [True, False, True]
    assert np.count_nonzero(r.llr[1:]) == 0
    assert not run(JACK, fi("EPISODE X"), card=None).available.any()
    assert np.count_nonzero(run(JACK, fi("EPISODE X"), card=None).llr) == 0


def test_alt_titles_and_non_numeral_titles():
    r = run(["Pilot", "Mix"], fi("EPISODE XI"), alt={0: ["XI"]}, card="x")
    assert hits(r) == [0]
    assert not hits(run(["The Beginning"], fi("EPISODE X")))


def test_duplicate_titles_ambiguous():
    assert not hits(run(["XI", "XI", "X"], fi("EPISODE XI", "EPISODE XI")))


def test_kind_number():
    c = dict(kind="number", pattern=r"(?:EP|NO\.?)\s*(?P<ref>\d+)")
    r = run(["1", "12", "Pilot"], fi("EP 12"), **c)
    assert hits(r) == [1]
    assert not hits(run(["1", "12"], fi("EP 12", "EP 1"), **c))
    assert not hits(run(["Pilot"], fi("EP 12"), **c))


def test_kind_title_fuzzy():
    c = dict(kind="title", pattern=r"(?P<ref>[A-Za-z ]+)")
    titles = ["The Long Night", "Winter Is Coming", "Episode 3"]
    assert hits(run(titles, fi("the long nigth", "THE LONG NIGHT"), **c)) == [0]
    # one fuzzy (non-exact) read is not enough, one exact read is
    assert not hits(run(titles, fi("the long nigth"), **c))
    assert hits(run(titles, fi("The Long Night"), **c)) == [0]
    assert not hits(run(titles, fi("Something Else"), **c))
    assert not hits(run(titles, fi("Episode 3"), **c))  # generic titles never match


def test_never_negative():
    r = run(JACK, fi("EPISODE LI", "EPISODE LII"), fi("EPISODE X"), fi("junk"))
    assert (r.llr >= 0).all() and TitleCardScorer.can_be_negative is False


def test_fusion_adds_eight_to_right_candidate_only():
    c = ctx(JACK)
    r = TitleCardScorer().score([fi("EPISODE XXIII", "EPISODE XXIII")], c)
    base = fuse({"titlecard": TitleCardScorer().score([fi(none=True)], c)},
                load_config(None, env={}).weights)
    out = fuse({"titlecard": r}, load_config(None, env={}).weights)
    diff = out.S - base.S
    assert diff[0, 3] == MATCH_LLR
    assert np.count_nonzero(diff) == 1
