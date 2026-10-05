import pytest

from showvenist.scoring.title import TitleScorer, is_generic
from tests.synth import make_bundle, make_ctx, make_file


def _bundle():
    b = make_bundle(3)
    for e, t in zip(b.episodes, ["The Heist", "Midnight Train", "Snow Globe"], strict=True):
        e.title = t
    return b


def run(tag, bundle=None):
    return TitleScorer().score([make_file(title_tag=tag)], make_ctx(bundle or _bundle()))


def test_flags():
    assert TitleScorer().name == "title" and not TitleScorer().can_be_negative


def test_exact_and_case_insensitive():
    r = run("midnight TRAIN")
    assert r.available[0] and r.llr[0].tolist() == [0.0, 6.0, 0.0]


def test_fuzzy_threshold():
    b = make_bundle(2)
    b.episodes[0].title = "The Long Goodbye Home"
    assert run("The Long Goodbye Hom", b).llr[0, 0] == 6.0
    assert run("The Long Goodbye", b).llr[0, 0] == 0.0  # ratio < 90


def test_alt_titles():
    b = make_bundle(2)
    b.episodes[1].alt_titles = ["Pilot Part Deux"]
    assert run("pilot part deux", b).llr[0, 1] == 6.0


@pytest.mark.parametrize("tag", ["Season 2 Disc 1", "Disc 3", "Title 5", "t00", "Chapter 3",
                                 "Episode 4", "SYNTH_SHOW_DISC2", "Synth Show", "42"])
def test_generic_tags_never_score(tag):
    b = make_bundle(3)
    b.episodes[0].title = tag if tag != "Synth Show" else "Synth Show"
    r = run(tag, b)
    assert r.available[0] and (r.llr == 0).all()


def test_generic_episode_title_never_matches():
    b = make_bundle(2)
    b.episodes[0].title = "Episode 5"
    b.episodes[1].title = "Part 1"
    assert (run("Episode 5", b).llr == 0).all()
    assert (run("Part 1", b).llr == 0).all()


def test_no_tag_unavailable():
    for tag in (None, "", "  "):
        r = run(tag)
        assert not r.available[0] and (r.llr == 0).all()


def test_is_generic():
    assert is_generic(None) and not is_generic("The Heist")


def test_discovery_is_not_a_generic_disc_name():
    from showvenist.scoring.title import is_generic

    assert not is_generic("Discovery")
    assert is_generic("Disc 2") and is_generic("DISC_2") and is_generic("Disc2")
    assert is_generic("Show Season 1 DVD")
