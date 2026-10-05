from showvenist.scoring.names import cast_names
from showvenist.scoring.text import Bm25, bm25_llr, chance_max, content_tokens, name_tokens, tokens
from tests.synth import make_bundle


def test_tokens_normalize_and_filter():
    assert tokens("[LAUGHS] Don't go, I am Mr. O'Brien! 42 ok") == ["dont", "i", "obrien"]


def test_name_tokens_ignore_parentheses():
    assert name_tokens("Jane Smith (voice)") == ["jane", "smith"]


def test_content_tokens_strip_names_and_stopwords():
    assert content_tokens("the zorblax went home with jane", {"jane"}) == ["zorblax"]


def test_cast_names_covers_actors_characters_crew():
    b = make_bundle(2, guests=[["Zorblax / Quenta"], []])
    b.episodes[0].directors = ["Dana Quill"]
    b.episodes[1].writers = ["Rob Vex"]
    names = cast_names(b)
    assert {"actor", "zorblax", "quenta", "dana", "quill", "rob", "vex"} <= names


def test_chance_max_and_no_credit_for_flat_scores():
    assert abs(chance_max(6) - 1.8930) < 1e-3
    bm = Bm25([["alpha", "beta"], ["gamma"], ["delta"], ["alpha"]])
    assert not bm25_llr(bm, ["nothing"]).any()
