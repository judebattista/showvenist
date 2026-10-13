from pathlib import Path

from showvenist.bundle.schema import Bundle, CastEntry
from showvenist.scoring.names import cast_names, main_cast, series_cast_tokens
from showvenist.scoring.text import Bm25, bm25_llr, chance_max, content_tokens, name_tokens, tokens
from tests.synth import make_bundle


def test_tokens_normalize_and_filter():
    assert tokens("[LAUGHS] Don't go, I am Mr. O'Brien! 42 ok") == ["dont", "i", "obrien"]


def test_name_tokens_ignore_parentheses():
    assert name_tokens("Jane Smith (voice)") == ["jane", "smith"]


def test_content_tokens_strip_names_and_stopwords():
    assert content_tokens("the zorblax went home with jane", {"jane"}) == ["zorblax"]


def test_cast_names_covers_guests_and_series_but_not_crew():
    b = make_bundle(2, guests=[["Zorblax / Quenta"], ["Doctor"]])
    b.episodes[0].directors = ["Dana Quill"]
    b.series_cast = [CastEntry(actor="Mal Hero", character="Captain Doctor")]
    names = cast_names(b)
    assert {"zorblax", "quenta", "mal", "hero", "captain"} <= names
    assert "doctor" in names  # series-cast tokens are stripped even when dictionary words
    assert "quill" not in names
    b.series_cast = []
    assert "doctor" not in cast_names(b)  # guest dictionary word is kept


def test_matches_spike_removable_names():
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "spike"))
    import score as spike

    b = make_bundle(2, guests=[["Zorblax / Quenta"], ["Doctor Nurse"]])
    b.series_cast = [CastEntry(actor="Mal Hero", character="Captain Doctor")]
    tmdb = {
        "series_cast": [{"actor": "Mal Hero", "character": "Captain Doctor"}],
        "episodes": [
            {"guest_stars": [{"actor": g.actor, "character": g.character} for g in e.guest_cast]}
            for e in b.episodes
        ],
    }
    series, allnames = spike.cast_token_sets(tmdb)
    assert cast_names(b) == spike.removable_names(series, allnames)


def test_old_show_json_without_series_cast_loads():
    data = make_bundle(2).model_dump(mode="json")
    del data["series_cast"]
    assert Bundle.model_validate(data).series_cast == []


def test_series_cast_enters_revision():
    b = make_bundle(2)
    r0 = b.compute_revision()
    b.series_cast = [CastEntry(actor="A B", character="C D")]
    assert b.compute_revision() != r0


def test_chance_max_and_no_credit_for_flat_scores():
    assert abs(chance_max(6) - 1.8930) < 1e-3
    bm = Bm25([["alpha", "beta"], ["gamma"], ["delta"], ["alpha"]])
    assert not bm25_llr(bm, ["nothing"]).any()


def test_main_cast_needs_enough_episodes():
    b = make_bundle(20)  # 15% of 20 = 3 episodes
    b.series_cast = [
        CastEntry(actor="Reg Ular", character="Rex", episodes=18),
        CastEntry(actor="Bit Part", character="Zorblax", episodes=2),
        CastEntry(actor="Edge Case", character="Quenta", episodes=3),
        CastEntry(actor="Old Bundle", character="Vera"),  # no count: kept
    ]
    assert [c.actor for c in main_cast(b)] == ["Reg Ular", "Edge Case", "Old Bundle"]
    toks = series_cast_tokens(b)
    assert "rex" in toks and "vera" in toks and "zorblax" not in toks


def test_main_cast_share_grows_with_the_season_count():
    b = make_bundle(100)  # 15% of 100 = 15 episodes
    b.series_cast = [CastEntry(actor="A B", character="C", episodes=14),
                     CastEntry(actor="D E", character="F", episodes=15)]
    assert [c.actor for c in main_cast(b)] == ["D E"]
