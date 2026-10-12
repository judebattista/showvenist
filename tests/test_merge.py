"""merge_sources (spec §6.5): synthetic primary/secondary bundles, no real titles."""

from __future__ import annotations

from datetime import date, timedelta

import yaml

from showvenist.bundle.merge import is_uninformative, merge_sources, split_part
from showvenist.bundle.overrides import apply_overrides
from showvenist.bundle.schema import (
    Bundle,
    CastEntry,
    Episode,
    Ordering,
    OrderingEntry,
    PartialBundle,
    Provenance,
    ShowIds,
    ShowInfo,
    Synopsis,
)

TITLES = [
    "The Glass Orchard", "Lanterns Over Kettle Bay", "A Quiet Sort of Mutiny",
    "Copper and Salt", "The Cartographer's Daughter", "Nine Bells for Morrow",
    "Weathervane", "Ashes on the Lintel",
]
FETCHED = "2026-10-11T00:00:00Z"


def day(n: int) -> str:
    return (date(2010, 1, 1) + timedelta(days=7 * n)).isoformat()


def primary(eps: list[tuple[int, int, Episode]]) -> Bundle:
    return Bundle(
        show=ShowInfo(title="Testshow", year=2010, ids=ShowIds(tmdb=1)),
        episodes=[e for _, _, e in eps],
        orderings={"aired": Ordering(source="tmdb", type="aired", entries=[
            OrderingEntry(season=s, episode=n, episode_id=e.id) for s, n, e in eps])},
        provenance=[Provenance(adapter="tmdb", url="https://tmdb/1", fetched_at=FETCHED)],
    ).with_revision()


def secondary(eps: list[tuple[int, int, Episode]]) -> PartialBundle:
    return PartialBundle(
        show=ShowInfo(title="Testshow", ids=ShowIds(tvmaze=77)),
        episodes=[e for _, _, e in eps],
        orderings={"tvmaze-aired": Ordering(source="tvmaze", type="aired", entries=[
            OrderingEntry(season=s, episode=n, episode_id=e.id) for s, n, e in eps])},
        provenance=[Provenance(adapter="tvmaze", url="https://tvmaze/77", fetched_at=FETCHED)],
    )


def p_ep(k: int, title: str | None = None, **kw) -> Episode:
    kw.setdefault("air_date", day(k))
    kw.setdefault("runtime_min", 44)
    return Episode(id=f"tmdb:{100 + k}", title=title or TITLES[k],
                   synopses=[Synopsis(source="tmdb", text=f"primary synopsis {k}")], **kw)


def s_ep(k: int, title: str | None = None, **kw) -> Episode:
    kw.setdefault("air_date", day(k))
    return Episode(id=f"tvmaze:{500 + k}", title=title or TITLES[k],
                   synopses=[Synopsis(source="tvmaze", text=f"secondary synopsis {k}")], **kw)


def conflicts(b: Bundle, kind: str | None = None) -> list[dict]:
    return [p.conflict for p in b.provenance
            if p.conflict and (kind is None or p.conflict["kind"] == kind)]


def test_title_helpers():
    assert is_uninformative("XXXVII") and is_uninformative("Episode 5")
    assert is_uninformative("Episode XII") and is_uninformative("TBA")
    assert not is_uninformative("Weathervane") and not is_uninformative("Mix")
    assert split_part("The Glass Orchard (1)") == ("glass orchard", 1)
    assert split_part("The Glass Orchard, Part Two") == ("glass orchard", 2)
    assert split_part("Weathervane") is None


def test_clean_one_to_one_merge():
    P = primary([(1, k + 1, p_ep(k, runtime_min=None, directors=["Dee Rector"],
                                 guest_cast=[CastEntry(actor="Ann Actor")])) for k in range(4)])
    S = secondary([(1, k + 1, s_ep(k, runtime_min=43, production_code=f"P{k}",
                                   directors=["dee rector", "Second Director"],
                                   guest_cast=[CastEntry(actor="Ann Actor", character="Vera"),
                                               CastEntry(actor="Bo Actor", character="Lux")]))
                   for k in range(4)])
    out = merge_sources(P, [S], fetched_at=FETCHED)
    assert conflicts(out) == []
    e = out.episode_by_id()["tmdb:100"]
    assert e.title == TITLES[0] and e.alt_titles == []
    assert [s.source for s in e.synopses] == ["tmdb", "tvmaze"]
    assert e.runtime_min == 43 and e.production_code == "P0"
    assert e.directors == ["Dee Rector", "Second Director"]
    assert [(c.actor, c.character) for c in e.guest_cast] == [("Ann Actor", "Vera"),
                                                              ("Bo Actor", "Lux")]
    assert out.show.ids.tvmaze == 77 and out.show.ids.tmdb == 1
    assert out.orderings == P.orderings and [x.id for x in out.episodes] == \
        [x.id for x in P.episodes]
    assert out.bundle_revision == out.compute_revision() != P.bundle_revision
    assert any(p.adapter == "tvmaze" and p.conflict is None for p in out.provenance)


def test_runtime_kept_when_primary_has_one():
    P = primary([(1, 1, p_ep(0, runtime_min=44))])
    S = secondary([(1, 1, s_ep(0, runtime_min=30))])
    assert merge_sources(P, [S]).episodes[0].runtime_min == 44


def test_roman_numeral_titles_gain_alt_titles():
    P = primary([(1, k + 1, p_ep(k, title=r)) for k, r in enumerate(["I", "II", "III", "IV"])])
    S = secondary([(1, k + 1, s_ep(k)) for k in range(4)])
    out = merge_sources(P, [S])
    assert [e.alt_titles for e in out.episodes] == [[t] for t in TITLES[:4]]
    assert [e.title for e in out.episodes] == ["I", "II", "III", "IV"]


def test_alt_title_only_when_different_after_normalizing():
    P = primary([(1, 1, p_ep(0, title="Copper & Salt")), (1, 2, p_ep(1))])
    S = secondary([(1, 1, s_ep(0, title="copper and salt!")),
                   (1, 2, s_ep(1, title="Lanterns over Kettle-Bay"))])
    out = merge_sources(P, [S])
    assert [e.alt_titles for e in out.episodes] == [[], []]


def test_renumbered_seasons_align_by_content():
    # primary: one season of 6; secondary: two seasons of 3, and a 3-day air-date wobble
    P = primary([(1, k + 1, p_ep(k)) for k in range(6)])
    S = secondary([(1 + k // 3, k % 3 + 1, s_ep(k, air_date=(date.fromisoformat(day(k))
                                                             + timedelta(days=2)).isoformat()))
                   for k in range(6)])
    out = merge_sources(P, [S])
    assert all(len(e.synopses) == 2 for e in out.episodes)
    assert conflicts(out, "unaligned_primary") == conflicts(out, "unaligned_secondary") == []
    assert conflicts(out, "air_date") == []
    num = conflicts(out, "numbering")
    assert len(num) == 1 and num[0]["season"] == "1" and num[0]["primary"] == "tmdb:103"
    assert num[0]["secondary"] == "tvmaze:503"


def test_offset_numbering_with_extra_secondary_episode():
    P = primary([(1, k + 1, p_ep(k)) for k in range(3)] + [(1, 4, p_ep(4))])
    extra = s_ep(3, title="An Unaired Detour", air_date=None)
    S = secondary([(1, 1, s_ep(0)), (1, 2, s_ep(1)), (1, 3, s_ep(2)), (1, 4, extra),
                   (1, 5, s_ep(4))])
    out = merge_sources(P, [S])
    un = conflicts(out, "unaligned_secondary")
    assert [c["secondary"] for c in un] == ["tvmaze:503"]
    assert len(out.episode_by_id()["tmdb:104"].synopses) == 2
    assert "tvmaze:503" not in out.episode_by_id()  # never added
    assert len(conflicts(out, "numbering")) == 1


def test_missing_secondary_episode():
    P = primary([(1, k + 1, p_ep(k)) for k in range(4)])
    S = secondary([(1, 1, s_ep(0)), (1, 2, s_ep(1)), (1, 3, s_ep(3))])
    out = merge_sources(P, [S])
    assert [(c["primary"], c["season"]) for c in conflicts(out, "unaligned_primary")] == \
        [("tmdb:102", "1")]
    assert len(out.episode_by_id()["tmdb:103"].synopses) == 2
    assert len(out.episode_by_id()["tmdb:102"].synopses) == 1


def test_uninformative_titles_without_dates_are_not_merged():
    P = primary([(1, 1, p_ep(0, title="I", air_date=None))])
    S = secondary([(1, 1, s_ep(0, title="Episode 1", air_date=None))])
    out = merge_sources(P, [S])
    assert len(out.episodes[0].synopses) == 1
    assert len(conflicts(out, "unaligned_primary")) == 1


def test_air_date_conflict_keeps_primary():
    P = primary([(1, 1, p_ep(0)), (1, 2, p_ep(1))])
    S = secondary([(1, 1, s_ep(0, air_date="2011-06-01")), (1, 2, s_ep(1))])
    out = merge_sources(P, [S], fetched_at="2026-01-01T00:00:00Z")
    c = conflicts(out, "air_date")
    assert len(c) == 1 and c[0]["primary"] == "tmdb:100" and c[0]["secondary"] == "tvmaze:500"
    assert out.episodes[0].air_date == day(0)
    assert len(out.episodes[0].synopses) == 2  # still merged
    pr = next(p for p in out.provenance if p.conflict)
    assert pr.adapter == "tvmaze" and pr.url == "https://tvmaze/77"
    assert pr.fetched_at == "2026-01-01T00:00:00Z"
    assert all(isinstance(v, str) for v in pr.conflict.values())


def test_air_date_filled_when_missing():
    P = primary([(1, 1, p_ep(0, air_date=None))])
    S = secondary([(1, 1, s_ep(0))])
    assert merge_sources(P, [S]).episodes[0].air_date == day(0)


def test_specials_are_quiet():
    P = primary([(0, 1, p_ep(6, title="Behind the Weathervane")),
                 (1, 1, p_ep(0)), (1, 2, p_ep(1))])
    S = secondary([(0, 1, s_ep(5, title="A Promo Reel Nobody Asked For")),
                   (0, 2, s_ep(7, title="Another Featurette", air_date=None)),
                   (1, 1, s_ep(0)), (1, 2, s_ep(1)),
                   (1, 3, s_ep(6, title="Behind the Weathervane"))])  # special placed in season
    out = merge_sources(P, [S])
    assert conflicts(out) == []
    assert out.episode_by_id()["tmdb:106"].synopses[-1].source == "tvmaze"


def test_split_with_shared_base_and_valid_fix(tmp_path):
    P = primary([(3, 1, p_ep(0)), (3, 2, p_ep(1, title="XXXVII", runtime_min=44)),
                 (3, 3, p_ep(2)), (4, 1, p_ep(4))])
    S = secondary([(3, 1, s_ep(0)),
                   (3, 2, s_ep(1, title="The Glass Orchard (1)", runtime_min=22,
                               production_code="X1")),
                   (3, 3, s_ep(8, title="The Glass Orchard (2)", runtime_min=22,
                               air_date=day(1))),
                   (3, 4, s_ep(2)), (4, 1, s_ep(4))])
    out = merge_sources(P, [S])
    sp = conflicts(out, "split")
    assert len(sp) == 1
    c = sp[0]
    assert c["primary"] == "tmdb:101" and c["secondary"] == "tvmaze:501, tvmaze:508"
    assert c["season"] == "3"
    assert conflicts(out, "unaligned_secondary") == conflicts(out, "unaligned_primary") == []
    whole = out.episode_by_id()["tmdb:101"]
    assert whole.alt_titles == ["The Glass Orchard"] and len(whole.synopses) == 1
    # the fix is paste-ready: the overrides loader accepts it on the merged bundle
    doc = yaml.safe_load(c["fix"])
    assert set(doc) == {"new_episodes", "orderings"}
    ov = tmp_path / "overrides.yaml"
    ov.write_text(c["fix"])
    b = apply_overrides(out, ov)
    ids = b.episode_by_id()
    assert ids["custom:tvmaze-501"].title == "The Glass Orchard (1)"
    assert ids["custom:tvmaze-501"].runtime_min == 22
    assert ids["custom:tvmaze-501"].synopses[0].source == "tvmaze"
    order = b.orderings["aired-split"]
    s3 = [(e.episode, e.episode_id) for e in order.entries if e.season == 3]
    assert s3 == [(1, "tmdb:100"), (2, "custom:tvmaze-501"), (3, "custom:tvmaze-508"),
                  (4, "tmdb:102")]
    assert [e.episode_id for e in order.entries if e.season == 4] == ["tmdb:104"]
    assert merge_sources(out, [S]).model_dump() == out.model_dump()  # idempotent with a split


def test_split_by_runtime_sum():
    P = primary([(1, 1, p_ep(0)), (1, 2, p_ep(1, title="XL", runtime_min=46)), (1, 3, p_ep(2))])
    S = secondary([(1, 1, s_ep(0)),
                   (1, 2, s_ep(1, title="Lanterns", runtime_min=23)),
                   (1, 3, s_ep(9, title="Kettle Bay", runtime_min=23, air_date=day(1))),
                   (1, 4, s_ep(2))])
    out = merge_sources(P, [S])
    sp = conflicts(out, "split")
    assert len(sp) == 1 and sp[0]["secondary"] == "tvmaze:501, tvmaze:509"
    assert out.episode_by_id()["tmdb:101"].alt_titles == []


def test_inputs_unchanged_and_deterministic():
    P = primary([(1, k + 1, p_ep(k, title=["I", "II", "III"][k])) for k in range(3)])
    S = secondary([(1, k + 1, s_ep(k, guest_cast=[CastEntry(actor="Z")])) for k in range(3)])
    before_p, before_s = P.model_dump(), S.model_dump()
    a = merge_sources(P, [S], fetched_at=FETCHED)
    b = merge_sources(P, [S], fetched_at=FETCHED)
    assert P.model_dump() == before_p and S.model_dump() == before_s
    assert a.model_dump() == b.model_dump()


def test_idempotent():
    P = primary([(1, k + 1, p_ep(k, title=["I", "II", "III", "IV"][k])) for k in range(4)])
    S = secondary([(1, 1, s_ep(0)), (1, 2, s_ep(1, air_date="2012-01-01")), (1, 3, s_ep(3))])
    once = merge_sources(P, [S], fetched_at=FETCHED)
    twice = merge_sources(once, [S], fetched_at=FETCHED)
    assert twice.model_dump() == once.model_dump()
    assert twice.bundle_revision == once.bundle_revision
    tmdb = [p for p in twice.provenance if p.adapter == "tmdb"]
    assert len(tmdb) == 1


def test_no_secondaries_is_identity_on_content():
    P = primary([(1, 1, p_ep(0))])
    out = merge_sources(P, [])
    assert out.model_dump() == P.model_dump()


def test_split_across_reordered_seasons_with_leading_article(tmp_path):
    # Shape of a real miss: the primary has the two-parter late in season 2 as one entry without
    # the article; the secondary puts its parts at the end of season 1, before season 2's
    # earlier-dated episodes, with placeholder runtimes that don't sum to the primary's.
    P = primary([(1, 1, p_ep(0, title="I")), (1, 2, p_ep(1, title="II")),
                 (2, 1, p_ep(3, title="IV")), (2, 2, p_ep(4, title="V")),
                 (2, 3, p_ep(10, title="Glass Orchard", runtime_min=44))])
    S = secondary([(1, 1, s_ep(0)), (1, 2, s_ep(1)),
                   (1, 3, s_ep(10, title="The Glass Orchard (1)", runtime_min=30)),
                   (1, 4, s_ep(11, title="The Glass Orchard (2)", runtime_min=30,
                               air_date=day(10))),
                   (2, 1, s_ep(3)), (2, 2, s_ep(4))])
    out = merge_sources(P, [S])
    sp = conflicts(out, "split")
    assert len(sp) == 1 and sp[0]["primary"] == "tmdb:110"
    assert sp[0]["secondary"] == "tvmaze:510, tvmaze:511"
    assert conflicts(out, "unaligned_secondary") == conflicts(out, "unaligned_primary") == []
    ov = tmp_path / "overrides.yaml"
    ov.write_text(sp[0]["fix"])
    b = apply_overrides(out, ov)
    assert [e.episode_id for e in b.orderings["aired-split"].entries if e.season == 2] == \
        ["tmdb:103", "tmdb:104", "custom:tvmaze-510", "custom:tvmaze-511"]


def test_same_night_episodes_with_different_titles_do_not_pair():
    P = primary([(1, 1, p_ep(0)), (1, 2, p_ep(1, title="Weathervane", air_date=day(0)))])
    S = secondary([(1, 1, s_ep(0)), (1, 2, s_ep(1, title="Copper and Salt", air_date=day(0)))])
    out = merge_sources(P, [S])
    assert [c["primary"] for c in conflicts(out, "unaligned_primary")] == ["tmdb:101"]
    assert [c["secondary"] for c in conflicts(out, "unaligned_secondary")] == ["tvmaze:501"]
    assert out.episodes[1].alt_titles == []


def test_pilot_is_not_a_real_title():
    P = primary([(1, 1, p_ep(0, title="Pilot")), (1, 2, p_ep(1))])
    S = secondary([(1, 1, s_ep(0)), (1, 2, s_ep(1))])
    out = merge_sources(P, [S])
    assert out.episodes[0].alt_titles == [TITLES[0]] and conflicts(out) == []


def test_numeral_titles_pair_within_three_days_not_a_week():
    shifted = (date.fromisoformat(day(1)) + timedelta(days=3)).isoformat()
    P = primary([(1, 1, p_ep(0, title="I")), (1, 2, p_ep(1, title="II")),
                 (1, 3, p_ep(2, title="III"))])
    # the secondary lacks the first episode and has an extra one a week off the third
    S = secondary([(1, 1, s_ep(1, air_date=shifted)),
                   (1, 2, s_ep(5, air_date=(date.fromisoformat(day(2))
                                            + timedelta(days=7)).isoformat()))])
    out = merge_sources(P, [S])
    assert out.episode_by_id()["tmdb:101"].alt_titles == [TITLES[1]]
    assert [c["primary"] for c in conflicts(out, "unaligned_primary")] == ["tmdb:100", "tmdb:102"]
    assert [c["secondary"] for c in conflicts(out, "unaligned_secondary")] == ["tvmaze:505"]


def test_split_part_marker_between_title_halves():
    assert split_part("Into the Orchard Part I: Lambs") == ("into the orchard lambs", 1)
    assert split_part("Into the Orchard - Lambs (1)") == ("into the orchard lambs", 1)
    assert split_part("Part 2: Lambs") is None


def test_different_part_numbers_never_pair():
    # the secondary lacks part 1 and dates part 2 on part 1's night: the shared base must not
    # pair part 1 with part 2's data
    P = primary([(1, 1, p_ep(0, title="Silver Kettle (1)")),
                 (1, 2, p_ep(1, title="Silver Kettle (2)")), (1, 3, p_ep(2))])
    S = secondary([(1, 1, s_ep(1, title="Silver Kettle (2)", air_date=day(0))), (1, 2, s_ep(2))])
    out = merge_sources(P, [S])
    ids = out.episode_by_id()
    assert len(ids["tmdb:100"].synopses) == 1
    assert ids["tmdb:101"].synopses[-1].text == "secondary synopsis 1"
    assert [c["primary"] for c in conflicts(out, "unaligned_primary")] == ["tmdb:100"]


def test_no_split_when_the_primary_already_has_the_parts():
    # shape of Cowboy Bebop: both sources have parts 1 and 2, the secondary numbers and dates
    # them elsewhere; that is two pairs, not a split with a harmful fix
    P = primary([(1, 1, p_ep(0)), (1, 2, p_ep(1, title="Silver Kettle (1)")),
                 (1, 3, p_ep(2, title="Silver Kettle (2)")), (1, 4, p_ep(3))])
    S = secondary([(1, 1, s_ep(0)), (1, 2, s_ep(3)),
                   (1, 3, s_ep(1, title="Silver Kettle (Part 1)", air_date=day(30))),
                   (1, 4, s_ep(2, title="Silver Kettle (Part 2)", air_date=day(31)))])
    out = merge_sources(P, [S])
    assert conflicts(out, "split") == []
    assert conflicts(out, "unaligned_primary") == conflicts(out, "unaligned_secondary") == []
    ids = out.episode_by_id()
    assert ids["tmdb:101"].synopses[-1].text == "secondary synopsis 1"
    assert ids["tmdb:102"].synopses[-1].text == "secondary synopsis 2"
    assert len(conflicts(out, "air_date")) == 2
    assert merge_sources(out, [S]).model_dump() == out.model_dump()


def test_unique_identical_title_pairs_whatever_the_dates():
    far = (date.fromisoformat(day(1)) + timedelta(days=100)).isoformat()
    P = primary([(1, k + 1, p_ep(k)) for k in range(4)])
    S = secondary([(1, 1, s_ep(0)), (1, 2, s_ep(2)), (1, 3, s_ep(3)),
                   (1, 4, s_ep(1, air_date=far))])  # moved to the end and dated 100 days later
    out = merge_sources(P, [S])
    assert conflicts(out, "unaligned_primary") == conflicts(out, "unaligned_secondary") == []
    assert out.episode_by_id()["tmdb:101"].synopses[-1].text == "secondary synopsis 1"
    assert [c["primary"] for c in conflicts(out, "air_date")] == ["tmdb:101"]
    assert merge_sources(out, [S]).model_dump() == out.model_dump()


def test_identical_title_far_apart_needs_uniqueness():
    # two secondary episodes carry the title: no date evidence, so neither pairs
    far = (date.fromisoformat(day(1)) + timedelta(days=100)).isoformat()
    farther = (date.fromisoformat(day(1)) + timedelta(days=200)).isoformat()
    P = primary([(1, 1, p_ep(0)), (1, 2, p_ep(1))])
    S = secondary([(1, 1, s_ep(0)), (2, 1, s_ep(1, air_date=far)),
                   (2, 2, s_ep(9, title=TITLES[1], air_date=farther))])
    out = merge_sources(P, [S])
    assert [c["primary"] for c in conflicts(out, "unaligned_primary")] == ["tmdb:101"]
    assert len(out.episode_by_id()["tmdb:101"].synopses) == 1
