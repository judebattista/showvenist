import json
from pathlib import Path

import httpx
import pytest

from showvenist.adapters.tvmaze import TvmazeAdapter, TvmazeError, tvmaze_id_from_ref

FIX = Path(__file__).parent / "fixtures" / "tvmaze"


def load(name):
    return json.loads((FIX / name).read_text())


def adapter(handler=None, sleeps=None):
    reqs = []

    def default(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/search/shows":
            return httpx.Response(200, json=load("search.json"))
        if path == "/shows/9001":
            return httpx.Response(200, json=load("show.json"))
        if path == "/shows/9001/episodes":
            return httpx.Response(200, json=load("episodes.json"))
        return httpx.Response(404)

    h = handler or default

    def wrapped(req):
        reqs.append(req)
        return h(req)

    client = httpx.Client(transport=httpx.MockTransport(wrapped))
    sl = sleeps if sleeps is not None else []
    return TvmazeAdapter(client, sleep=sl.append), reqs


def test_search():
    ad, reqs = adapter()
    res = ad.search("Quillfeather Academy")
    assert res[0] == {
        "tvmaze_id": 9001,
        "title": "Quillfeather Academy",
        "year": 2011,
        "status": "Ended",
    }
    assert len(res) == 2
    assert reqs[0].url.params["q"] == "Quillfeather Academy"
    assert [r["tvmaze_id"] for r in ad.search("Quillfeather Academy", 2020)] == [9002]
    # premiere years may differ by one between sources
    assert [r["tvmaze_id"] for r in ad.search("Quillfeather Academy", 2012)] == [9001]
    assert ad.search("Quillfeather Academy", 2013) == []


def test_fetch_mapping():
    ad, reqs = adapter()
    pb = ad.fetch(9001)
    assert pb.show.title == "Quillfeather Academy"
    assert pb.show.year == 2011
    assert pb.show.in_production is False
    assert pb.show.ids.tvmaze == 9001
    assert any(r.url.params.get("specials") == "1" for r in reqs)
    by = {e.id: e for e in pb.episodes}
    assert set(by) == {f"tvmaze:{i}" for i in range(70001, 70007)}
    e1 = by["tvmaze:70001"]
    assert e1.title == "The Lantern Door (1)"
    assert e1.air_date == "2011-03-04"
    assert e1.runtime_min == 30
    assert e1.production_code is None
    assert by["tvmaze:70002"].runtime_min == 24  # averageRuntime fallback
    assert by["tvmaze:70003"].air_date is None
    assert [u.url for u in pb.provenance if u.adapter == "tvmaze"]
    assert all(u.fetched_at for u in pb.provenance)


def test_html_stripping_and_no_synopsis():
    pb = adapter()[0].fetch(9001)
    by = {e.id: e for e in pb.episodes}
    syn = by["tvmaze:70001"].synopses
    assert len(syn) == 1 and syn[0].source == "tvmaze"
    assert syn[0].text == "Mira finds a lantern & a locked door. Nobody believes her."
    assert by["tvmaze:70003"].synopses == []  # null
    assert by["tvmaze:70004"].synopses == []  # empty


def test_ordering_and_specials():
    pb = adapter()[0].fetch(9001)
    assert list(pb.orderings) == ["tvmaze-aired"]
    o = pb.orderings["tvmaze-aired"]
    assert o.source == "tvmaze" and o.type == "aired"
    got = [(e.season, e.episode, e.episode_id) for e in o.entries]
    assert got == [
        (1, 1, "tvmaze:70001"),
        (1, 2, "tvmaze:70002"),
        (1, 3, "tvmaze:70003"),
        (0, 1, "tvmaze:70004"),
        (0, 2, "tvmaze:70005"),
        (2, 1, "tvmaze:70006"),
    ]


def test_not_found_is_error():
    ad, _ = adapter()
    with pytest.raises(TvmazeError, match="404"):
        ad.fetch(1)


def test_retry_on_429_then_success():
    calls = {"n": 0}

    def h(req):
        if req.url.path == "/shows/9001":
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, headers={"Retry-After": "3"})
            return httpx.Response(200, json=load("show.json"))
        return httpx.Response(200, json=[])

    sleeps = []
    ad, _ = adapter(h, sleeps)
    pb = ad.fetch(9001)
    assert pb.show.title == "Quillfeather Academy"
    assert sleeps == [3.0, 3.0]


def test_retry_exhausted_5xx_backoff():
    ad, reqs = adapter(lambda r: httpx.Response(503), sleeps=(s := []))
    with pytest.raises(TvmazeError, match="503"):
        ad.fetch(9001)
    assert len(reqs) == 4 and s == [1.0, 2.0, 4.0]


def test_no_secrets_in_errors():
    def boom(req):
        raise httpx.ConnectError("secret-token-123 failed", request=req)

    ad, _ = adapter(boom)
    with pytest.raises(TvmazeError) as ei:
        ad.search("x")
    assert "secret" not in str(ei.value)


@pytest.mark.parametrize(
    "ref,expected",
    [
        ("https://www.tvmaze.com/shows/1603/samurai-jack", 1603),
        ("http://tvmaze.com/shows/1603", 1603),
        ("tvmaze:1603", 1603),
        ("https://www.themoviedb.org/tv/1603", None),
        ("tvmaze:abc", None),
        ("1603", None),
    ],
)
def test_ref_parsing(ref, expected):
    assert tvmaze_id_from_ref(ref) == expected
