import json
from pathlib import Path

import httpx
import pytest

from showvenist.adapters.tmdb import TmdbAdapter, TmdbError

FIX = Path(__file__).parent / "fixtures" / "tmdb"
V3 = "0123456789abcdef0123456789abcdef"
V4 = "eyJhbGciOiJIUzI1NiJ9." + "x" * 60 + ".sig"


def load(name):
    return json.loads((FIX / name).read_text())


def make_handler(requests, show=None, groups=None, details=None):
    show = show or load("show.json")
    groups = groups or load("episode_groups.json")
    details = details or {g["id"]: load(f"group_{g['id']}.json") for g in groups["results"]}

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        path = req.url.path.removeprefix("/3")
        if path == "/tv/1437":
            ap = req.url.params.get("append_to_response", "")
            if ap.startswith("season/"):
                return httpx.Response(200, json=show)
            return httpx.Response(200, json={k: v for k, v in show.items() if "/" not in k})
        if path == "/tv/1437/aggregate_credits":
            return httpx.Response(200, json=load("aggregate_credits.json"))
        if path == "/tv/1437/episode_groups":
            return httpx.Response(200, json=groups)
        if path.startswith("/tv/episode_group/"):
            return httpx.Response(200, json=details[path.rsplit("/", 1)[1]])
        if path == "/search/tv":
            return httpx.Response(
                200,
                json={"results": [{"id": 1437, "name": "Firefly", "first_air_date": "2002-09-20"}]},
            )
        if path.startswith("/find/"):
            return httpx.Response(200, json={"tv_results": [{"id": 1437, "name": "Firefly"}]})
        return httpx.Response(404)

    return handler


def adapter(key=V3, **kw):
    reqs = []
    client = httpx.Client(transport=httpx.MockTransport(make_handler(reqs, **kw)))
    return TmdbAdapter(key, client, sleep=lambda s: None), reqs


def test_fetch_firefly():
    ad, _ = adapter()
    pb = ad.fetch(1437)
    assert pb.show.title == "Firefly"
    assert pb.show.year == 2002
    assert pb.show.ids.tmdb == 1437
    aired = pb.orderings["aired"]
    assert aired.source == "tmdb" and aired.type == "aired"
    ids = {e.id for e in pb.episodes}
    assert {f"tmdb:{x}" for x in map(lambda e: e.episode_id.split(":")[1], aired.entries)} <= ids
    assert all(e.id.startswith("tmdb:") for e in pb.episodes)
    assert any(e.writers or e.directors for e in pb.episodes)


def test_specials_in_season_zero():
    ad, _ = adapter()
    pb = ad.fetch(1437)
    seasons = {e.season for e in pb.orderings["aired"].entries}
    assert seasons == {0, 1}
    assert len(pb.orderings["aired"].entries) == 23


def test_groups_become_orderings_with_rank():
    ad, _ = adapter()
    pb = ad.fetch(1437)
    io = pb.orderings["intended-order"]
    assert (io.type, io.type_rank, io.name) == ("absolute", 1, "Intended Order")
    assert io.tmdb_group == "5acfb484c3a368738a004068"
    dvd = pb.orderings["dvd-order"]
    assert (dvd.type, dvd.type_rank) == ("dvd", 1)
    assert dvd.entries and min(e.episode for e in dvd.entries) == 1
    ids = {e.id for e in pb.episodes}
    assert all(e.episode_id in ids for o in pb.orderings.values() for e in o.entries)


def test_provenance_has_no_key_and_one_per_url():
    ad, reqs = adapter(V3)
    pb = ad.fetch(1437)
    assert len(pb.provenance) == len(reqs)
    assert all(V3 not in (p.url or "") and "api_key" not in (p.url or "") for p in pb.provenance)
    assert all(p.adapter == "tmdb" for p in pb.provenance)


def group_fixture(names_types):
    results, details = [], {}
    for i, (name, typ) in enumerate(names_types):
        gid = f"g{i}"
        results.append({"id": gid, "name": name, "type": typ})
        details[gid] = {
            "id": gid,
            "groups": [
                {
                    "name": "Season 1",
                    "order": 1,
                    "episodes": [
                        {
                            "id": 900 + i,
                            "name": f"Extra {i}",
                            "order": 0,
                            "season_number": 1,
                            "episode_number": 1,
                        }
                    ],
                }
            ],
        }
    return {"results": results}, details


def test_slug_collisions_rank_and_missing_episodes_added():
    groups, details = group_fixture(
        [("Intended Order", 3), ("Intended Order", 3), ("Intended order!", 2), ("Story", 3)]
    )
    ad, _ = adapter(groups=groups, details=details)
    pb = ad.fetch(1437)
    assert {"intended-order", "intended-order-2", "intended-order-3", "story"} <= set(pb.orderings)
    assert pb.orderings["intended-order"].type_rank == 1
    assert pb.orderings["intended-order-2"].type_rank == 2
    assert pb.orderings["intended-order-3"].type == "absolute"
    assert pb.orderings["intended-order-3"].type_rank == 1
    assert pb.orderings["story"].type_rank == 3
    assert {"tmdb:900", "tmdb:903"} <= {e.id for e in pb.episodes}


def test_v3_auth_uses_query_param():
    ad, reqs = adapter(V3)
    ad.fetch(1437)
    assert all(r.url.params.get("api_key") == V3 for r in reqs)
    assert all("authorization" not in r.headers for r in reqs)


def test_v4_auth_uses_bearer():
    ad, reqs = adapter(V4)
    ad.fetch(1437)
    assert all(r.headers["authorization"] == f"Bearer {V4}" for r in reqs)
    assert all("api_key" not in r.url.params for r in reqs)


def test_search_and_find():
    ad, _ = adapter()
    r = ad.search("Firefly", 2002)
    assert r[0]["tmdb_id"] == 1437 and r[0]["year"] == 2002
    assert ad.find_by_imdb("tt0303461")[0]["tmdb_id"] == 1437


def test_retry_after_honoured():
    calls, sleeps = [], []

    def handler(req):
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"results": []})

    ad = TmdbAdapter(V3, httpx.Client(transport=httpx.MockTransport(handler)), sleep=sleeps.append)
    assert ad.search("x") == []
    assert sleeps == [3.0, 3.0]


def test_retry_gives_up_after_three_and_error_hides_key():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, headers={"Retry-After": "0"})

    ad = TmdbAdapter(V3, httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda s: None)
    with pytest.raises(TmdbError) as ei:
        ad.search("x")
    assert len(calls) == 4
    assert V3 not in str(ei.value)


def test_series_cast_from_aggregate_credits():
    ad, reqs = adapter()
    pb = ad.fetch(1437)
    assert len(pb.series_cast) == 15
    first = pb.series_cast[0]
    assert (first.actor, first.character) == ("Nathan Fillion", "Mal Reynolds")
    assert any(r.url.path.endswith("/tv/1437/aggregate_credits") for r in reqs)
