import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from showvenist.adapters.opensubtitles import (
    STATE_NAME,
    OpenSubtitlesAdapter,
    OpenSubtitlesError,
    QuotaExhausted,
    rank_candidates,
)
from showvenist.bundle.schema import Bundle, Episode, Ordering, OrderingEntry, ShowIds, ShowInfo

FIX = Path(__file__).parent / "fixtures" / "opensubtitles"
KEY, USER, PW, TOKEN = "KEY-secret", "alice", "pw-secret", "tok-abc"
SRT = b"".join(
    f"{i}\n00:00:0{i},000 --> 00:00:0{i},500\n{'hi' if i == 3 else 'yo'}\n\n".encode()
    for i in (1, 2, 3)
)


def load(name):
    return json.loads((FIX / name).read_text())


class Server:
    """Hand-written OpenSubtitles stand-in. Episode N (1..n) has a search hit unless missing."""

    def __init__(self, n=4, quota=None, missing=()):
        self.reqs: list[httpx.Request] = []
        self.quota = quota  # downloads allowed before 406
        self.missing = set(missing)
        self.downloads: list[int] = []
        self.rate_limits = 0

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.reqs.append(req)
        p = req.url.path
        if p.endswith("/login"):
            return httpx.Response(200, json=load("login.json"))
        if req.url.host.startswith("dl."):
            return httpx.Response(200, content=SRT)
        assert req.headers["api-key"] == KEY and req.headers["user-agent"] == "showvenist v0.1"
        if p.endswith("/subtitles"):
            assert "authorization" not in req.headers  # search needs only the Api-Key
        else:
            assert req.headers["authorization"] == f"Bearer {TOKEN}"
            assert req.url.host == "vip-api.opensubtitles.com"
        if p.endswith("/subtitles"):
            if self.rate_limits:
                self.rate_limits -= 1
                return httpx.Response(429, headers={"Retry-After": "2"}, json={})
            e = int(req.url.params["episode_number"])
            if e in self.missing:
                return httpx.Response(200, json=load("search_empty.json"))
            data = load("search_s1e1.json")
            for r in data["data"]:  # make file ids unique per episode
                for f in r["attributes"]["files"]:
                    f["file_id"] += 1000 * e
                r["attributes"]["feature_details"] = {
                    "season_number": int(req.url.params["season_number"]), "episode_number": e,
                    "parent_tmdb_id": 77,
                }
            return httpx.Response(200, json=data)
        if p.endswith("/download"):
            fid = json.loads(req.content)["file_id"]
            if self.quota is not None and len(self.downloads) >= self.quota:
                return httpx.Response(406, json=load("download_quota.json"))
            self.downloads.append(fid)
            body = load("download_ok.json")
            body["link"] = f"https://dl.example/{fid}.srt"
            return httpx.Response(200, json=body)
        raise AssertionError(p)


def adapter(server, **kw):
    c = httpx.Client(transport=httpx.MockTransport(server))
    kw.setdefault("sleep", lambda s: None)
    kw.setdefault("clock", lambda: 0.0)
    return OpenSubtitlesAdapter(KEY, USER, PW, client=c, **kw)


def bundle(n=4, tmdb=77, imdb=None):
    eps = [Episode(id=f"tmdb:{i}", title=f"t{i}") for i in range(1, n + 1)]
    ents = [OrderingEntry(season=1, episode=i, episode_id=f"tmdb:{i}") for i in range(1, n + 1)]
    ents.append(OrderingEntry(season=2, episode=1, episode_id="tmdb:99"))
    eps.append(Episode(id="tmdb:99", title="s2"))
    return Bundle(
        show=ShowInfo(title="X", ids=ShowIds(tmdb=tmdb, imdb=imdb)),
        episodes=eps,
        orderings={"aired": Ordering(source="tmdb", type="aired", entries=ents)},
    )


def test_login_and_base_url():
    s = Server()
    a = adapter(s)
    hit = a.search_episode(parent_tmdb_id=77, season=1, episode=1, language="en")
    assert hit is not None
    assert [r.url.path for r in s.reqs] == ["/api/v1/subtitles"]  # no login for search
    assert s.reqs[0].url.host == "api.opensubtitles.com"
    q = s.reqs[0].url.params
    assert q["parent_tmdb_id"] == "77" and q["season_number"] == "1" and q["type"] == "episode"
    a.download(5)
    login = next(r for r in s.reqs if r.url.path.endswith("/login"))
    assert json.loads(login.content) == {"username": USER, "password": PW}
    assert "api-key" in login.headers and "authorization" not in login.headers
    dl = next(r for r in s.reqs if r.url.path.endswith("/download"))
    assert dl.headers["accept"] == "*/*" and dl.url.host == "vip-api.opensubtitles.com"
    a.download(6)
    assert sum(r.url.path.endswith("/login") for r in s.reqs) == 1  # token cached


def test_imdb_without_leading_zeros_and_canonical_query():
    s = Server()
    a = adapter(s)
    s.reqs.clear()
    a.search_episode(imdb="tt0108778", season=1, episode=2, language="EN")
    q = s.reqs[-1].url
    assert q.params["parent_imdb_id"] == "108778"
    assert str(q.query, "ascii") == (
        "episode_number=2&languages=en&parent_imdb_id=108778&season_number=1&type=episode"
    )


def test_login_failure_hides_secrets():
    def h(req):
        return httpx.Response(401, json={"message": "bad"})

    a = OpenSubtitlesAdapter(KEY, USER, PW, client=httpx.Client(transport=httpx.MockTransport(h)))
    with pytest.raises(OpenSubtitlesError) as ei:
        a.login()
    assert not any(x in str(ei.value) for x in (KEY, USER, PW))


def test_ranking_rules():
    results = load("search_s1e1.json")["data"]
    ranked = rank_candidates(results, "en")
    # machine/AI and French excluded; non-SDH first by downloads; SDH last
    assert [f for f, _ in ranked] == [106, 105, 104]
    # SDH accepted when it is all there is
    only_sdh = [r for r in results if r["id"] == "4"]
    assert [f for f, _ in rank_candidates(only_sdh, "en")] == [104]
    assert [f for f, _ in rank_candidates(results, "fr")] == [103]
    assert rank_candidates(results, "de") == []


def test_ranking_text_format():
    def r(fid, name, dl):
        return {"attributes": {"language": "en", "download_count": dl,
                               "files": [{"file_id": fid, "file_name": name}]}}

    ranked = rank_candidates([r(1, "a.sub", 99), r(2, "b.ass", 50), r(3, "c.srt", 10)], "en")
    assert [f for f, _ in ranked] == [3, 2]  # image .sub dropped; srt preferred over ass


def test_download_quota_error():
    s = Server(quota=0)
    a = adapter(s)
    with pytest.raises(QuotaExhausted) as ei:
        a.download(5)
    assert ei.value.reset_time == "2026-10-06T00:00:00.000Z"
    assert a.remaining == 0


def test_429_retry_after():
    s = Server()
    s.rate_limits = 2
    sleeps = []
    a = adapter(s, sleep=sleeps.append, min_interval=0.0)
    assert a.search_episode(parent_tmdb_id=77, season=1, episode=1) is not None
    assert sleeps == [2.0, 2.0]
    s.rate_limits = 99
    with pytest.raises(OpenSubtitlesError):
        a.search_episode(parent_tmdb_id=77, season=1, episode=1)


def test_season_quota_then_resume(tmp_path: Path):
    s = Server(quota=2)
    a = adapter(s, now=lambda: datetime(2026, 10, 5, 12, tzinfo=UTC))
    b = bundle()
    part, rep = a.fetch_season_subs(b, tmp_path, season=1)
    assert rep.downloaded == ["tmdb:1", "tmdb:2"]
    assert rep.quota_stopped == ["tmdb:3", "tmdb:4"]
    assert rep.reset_time == "2026-10-06T00:00:00.000Z"
    assert (tmp_path / "subs" / "tmdb-1.en.srt").read_text().endswith("hi\n\n")
    ref = part.episodes[0].subtitles[0]
    assert (ref.source, ref.lang, ref.path) == ("opensubtitles", "en", "subs/tmdb-1.en.srt")
    assert part.provenance and all(p.adapter == "opensubtitles" for p in part.provenance)

    # Same day: state says blocked, so no network at all.
    n = len(s.reqs)
    _, rep2 = a.fetch_season_subs(b, tmp_path, season=1)
    assert len(s.reqs) == n and rep2.quota_stopped == ["tmdb:3", "tmdb:4"]
    assert rep2.skipped == ["tmdb:1", "tmdb:2"]

    # After reset: the remaining two only; merge the first run's refs into the bundle.
    s.quota = None
    done = {e.id: e for e in part.episodes}
    b2 = b.model_copy(update={"episodes": [done.get(e.id, e) for e in b.episodes]})
    a2 = adapter(s, now=lambda: datetime(2026, 10, 6, 1, tzinfo=UTC))
    before = list(s.downloads)
    part3, rep3 = a2.fetch_season_subs(b2, tmp_path, season=1)
    assert rep3.downloaded == ["tmdb:3", "tmdb:4"] and not rep3.quota_stopped
    assert rep3.skipped == ["tmdb:1", "tmdb:2"]
    assert len(s.downloads) == len(before) + 2 and not set(before) & set(s.downloads[2:])
    state = json.loads((tmp_path / "subs" / STATE_NAME).read_text())
    assert "quota_reset" not in state and len(state["done"]) == 4


def test_file_on_disk_skipped_and_season_filter(tmp_path: Path):
    (tmp_path / "subs").mkdir()
    (tmp_path / "subs" / "tmdb-1.en.srt").write_text("x")
    s = Server()
    a = adapter(s)
    part, rep = a.fetch_season_subs(bundle(), tmp_path, season=1)
    assert rep.skipped == ["tmdb:1"] and rep.downloaded == ["tmdb:2", "tmdb:3", "tmdb:4"]
    assert "tmdb:99" not in rep.downloaded  # season 2 not requested
    assert {e.id for e in part.episodes} == {"tmdb:1", "tmdb:2", "tmdb:3", "tmdb:4"}


def test_not_found_and_language(tmp_path: Path):
    s = Server(missing={2})
    a = adapter(s)
    part, rep = a.fetch_season_subs(bundle(3), tmp_path, season=1)
    assert rep.not_found == ["tmdb:2"] and rep.downloaded == ["tmdb:1", "tmdb:3"]
    state = json.loads((tmp_path / "subs" / STATE_NAME).read_text())
    assert "tmdb:2" in state["not_found"]
    # language filter: fr picks the French file and names the path accordingly
    part, rep = adapter(Server()).fetch_season_subs(bundle(1), tmp_path / "fr", season=1,
                                                    language="fr")
    assert part.episodes[0].subtitles[0].path == "subs/tmdb-1.fr.srt"
    assert (tmp_path / "fr" / "subs" / "tmdb-1.fr.srt").is_file()


def test_failed_episode_recorded(tmp_path: Path):
    def h(req):
        if req.url.path.endswith("/login"):
            return httpx.Response(200, json=load("login.json"))
        if req.url.path.endswith("/subtitles"):
            return httpx.Response(500)
        raise AssertionError

    a = OpenSubtitlesAdapter(KEY, USER, PW, client=httpx.Client(transport=httpx.MockTransport(h)))
    _, rep = a.fetch_season_subs(bundle(1), tmp_path, season=1)
    assert list(rep.failed) == ["tmdb:1"]


def test_no_secrets_leak(tmp_path: Path):
    s = Server(quota=1, missing={3})
    a = adapter(s)
    part, rep = a.fetch_season_subs(bundle(), tmp_path, season=1)
    blob = (
        (tmp_path / "subs" / STATE_NAME).read_text()
        + part.model_dump_json()
        + json.dumps(rep.__dict__)
    )
    for secret in (KEY, USER, PW, TOKEN):
        assert secret not in blob


def _search_json(season=1, episode=1, **fd):
    d = load("search_s1e1.json")
    for r in d["data"]:
        r["attributes"]["feature_details"] = {
            "season_number": season, "episode_number": episode, **fd}
    return d


def _custom(handler):
    return OpenSubtitlesAdapter(
        KEY, USER, PW, client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda s: None,
    )


def test_search_redirect_followed():
    def h(req):
        if req.url.params.get("languages") == "EN":
            return httpx.Response(301, headers={"Location": str(req.url.copy_with(
                params={**req.url.params, "languages": "en"}))})
        return httpx.Response(200, json=_search_json())

    # an injected client that does not follow redirects by default still follows them
    c = httpx.Client(transport=httpx.MockTransport(h), follow_redirects=False)
    a = OpenSubtitlesAdapter(KEY, USER, PW, client=c, sleep=lambda s: None)
    r = a._send("GET", "https://api.opensubtitles.com/api/v1/subtitles?languages=EN", auth=False)
    assert r.status_code == 200 and r.request.url.params["languages"] == "en"
    assert a.search_episode(parent_tmdb_id=1, season=1, episode=1) is not None


def test_unfollowed_3xx_is_error_not_not_found():
    a = _custom(lambda req: httpx.Response(301))  # no Location: cannot follow
    with pytest.raises(OpenSubtitlesError):
        a.search_episode(parent_tmdb_id=1, season=1, episode=1)


def _dl_handler(link_resp):
    def h(req):
        if req.url.path.endswith("/login"):
            return httpx.Response(200, json=load("login.json"))
        if req.url.path.endswith("/download"):
            return httpx.Response(200, json={**load("download_ok.json"),
                                             "link": "https://dl.example/f.srt"})
        if req.url.path.endswith("/subtitles"):
            return httpx.Response(200, json=_search_json())
        return link_resp
    return h


@pytest.mark.parametrize("resp", [
    httpx.Response(302),
    httpx.Response(200, content=b""),
    httpx.Response(200, content=b"<html><body>Please log in</body></html>"),
    httpx.Response(200, content=b"1\n00:00:01,000 --> 00:00:02,000\nonly one\n"),
    httpx.Response(404, content=SRT),
])
def test_bad_download_body_refused_and_no_file(tmp_path: Path, resp):
    a = _custom(_dl_handler(resp))
    with pytest.raises(OpenSubtitlesError):
        a.download(5)
    _, rep = a.fetch_season_subs(bundle(1), tmp_path, season=1)
    assert list(rep.failed) == ["tmdb:1"] and not rep.downloaded
    assert list((tmp_path / "subs").glob("*.srt*")) == []


def test_download_redirect_followed(tmp_path: Path):
    def h(req):
        if req.url.host == "dl.example":
            return httpx.Response(302, headers={"Location": "https://cdn.example/f.srt"})
        if req.url.host == "cdn.example":
            return httpx.Response(200, content=SRT)
        return _dl_handler(None)(req)

    assert _custom(h).download(5) == SRT


def test_wrong_episode_results_dropped():
    def h(req):
        return httpx.Response(200, json=_search_json(season=1, episode=2))

    assert _custom(h).search_episode(parent_tmdb_id=1, season=1, episode=1) is None
    wrong_show = lambda req: httpx.Response(200, json=_search_json(parent_tmdb_id=5))  # noqa: E731
    assert _custom(wrong_show).search_episode(parent_tmdb_id=1, season=1, episode=1) is None
    no_details = lambda req: httpx.Response(200, json=load("search_s1e1.json"))  # noqa: E731
    assert _custom(no_details).search_episode(parent_tmdb_id=1, season=1, episode=1) is None
    ok = lambda req: httpx.Response(200, json=_search_json(parent_tmdb_id=1))  # noqa: E731
    assert _custom(ok).search_episode(parent_tmdb_id=1, season=1, episode=1) is not None


def test_406_without_quota_body_is_not_quota():
    def h(req):
        if req.url.path.endswith("/login"):
            return httpx.Response(200, json=load("login.json"))
        return httpx.Response(406, json={"message": "Not Acceptable"})

    a = _custom(h)
    with pytest.raises(OpenSubtitlesError) as ei:
        a.download(5)
    assert not isinstance(ei.value, QuotaExhausted)


def test_throttle_spaces_requests():
    t = [0.0]
    sleeps = []

    def sleep(d):
        sleeps.append(d)
        t[0] += d

    s = Server()
    a = adapter(s, sleep=sleep, clock=lambda: t[0], min_interval=0.25)
    for e in (1, 2, 3):
        a.search_episode(parent_tmdb_id=77, season=1, episode=e)
    assert sleeps == [0.25, 0.25]
    t[0] += 1.0  # idle time counts: no extra sleep
    a.search_episode(parent_tmdb_id=77, season=1, episode=1)
    assert sleeps == [0.25, 0.25]
