"""OpenSubtitles adapter (spec §4.6, §6.4, §15.3; decision 38): resumable reference-subtitle ingest.

REST API v1 docs: https://opensubtitles.stoplight.io/docs/opensubtitles-api (endpoints
/login, /subtitles, /download on api.opensubtitles.com). Shapes used here:
- every request carries `Api-Key` and `User-Agent` headers;
- POST /login {username, password} -> {token, base_url, user:{allowed_downloads,...}}; later
  calls go to https://<base_url>/api/v1 with `Authorization: Bearer <token>`;
- GET /subtitles?parent_tmdb_id|parent_imdb_id&season_number&episode_number&languages&type=episode
  -> data[].attributes{language, download_count, hearing_impaired, machine_translated,
  ai_translated, files[{file_id, file_name}]};
- POST /download {file_id, sub_format} -> {link, remaining, requests, reset_time_utc}; when the
  daily quota is used up the API answers 406 with the same fields; 429 carries Retry-After.

No live account was available when this was written: it is tested only against
httpx.MockTransport. Credentials never appear in errors, provenance or the state file.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from showvenist.bundle.schema import (
    Bundle,
    Episode,
    PartialBundle,
    Provenance,
    SubtitleRef,
    to_639_1,
)

API_HOST = "api.opensubtitles.com"
MAX_RETRIES = 3
STATE_NAME = ".opensubtitles-state.json"
REF_SOURCES = {"localsubs", "opensubtitles"}
TEXT_EXTS = {".srt", ".ass", ".ssa", ".vtt"}  # sub/idx are image formats


class OpenSubtitlesError(RuntimeError):
    """Request failure; messages never contain credentials."""


class QuotaExhausted(OpenSubtitlesError):
    def __init__(self, reset_time: str | None = None) -> None:
        super().__init__(f"OpenSubtitles daily download quota exhausted (resets {reset_time})")
        self.reset_time = reset_time


@dataclass
class FetchReport:
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # episode id -> reason
    quota_stopped: list[str] = field(default_factory=list)  # episodes not attempted/finished
    reset_time: str | None = None


def _ext(name: str) -> str:
    return Path(name).suffix.lower() if name else ""


def rank_candidates(
    results: list[dict[str, Any]], language: str
) -> list[tuple[int, dict[str, Any]]]:
    """Eligible (file_id, attributes) pairs, best first.

    Rules: language matches; not machine- or AI-translated; text format; prefer
    non-hearing-impaired (SDH accepted when nothing else); then highest download_count.
    """
    out: list[tuple[tuple[int, int, int], int, dict[str, Any]]] = []
    for r in results:
        a = r.get("attributes") or {}
        if to_639_1(str(a.get("language") or "")) != language and a.get("language") != language:
            continue
        if a.get("machine_translated") or a.get("ai_translated"):
            continue
        for f in a.get("files") or []:
            fid = f.get("file_id")
            name = f.get("file_name") or ""
            if fid is None or (name and _ext(name) not in TEXT_EXTS):
                continue
            key = (
                1 if a.get("hearing_impaired") else 0,
                0 if _ext(name) == ".srt" or not name else 1,
                -int(a.get("download_count") or 0),
            )
            out.append((key, int(fid), a))
    out.sort(key=lambda t: t[0])
    return [(fid, a) for _, fid, a in out]


class OpenSubtitlesAdapter:
    name = "opensubtitles"

    def __init__(
        self,
        api_key: str,
        username: str,
        password: str,
        client: httpx.Client | None = None,
        user_agent: str = "showvenist v0.1",
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._key = api_key.strip()
        self._user = username
        self._password = password
        self._ua = user_agent
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        self._now = now
        self._token: str | None = None
        self._base = f"https://{API_HOST}/api/v1"
        self.remaining: int | None = None
        self._provenance: list[Provenance] = []

    # -- http -------------------------------------------------------------
    def _headers(self, auth: bool) -> dict[str, str]:
        h = {
            "Api-Key": self._key,
            "User-Agent": self._ua,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if auth and self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _send(self, method: str, url: str, *, auth: bool = True, retry_login: bool = True,
              **kw: Any) -> httpx.Response:
        what = f"{method} {httpx.URL(url).path}"
        for attempt in range(MAX_RETRIES + 1):
            if auth and self._token is None:
                self.login()
            try:
                resp = self._client.request(method, url, headers=self._headers(auth), **kw)
            except httpx.HTTPError as exc:
                raise OpenSubtitlesError(f"{what} failed: {type(exc).__name__}") from None
            if resp.status_code == 401 and auth and retry_login:
                self._token = None
                retry_login = False
                continue
            if resp.status_code == 429:
                if _json(resp).get("remaining") == 0:
                    raise QuotaExhausted(_json(resp).get("reset_time_utc"))
                if attempt < MAX_RETRIES:
                    try:
                        delay = float(resp.headers.get("Retry-After", "1"))
                    except ValueError:
                        delay = 1.0
                    self._sleep(max(0.0, delay))
                    continue
            return resp
        raise OpenSubtitlesError(f"{what} failed")  # pragma: no cover

    def login(self) -> None:
        resp = self._send(
            "POST", f"{self._base}/login", auth=False,
            json={"username": self._user, "password": self._password},
        )
        if resp.status_code >= 400:
            raise OpenSubtitlesError(f"login failed: HTTP {resp.status_code}")
        data = _json(resp)
        token = data.get("token")
        if not token:
            raise OpenSubtitlesError("login failed: no token in response")
        self._token = token
        base = (data.get("base_url") or API_HOST).removeprefix("https://").strip("/")
        self._base = f"https://{base}/api/v1"

    # -- search / download --------------------------------------------------
    def search_episode(
        self,
        parent_tmdb_id: int | None = None,
        imdb: str | None = None,
        season: int = 1,
        episode: int = 1,
        language: str = "en",
    ) -> tuple[int, dict[str, Any]] | None:
        """Best (file_id, attributes) for the episode, or None."""
        if parent_tmdb_id is None and not imdb:
            raise ValueError("parent_tmdb_id or imdb is required")
        params: dict[str, Any] = {
            "episode_number": episode,
            "languages": language,
            "season_number": season,
            "type": "episode",
        }
        if parent_tmdb_id is not None:
            params["parent_tmdb_id"] = parent_tmdb_id
        else:
            params["parent_imdb_id"] = re.sub(r"\D", "", str(imdb))
        if self._token is None:
            self.login()  # the login response decides the base URL
        url = f"{self._base}/subtitles"
        resp = self._send("GET", url, params=dict(sorted(params.items())))
        if resp.status_code >= 400:
            raise OpenSubtitlesError(f"GET /subtitles returned HTTP {resp.status_code}")
        self._provenance.append(
            Provenance(
                adapter="opensubtitles",
                url=str(httpx.URL(url, params=dict(sorted(params.items())))),
                fetched_at=self._now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
        )
        ranked = rank_candidates(_json(resp).get("data") or [], language)
        return ranked[0] if ranked else None

    def download(self, file_id: int) -> bytes:
        """Download one subtitle file (counts against the daily quota)."""
        if self._token is None:
            self.login()
        resp = self._send(
            "POST", f"{self._base}/download", json={"file_id": file_id, "sub_format": "srt"}
        )
        body = _json(resp)
        if "remaining" in body:
            self.remaining = body.get("remaining")
        if resp.status_code == 406:
            raise QuotaExhausted(body.get("reset_time_utc") or body.get("reset_time"))
        if resp.status_code >= 400 or not body.get("link"):
            raise OpenSubtitlesError(f"POST /download returned HTTP {resp.status_code}")
        try:
            # The link is pre-signed: no API key or token is sent to the file host.
            got = self._client.get(body["link"], headers={"User-Agent": self._ua})
        except httpx.HTTPError as exc:
            raise OpenSubtitlesError(f"file fetch failed: {type(exc).__name__}") from None
        if got.status_code >= 400:
            raise OpenSubtitlesError(f"file fetch returned HTTP {got.status_code}")
        return got.content

    # -- season ingest ------------------------------------------------------
    def fetch_season_subs(
        self,
        bundle: Bundle,
        bundle_dir: Path,
        ordering: str = "aired",
        language: str | None = None,
        season: int | None = None,
    ) -> tuple[PartialBundle, FetchReport]:
        lang = language or bundle.show.language
        subs = Path(bundle_dir) / "subs"
        subs.mkdir(parents=True, exist_ok=True)
        state_path = subs / STATE_NAME
        state = _load_state(state_path)
        report = FetchReport()
        by_id = bundle.episode_by_id()
        ids = bundle.show.ids
        entries = [
            e for e in bundle.orderings[ordering].entries
            if season is None or e.season == season
        ]
        # Already-reported quota that has not reset yet: do nothing over the network.
        blocked_until = _parse_time(state.get("quota_reset"))
        blocked = blocked_until is not None and self._now() < blocked_until
        new: list[Episode] = []
        stopped = False
        for ent in entries:
            eid = ent.episode_id
            rel = f"subs/{eid.replace(':', '-')}.{lang}.srt"
            ep = by_id.get(eid)
            if ep is not None and any(s.source in REF_SOURCES for s in ep.subtitles):
                report.skipped.append(eid)
                continue
            if (Path(bundle_dir) / rel).is_file():
                report.skipped.append(eid)
                new.append(_partial_ep(eid, lang, rel))
                state.setdefault("done", {})[eid] = rel
                continue
            if stopped or blocked:
                report.quota_stopped.append(eid)
                report.reset_time = state.get("quota_reset")
                continue
            try:
                hit = self.search_episode(
                    ids.tmdb, ids.imdb, ent.season, ent.episode, lang
                ) if (ids.tmdb is not None or ids.imdb) else None
                if hit is None:
                    report.not_found.append(eid)
                    state.setdefault("not_found", {})[eid] = "no eligible subtitle"
                else:
                    data = self.download(hit[0])
                    (Path(bundle_dir) / rel).write_bytes(data)
                    report.downloaded.append(eid)
                    new.append(_partial_ep(eid, lang, rel))
                    state.setdefault("done", {})[eid] = rel
                    state.get("not_found", {}).pop(eid, None)
                    state.get("failed", {}).pop(eid, None)
            except QuotaExhausted as q:
                stopped = True
                report.quota_stopped.append(eid)
                report.reset_time = q.reset_time
                state["quota_reset"] = q.reset_time
            except OpenSubtitlesError as exc:
                report.failed[eid] = str(exc)
                state.setdefault("failed", {})[eid] = str(exc)
            state["remaining"] = self.remaining
            _save_state(state_path, state)
        if not stopped and not blocked:
            state.pop("quota_reset", None)
        _save_state(state_path, state)
        partial = PartialBundle(episodes=new, provenance=list(self._provenance))
        self._provenance = []
        return partial, report


def _partial_ep(eid: str, lang: str, rel: str) -> Episode:
    return Episode(
        id=eid, title="", subtitles=[SubtitleRef(lang=lang, path=rel, source="opensubtitles")]
    )


def _json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_time(s: Any) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    tmp.replace(path)
