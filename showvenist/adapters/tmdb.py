"""TMDB adapter (spec §6.4): show, episodes, aired ordering and every episode group."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from showvenist.bundle.schema import (
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

BASE_URL = "https://api.themoviedb.org/3"
MAX_RETRIES = 3
SEASON_BATCH = 20
WRITER_JOBS = {"Writer", "Screenplay", "Teleplay", "Story"}
GROUP_TYPES = {
    1: "original-air-date",
    2: "absolute",
    3: "dvd",
    4: "digital",
    5: "story-arc",
    6: "production",
    7: "tv",
}


class TmdbError(RuntimeError):
    """Request failure; messages never contain credentials."""


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "group"


class TmdbAdapter:
    name = "tmdb"

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key.strip()
        self._v3 = re.fullmatch(r"[0-9a-fA-F]{32}", key) is not None
        self._key = key
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        self._provenance: list[Provenance] = []

    # -- http -------------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None, *, record: bool = True) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        shown = httpx.URL(BASE_URL + path, params=params)  # never includes the key
        send_params = dict(params)
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._v3:
            send_params["api_key"] = self._key
        else:
            headers["Authorization"] = f"Bearer {self._key}"
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.get(BASE_URL + path, params=send_params, headers=headers)
            except httpx.HTTPError as exc:
                raise TmdbError(f"GET {path} failed: {type(exc).__name__}") from None
            if resp.status_code == 429 and attempt < MAX_RETRIES:
                try:
                    delay = float(resp.headers.get("Retry-After", "1"))
                except ValueError:
                    delay = 1.0
                self._sleep(max(0.0, delay))
                continue
            if resp.status_code >= 400:
                raise TmdbError(f"GET {path} returned HTTP {resp.status_code}")
            if record:
                self._provenance.append(
                    Provenance(
                        adapter="tmdb",
                        url=str(shown),
                        fetched_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )
                )
            return resp.json()
        raise TmdbError(f"GET {path} failed")  # pragma: no cover

    # -- lookups ----------------------------------------------------------
    def search(self, title: str, year: int | None = None) -> list[dict[str, Any]]:
        data = self._get("/search/tv", {"query": title, "first_air_date_year": year}, record=False)
        out = []
        for r in data.get("results", []):
            fad = r.get("first_air_date") or ""
            out.append(
                {
                    "tmdb_id": r["id"],
                    "title": r.get("name", ""),
                    "original_title": r.get("original_name", ""),
                    "year": int(fad[:4]) if fad[:4].isdigit() else None,
                    "overview": r.get("overview", ""),
                    "popularity": r.get("popularity", 0),
                }
            )
        return out

    def find_by_imdb(self, imdb_id: str) -> list[dict[str, Any]]:
        data = self._get("/find/" + imdb_id, {"external_source": "imdb_id"}, record=False)
        out = []
        for r in data.get("tv_results", []):
            fad = r.get("first_air_date") or ""
            out.append(
                {
                    "tmdb_id": r["id"],
                    "title": r.get("name", ""),
                    "year": int(fad[:4]) if fad[:4].isdigit() else None,
                    "overview": r.get("overview", ""),
                }
            )
        return out

    # -- fetch ------------------------------------------------------------
    def fetch(self, tmdb_id: int) -> PartialBundle:
        self._provenance = []
        show = self._get(f"/tv/{tmdb_id}", {"append_to_response": "external_ids"})
        numbers = sorted(
            {s["season_number"] for s in show.get("seasons", []) if "season_number" in s}
        )
        seasons: dict[int, dict[str, Any]] = {}
        for i in range(0, len(numbers), SEASON_BATCH):
            batch = numbers[i : i + SEASON_BATCH]
            data = self._get(
                f"/tv/{tmdb_id}",
                {"append_to_response": ",".join(f"season/{n}" for n in batch)},
            )
            for n in batch:
                if f"season/{n}" in data:
                    seasons[n] = data[f"season/{n}"]

        episodes: dict[str, Episode] = {}
        aired: list[OrderingEntry] = []
        for n in sorted(seasons):
            for ep in seasons[n].get("episodes", []):
                e = self._episode(ep)
                episodes.setdefault(e.id, e)
                aired.append(
                    OrderingEntry(
                        season=ep.get("season_number", n),
                        episode=ep["episode_number"],
                        episode_id=e.id,
                    )
                )
        orderings = {"aired": Ordering(source="tmdb", type="aired", entries=aired)}

        groups = self._get(f"/tv/{tmdb_id}/episode_groups").get("results", [])
        rank: dict[int, int] = {}
        for g in groups:
            gtype = g.get("type")
            rank[gtype] = rank.get(gtype, 0) + 1
            detail = self._get(f"/tv/episode_group/{g['id']}")
            entries: list[OrderingEntry] = []
            for sub in detail.get("groups", []):
                for ep in sub.get("episodes", []):
                    e = self._episode(ep)
                    episodes.setdefault(e.id, e)
                    entries.append(
                        OrderingEntry(
                            season=sub.get("order", 0),
                            episode=ep.get("order", 0) + 1,
                            episode_id=e.id,
                        )
                    )
            base = slugify(g.get("name", ""))
            key, n = base, 1
            while key in orderings:
                n += 1
                key = f"{base}-{n}"
            orderings[key] = Ordering(
                source="tmdb",
                type=GROUP_TYPES.get(gtype, f"type-{gtype}"),
                entries=entries,
                name=g.get("name"),
                tmdb_group=g["id"],
                type_rank=rank[gtype],
            )

        credits = self._get(f"/tv/{tmdb_id}/aggregate_credits")
        series_cast = [
            CastEntry(
                actor=c["name"],
                character=((c.get("roles") or [{}])[0].get("character") or ""),
            )
            for c in credits.get("cast", []) or []
            if c.get("name")
        ]

        fad = show.get("first_air_date") or ""
        ext = show.get("external_ids") or {}
        info = ShowInfo(
            title=show.get("name", ""),
            year=int(fad[:4]) if fad[:4].isdigit() else None,
            language=show.get("original_language") or "en",
            in_production=bool(show.get("in_production", False)),
            ids=ShowIds(tmdb=tmdb_id, imdb=ext.get("imdb_id") or None),
        )
        return PartialBundle(
            show=info,
            episodes=list(episodes.values()),
            orderings=orderings,
            provenance=list(self._provenance),
            series_cast=series_cast,
        )

    @staticmethod
    def _episode(ep: dict[str, Any]) -> Episode:
        writers: list[str] = []
        directors: list[str] = []
        for c in ep.get("crew", []) or []:
            name = c.get("name") or ""
            if not name:
                continue
            if c.get("job") in WRITER_JOBS and name not in writers:
                writers.append(name)
            elif c.get("job") == "Director" and name not in directors:
                directors.append(name)
        overview = (ep.get("overview") or "").strip()
        runtime = ep.get("runtime")
        return Episode(
            id=f"tmdb:{ep['id']}",
            title=ep.get("name") or "",
            air_date=ep.get("air_date") or None,
            runtime_min=int(runtime) if runtime else None,
            production_code=ep.get("production_code") or None,
            synopses=[Synopsis(source="tmdb", text=overview)] if overview else [],
            directors=directors,
            writers=writers,
            guest_cast=[
                CastEntry(actor=g["name"], character=g.get("character") or "")
                for g in ep.get("guest_stars", []) or []
                if g.get("name")
            ],
        )
