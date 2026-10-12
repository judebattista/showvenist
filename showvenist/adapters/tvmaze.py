"""TVmaze adapter (spec §6.4, §6.5): secondary source for titles, summaries and runtimes."""

from __future__ import annotations

import html
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from showvenist.bundle.schema import (
    Episode,
    Ordering,
    OrderingEntry,
    PartialBundle,
    Provenance,
    ShowIds,
    ShowInfo,
    Synopsis,
)

BASE_URL = "https://api.tvmaze.com"
MAX_RETRIES = 3
_TAG = re.compile(r"<[^>]*>")
_BLOCK = re.compile(r"</?(?:p|br|div|li|ul|ol)\b[^>]*>", re.IGNORECASE)
_REF_URL = re.compile(r"^https?://(?:www\.)?tvmaze\.com/shows/(\d+)(?:/|$)", re.IGNORECASE)
_REF_ID = re.compile(r"^tvmaze:(\d+)$", re.IGNORECASE)


class TvmazeError(RuntimeError):
    """Request failure; messages never contain secrets."""


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    text = _BLOCK.sub(" ", text)
    text = _TAG.sub("", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def tvmaze_id_from_ref(ref: str) -> int | None:
    ref = ref.strip()
    m = _REF_URL.match(ref) or _REF_ID.match(ref)
    return int(m.group(1)) if m else None


def _year(date: str | None) -> int | None:
    d = date or ""
    return int(d[:4]) if d[:4].isdigit() else None


class TvmazeAdapter:
    name = "tvmaze"

    def __init__(
        self,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        self._provenance: list[Provenance] = []

    def _get(self, path: str, params: dict[str, Any] | None = None, *, record: bool = True) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        shown = httpx.URL(BASE_URL + path, params=params)
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.get(
                    BASE_URL + path, params=params, headers={"Accept": "application/json"}
                )
            except httpx.HTTPError as exc:
                raise TvmazeError(f"GET {path} failed: {type(exc).__name__}") from None
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt < MAX_RETRIES:
                try:
                    delay = float(resp.headers.get("Retry-After", ""))
                except ValueError:
                    delay = 2.0**attempt
                self._sleep(max(0.0, delay))
                continue
            if resp.status_code >= 400:
                raise TvmazeError(f"GET {path} returned HTTP {resp.status_code}")
            if record:
                self._provenance.append(
                    Provenance(
                        adapter="tvmaze",
                        url=str(shown),
                        fetched_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )
                )
            return resp.json()
        raise TvmazeError(f"GET {path} failed")  # pragma: no cover

    def search(self, title: str, year: int | None = None) -> list[dict[str, Any]]:
        data = self._get("/search/shows", {"q": title}, record=False)
        out = []
        for r in data or []:
            s = r.get("show") or {}
            if "id" not in s:
                continue
            y = _year(s.get("premiered"))
            # premiere years can differ by one between sources (Babylon 5: 1993 vs 1994)
            if year is not None and y is not None and abs(y - year) > 1:
                continue
            out.append(
                {
                    "tvmaze_id": s["id"],
                    "title": s.get("name", ""),
                    "year": y,
                    "status": s.get("status") or "",
                }
            )
        return out

    def fetch(self, tvmaze_id: int) -> PartialBundle:
        self._provenance = []
        show = self._get(f"/shows/{tvmaze_id}")
        raw = self._get(f"/shows/{tvmaze_id}/episodes", {"specials": 1})

        episodes: list[Episode] = []
        entries: list[OrderingEntry] = []
        specials = 0
        for ep in raw or []:
            e = self._episode(ep)
            episodes.append(e)
            if ep.get("number") is None:
                specials += 1
                season, number = 0, specials
            else:
                season, number = ep["season"], ep["number"]
            entries.append(OrderingEntry(season=season, episode=number, episode_id=e.id))

        info = ShowInfo(
            title=show.get("name", ""),
            year=_year(show.get("premiered")),
            in_production=show.get("status") != "Ended",
            ids=ShowIds(tvmaze=tvmaze_id),
        )
        return PartialBundle(
            show=info,
            episodes=episodes,
            orderings={"tvmaze-aired": Ordering(source="tvmaze", type="aired", entries=entries)},
            provenance=list(self._provenance),
        )

    @staticmethod
    def _episode(ep: dict[str, Any]) -> Episode:
        text = strip_html(ep.get("summary"))
        runtime = ep.get("runtime") or ep.get("averageRuntime")
        return Episode(
            id=f"tvmaze:{ep['id']}",
            title=ep.get("name") or "",
            air_date=ep.get("airdate") or None,
            runtime_min=int(runtime) if runtime else None,
            synopses=[Synopsis(source="tvmaze", text=text)] if text else [],
        )
