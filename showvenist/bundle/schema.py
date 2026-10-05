"""Reference bundle schema (spec §6.3). The M1 contract: other modules import these models."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

# Language vocabulary shared with analysis (M2): ISO 639-1 codes in the bundle ("en"),
# mapped to Tesseract traineddata names by OCR_LANG_FOR. Subtitle track tags are ISO 639-2
# ("eng"); analysis converts via LANG_639_2.
LANG_639_2: dict[str, str] = {"en": "eng"}
OCR_LANG_FOR: dict[str, str] = {"en": "eng"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ShowIds(_Model):
    tmdb: int | None = None
    tvmaze: int | None = None
    imdb: str | None = None
    wikipedia: str | None = None


class ShowInfo(_Model):
    title: str
    year: int | None = None
    language: str = "en"
    in_production: bool = False
    ids: ShowIds = Field(default_factory=ShowIds)


class Synopsis(_Model):
    source: str
    text: str


class CastEntry(_Model):
    actor: str
    character: str = ""


class SubtitleRef(_Model):
    lang: str
    path: str  # relative to the bundle directory, e.g. "subs/tmdb-62091.en.srt"
    source: str  # "localsubs" | "opensubtitles"


class LlmCues(_Model):
    names: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    lines: list[str] = Field(default_factory=list)
    model: str
    generated_at: str


class Episode(_Model):
    id: str  # ordering-independent, e.g. "tmdb:62091"
    title: str
    air_date: str | None = None
    runtime_min: int | None = None
    production_code: str | None = None
    synopses: list[Synopsis] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    guest_cast: list[CastEntry] = Field(default_factory=list)
    subtitles: list[SubtitleRef] = Field(default_factory=list)
    alt_titles: list[str] = Field(default_factory=list)
    excluded: bool = False
    llm_cues: LlmCues | None = None


class OrderingEntry(_Model):
    season: int
    episode: int
    episode_id: str


class Ordering(_Model):
    source: str
    type: str  # "aired" | TMDB group type slug: "absolute", "dvd", "production", ...
    entries: list[OrderingEntry] = Field(default_factory=list)
    name: str | None = None
    tmdb_group: str | None = None
    type_rank: int | None = None  # position among groups of the same type in TMDB's list


class Provenance(_Model):
    adapter: str | None = None
    url: str | None = None
    fetched_at: str | None = None
    conflict: dict[str, str] | None = None


class Bundle(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    bundle_revision: str = ""
    show: ShowInfo
    episodes: list[Episode] = Field(default_factory=list)
    orderings: dict[str, Ordering] = Field(default_factory=dict)
    provenance: list[Provenance] = Field(default_factory=list)

    def episode_by_id(self) -> dict[str, Episode]:
        return {e.id: e for e in self.episodes}

    def compute_revision(self) -> str:
        """Content hash over everything except bundle_revision and provenance timestamps."""
        data = self.model_dump(mode="json", exclude={"bundle_revision", "provenance"})
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "b3:" + hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()

    def with_revision(self) -> Bundle:
        return self.model_copy(update={"bundle_revision": self.compute_revision()})


class PartialBundle(_Model):
    """What an adapter returns from fetch(show_ref); the store merges it into the Bundle."""

    show: ShowInfo | None = None
    episodes: list[Episode] = Field(default_factory=list)
    orderings: dict[str, Ordering] = Field(default_factory=dict)
    provenance: list[Provenance] = Field(default_factory=list)
