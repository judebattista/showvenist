"""Reference bundle schema (spec §6.3). The M1 contract: other modules import these models."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1

# Language vocabulary shared with analysis (M2): the bundle stores ISO 639-1 ("en").
# Subtitle track tags are ISO 639-2, which has bibliographic (B) and terminologic (T) variants.
# value = (639-2/B, 639-2/T, Tesseract traineddata name)
LANGUAGES: dict[str, tuple[str, str, str]] = {
    "en": ("eng", "eng", "eng"), "fr": ("fre", "fra", "fra"), "de": ("ger", "deu", "deu"),
    "es": ("spa", "spa", "spa"), "it": ("ita", "ita", "ita"), "pt": ("por", "por", "por"),
    "nl": ("dut", "nld", "nld"), "sv": ("swe", "swe", "swe"), "da": ("dan", "dan", "dan"),
    "no": ("nor", "nor", "nor"), "fi": ("fin", "fin", "fin"), "pl": ("pol", "pol", "pol"),
    "ru": ("rus", "rus", "rus"), "ja": ("jpn", "jpn", "jpn"), "ko": ("kor", "kor", "kor"),
    "zh": ("chi", "zho", "chi_sim"), "cs": ("cze", "ces", "ces"), "hu": ("hun", "hun", "hun"),
    "tr": ("tur", "tur", "tur"), "el": ("gre", "ell", "ell"), "he": ("heb", "heb", "heb"),
}


def lang_codes(lang: str) -> set[str]:
    """Every tag a subtitle track may carry for a bundle language (639-1, 639-2/B and /T)."""
    b, t, _ = LANGUAGES.get(lang, (lang, lang, lang))
    return {lang, b, t}


def ocr_language(lang: str) -> str:
    """Tesseract traineddata name for a bundle language."""
    try:
        return LANGUAGES[lang][2]
    except KeyError:
        raise ValueError(f"unsupported language {lang!r}") from None


def to_639_1(code: str) -> str | None:
    code = code.lower()
    for l1, (b, t, _) in LANGUAGES.items():
        if code in (l1, b, t):
            return l1
    return None


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
    # Series cast only: how many episodes the person is credited in (TMDB total_episode_count);
    # None = unknown (bundles written before this field), treated as main cast.
    episodes: int | None = None


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


class TitleCardWindow(_Model):
    """One span of video searched for the card (§6.6): `length_seconds` starting `skip_seconds`
    from the chosen edge of the file (inward)."""

    # `from` is a Python keyword; the YAML key is `from`
    edge: Literal["start", "end"] = Field(alias="from")
    skip_seconds: float = Field(0.0, ge=0)
    length_seconds: float = Field(gt=0, le=600)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TitleCardConfig(_Model):
    """`title_card` in overrides.yaml (§6.6, decision 54). Absent = no card is ever read."""

    windows: list[TitleCardWindow] = Field(min_length=1)
    pattern: str  # regex with a named group `ref`; matched with re.search, case-insensitive
    kind: Literal["roman", "number", "title"]
    background: Literal["dark", "any"] = "dark"

    @field_validator("pattern")
    @classmethod
    def _pattern_ok(cls, v: str) -> str:
        try:
            rx = re.compile(v, re.I)
        except re.error as e:
            raise ValueError(f"pattern is not a valid regex: {e}") from e
        if "ref" not in rx.groupindex:
            raise ValueError("pattern needs a named group (?P<ref>...)")
        return v


class Bundle(_Model):
    schema_version: Literal[1] = SCHEMA_VERSION
    bundle_revision: str = ""
    show: ShowInfo
    episodes: list[Episode] = Field(default_factory=list)
    orderings: dict[str, Ordering] = Field(default_factory=dict)
    provenance: list[Provenance] = Field(default_factory=list)
    # Series-level (main) cast; optional so bundles written before M3 still load.
    series_cast: list[CastEntry] = Field(default_factory=list)
    # From overrides.yaml only (never written to show.json); None = the show has no title card.
    title_card: TitleCardConfig | None = None

    def episode_by_id(self) -> dict[str, Episode]:
        return {e.id: e for e in self.episodes}

    def compute_revision(self, bundle_dir: Path | None = None) -> str:
        """Content hash over everything except bundle_revision and provenance.

        With `bundle_dir`, the content of each referenced subtitle file is hashed too, so a
        corrected reference .srt at the same path gives a new revision.
        """
        data = self.model_dump(mode="json", exclude={"bundle_revision", "provenance"})
        if self.title_card is None:  # bundles without a card keep their earlier revision hash
            data.pop("title_card", None)
        if bundle_dir is not None:
            files = {
                s.path: hashlib.blake2b(
                    (bundle_dir / s.path).read_bytes(), digest_size=16
                ).hexdigest()
                for e in self.episodes
                for s in e.subtitles
                if (bundle_dir / s.path).is_file()
            }
            if files:  # bundles without reference subtitles keep their plain content hash
                data["subtitle_files"] = files
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return "b3:" + hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()

    def with_revision(self, bundle_dir: Path | None = None) -> Bundle:
        return self.model_copy(update={"bundle_revision": self.compute_revision(bundle_dir)})


class PartialBundle(_Model):
    """What an adapter returns from fetch(show_ref); the store merges it into the Bundle."""

    show: ShowInfo | None = None
    episodes: list[Episode] = Field(default_factory=list)
    orderings: dict[str, Ordering] = Field(default_factory=dict)
    provenance: list[Provenance] = Field(default_factory=list)
    series_cast: list[CastEntry] = Field(default_factory=list)
