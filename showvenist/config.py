"""Configuration (spec §4.6): config.toml < SHOWVENIST_* env < command-line flags."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigError(ValueError):
    """A configuration value that can't be used; the message names where it came from."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TmdbConfig(_Section):
    api_key: str | None = None


class OpenSubtitlesConfig(_Section):
    api_key: str | None = None
    username: str | None = None
    password_env: str = "SHOWVENIST_OPENSUBTITLES_PASSWORD"


class NamingConfig(_Section):
    template: str = "{show_snake}_s{s:02}e{e:02}"
    move_extras: bool = True


class HistoryConfig(_Section):
    """[history]: the soft prior against an episode already confirmed for another file (§11.4)."""

    penalty: float = 2.302585092994046  # ln 10


class AnalysisConfig(_Section):
    io_concurrency: int = 1
    ocr_workers: int = 0
    window_min_lines: int = 60
    window_max_minutes: int = 5
    readiness_stable_seconds: int = 60
    duplicate_overlap: float = 0.5  # near-duplicate rips: dialogue containment to group (§14)
    duplicate_min_shingles: int = 100  # files with less sampled dialogue are never grouped


class LlmConfig(_Section):
    mode: str = "off"  # off | auto | all
    model: str = "claude-sonnet-5-5"
    effort: str = "low"
    send_both_orders: bool = True
    scope: str = "disc"  # disc | season


class LlmWeights(_Section):
    """[weights.llm]: files the LLM has scored (§11.1)."""

    duration: float = 1.0
    title: float = 1.0
    titlecard: float = 1.0
    refsubs: float = 1.0
    llm: float = 1.0


class WeightsConfig(_Section):
    """[weights]: base regime; `llm` holds the LLM-regime weights. Written by `eval --fit`."""

    duration: float = 1.0
    title: float = 1.0
    titlecard: float = 1.0
    refsubs: float = 1.0
    synopsis: float = 1.0
    guests: float = 1.0
    cues: float = 1.0
    llm: LlmWeights = Field(default_factory=LlmWeights)


class HierarchyConfig(_Section):
    """[hierarchy]: name patterns (regexes, case-insensitive); empty = the built-in defaults."""

    season_patterns: list[str] = Field(default_factory=list)
    disc_patterns: list[str] = Field(default_factory=list)
    named_patterns: list[str] = Field(default_factory=list)


class BlocksConfig(_Section):
    """[blocks]: placeholders until tuned by eval (spec §18)."""

    unclaimed_penalty: float = 2.0
    gap_penalty: float = 2.0
    ordering_threshold: float = 15.0
    ordering_margin: float = 4.0  # another ordering must beat the starting one by this much
    block_slack: int = 1  # positions a disc block may span beyond its file count (skipped titles)
    title_order_weight: float = 0.3  # title-order tie-breaker, nats (decision 63); 0 = off


INGEST_SOURCES = ("tmdb", "tvmaze")


class IngestConfig(_Section):
    """[ingest]: which sources `ingest` fetches (TMDB is primary and required; §4.2, §6.5)."""

    sources: list[str] = Field(default_factory=lambda: ["tmdb", "tvmaze"])
    stale_days: int = 30  # identify's refresh notice (§6.8)

    @field_validator("sources")
    @classmethod
    def _sources_ok(cls, v: list[str]) -> list[str]:
        v = [s.strip().lower() for s in v if s.strip()]
        if "wikipedia" in v:
            raise ValueError("the wikipedia source is not built yet (decision 62)")
        bad = [s for s in v if s not in INGEST_SOURCES]
        if bad:
            raise ValueError(
                f"unknown ingest source {', '.join(map(repr, bad))} "
                f"(known: {', '.join(INGEST_SOURCES)})")
        if "tmdb" not in v:
            raise ValueError("ingest sources must include 'tmdb' (the primary source)")
        return list(dict.fromkeys(v))


class Config(_Section):
    tmdb: TmdbConfig = Field(default_factory=TmdbConfig)
    opensubtitles: OpenSubtitlesConfig = Field(default_factory=OpenSubtitlesConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    weights: WeightsConfig = Field(default_factory=WeightsConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    blocks: BlocksConfig = Field(default_factory=BlocksConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)


def _xdg(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / default)


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / "showvenist"


def data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / "showvenist"


def cache_dir() -> Path:
    return _xdg("XDG_CACHE_HOME", ".cache") / "showvenist"


def bundles_dir() -> Path:
    return data_dir() / "shows"


def load_config(path: Path | None = None, env: dict[str, str] | None = None) -> Config:
    """Load config.toml (if present), then apply SHOWVENIST_* overrides.

    Env mapping: SHOWVENIST_TMDB_API_KEY, SHOWVENIST_OPENSUBTITLES_API_KEY,
    SHOWVENIST_LLM_MODE, ... i.e. SHOWVENIST_<SECTION>_<FIELD> for scalar fields.
    """
    env = dict(os.environ) if env is None else env
    path = path or config_dir() / "config.toml"
    raw: dict = tomllib.loads(path.read_text()) if path.exists() else {}
    cfg = Config.model_validate(raw)
    for name, section in cfg:
        if not isinstance(section, _Section):
            continue
        for field in type(section).model_fields:
            val = env.get(f"SHOWVENIST_{name}_{field}".upper())
            if val is None:
                continue
            current = getattr(section, field)
            if isinstance(current, list):
                val = [v.strip() for v in val.split(",") if v.strip()]
            elif isinstance(current, bool):
                val = val.lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int | float):
                kind = type(current)
                try:
                    val = kind(val)
                except ValueError:
                    raise ConfigError(
                        f"SHOWVENIST_{name}_{field}".upper() + f"={val!r}: expected "
                        f"{'an integer' if kind is int else 'a number'}") from None
            setattr(section, field, val)
    return Config.model_validate(cfg.model_dump())
