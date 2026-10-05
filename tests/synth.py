"""Shared synthetic data for scorer, fusion and eval tests (no real media, no network)."""

from __future__ import annotations

from pathlib import Path

from showvenist.analysis.types import Cue, Dialogue, SubtitleTrack, Window
from showvenist.bundle.schema import (
    Bundle,
    CastEntry,
    Episode,
    Ordering,
    OrderingEntry,
    ShowInfo,
    Synopsis,
)
from showvenist.config import Config
from showvenist.scoring.candidates import build_candidates
from showvenist.scoring.types import FileInfo, ScoreContext


def make_bundle(n: int = 6, season: int = 1, runtimes: list[int] | None = None,
                synopses: list[str] | None = None, guests: list[list[str]] | None = None) -> Bundle:
    eps, entries = [], []
    for k in range(n):
        eid = f"tmdb:{100 + k}"
        eps.append(Episode(
            id=eid, title=f"Episode Title {k + 1}",
            runtime_min=(runtimes or [44] * n)[k],
            synopses=[Synopsis(source="tmdb", text=(synopses or [""] * n)[k])] if synopses else [],
            guest_cast=[CastEntry(actor=f"Actor {g}", character=g)
                        for g in (guests[k] if guests else [])],
        ))
        entries.append(OrderingEntry(season=season, episode=k + 1, episode_id=eid))
    return Bundle(
        show=ShowInfo(title="Synth Show", year=2000),
        episodes=eps,
        orderings={"aired": Ordering(source="tmdb", type="aired", entries=entries)},
    ).with_revision()


def make_ctx(bundle: Bundle | None = None, **kw) -> ScoreContext:
    b = bundle or make_bundle()
    return ScoreContext(bundle=b, candidates=build_candidates(b), config=Config(), **kw)


def make_file(name: str = "t00.mkv", minutes: float = 44.0, lines: list[str] | None = None,
              title_tag: str | None = None, ocr_quality: float | None = 1.0) -> FileInfo:
    dialogue = None
    if lines is not None:
        cues = [Cue(float(i), float(i) + 1.5, t) for i, t in enumerate(lines)]
        dialogue = Dialogue(
            track=SubtitleTrack(index=3, codec="subrip", lang="eng", events=len(cues)),
            windows=[Window(0.0, float(len(cues)), cues)], ocr_quality=ocr_quality,
        )
    return FileInfo(path=Path("/rips") / name, fingerprint=f"fp-{name}", duration_s=minutes * 60,
                    title_tag=title_tag, dialogue=dialogue, ocr_quality=ocr_quality)
