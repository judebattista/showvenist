from pathlib import Path

from showvenist.bundle.quality import quality_report, render
from showvenist.bundle.schema import (
    Bundle,
    CastEntry,
    Episode,
    Ordering,
    OrderingEntry,
    Provenance,
    ShowInfo,
    SubtitleRef,
    Synopsis,
)


def _bundle() -> Bundle:
    long = "a detective finds a body in the harbor and chases a suspect"
    eps = [
        Episode(
            id="e1",
            title="1",
            runtime_min=45,
            directors=["D"],
            synopses=[Synopsis(source="tmdb", text=long)],
            guest_cast=[
                CastEntry(actor="A", character="Bob"),
                CastEntry(actor="B", character="Cop #2"),
            ],
            subtitles=[SubtitleRef(lang="en", path="subs/e1.en.srt", source="localsubs")],
        ),
        Episode(id="e2", title="2", runtime_min=45, synopses=[Synopsis(source="tmdb", text=long)]),
        Episode(id="e3", title="3", runtime_min=45),
        Episode(id="e4", title="4", excluded=True),
    ]
    ents = [
        OrderingEntry(season=s, episode=n, episode_id=i)
        for s, n, i in [(1, 1, "e1"), (1, 2, "e2"), (2, 1, "e3"), (2, 2, "e4")]
    ]
    return Bundle(
        show=ShowInfo(title="X"),
        episodes=eps,
        orderings={"aired": Ordering(source="tmdb", type="aired", entries=ents)},
        provenance=[Provenance(adapter="tmdb", conflict={"episode": "e3"})],
    )


def test_report(tmp_path: Path):
    r = quality_report(_bundle(), tmp_path)
    s1, s2 = r.seasons
    assert s1.synopsis_coverage == 1.0 and s1.mean_synopsis_words == 12
    assert s1.guest_cast_coverage == 0.5 and s1.crew_coverage == 0.5
    assert s1.subtitle_coverage == 0.0  # file missing on disk
    assert s2.episodes == 1 and s2.synopsis_coverage == 0
    assert r.excluded == ["e4"] and len(r.conflicts) == 1
    assert r.duplicate_synopses == [("e1", "e2")]
    (tmp_path / "subs").mkdir()
    (tmp_path / "subs" / "e1.en.srt").write_text("x")
    assert quality_report(_bundle(), tmp_path).seasons[0].subtitle_coverage == 0.5
    assert quality_report(_bundle()).seasons[0].subtitle_coverage == 0.5


def test_render():
    text = render(quality_report(_bundle()))
    assert "Season 2: synopsis coverage 0%, no reference subtitles. Expect many files" in text
    assert "Consider `--subs` or enabling the LLM scorer." in text
