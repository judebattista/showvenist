from pathlib import Path

from showvenist.adapters.localsubs import install, parse_season_episode, scan_folder
from showvenist.bundle.schema import Bundle, Episode, Ordering, OrderingEntry, ShowInfo


def _bundle() -> Bundle:
    eps = [Episode(id=f"tmdb:{i}", title=f"t{i}") for i in (1, 2, 3)]
    ents = [OrderingEntry(season=1, episode=i, episode_id=f"tmdb:{i}") for i in (1, 2, 3)]
    return Bundle(
        show=ShowInfo(title="X"),
        episodes=eps,
        orderings={"aired": Ordering(source="tmdb", type="aired", entries=ents)},
    )


def test_parse():
    assert parse_season_episode("a.S01E02.en.srt") == (1, 2)
    assert parse_season_episode("a 1x02.srt") == (1, 2)
    assert parse_season_episode("a.srt") is None


def test_scan_and_install(tmp_path: Path):
    f = tmp_path / "in"
    f.mkdir()
    (f / "Show.s01e01.fr.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
    (f / "Show.S01E02.srt").write_text("x")
    (f / "Show 1x02.srt").write_text("dup")
    (f / "random.srt").write_text("x")
    (f / "Show.S09E09.srt").write_text("x")
    partial, rep = scan_folder(f, _bundle())
    assert {e.id for e in partial.episodes} == {"tmdb:1", "tmdb:2"}
    langs = {e.id: e.subtitles[0].lang for e in partial.episodes}
    assert langs == {"tmdb:1": "fr", "tmdb:2": "en"}
    assert len(rep.unmatched) == 2
    assert list(rep.duplicates) == ["tmdb:2"]
    assert rep.missing == ["tmdb:3"]
    out = install(partial, f, tmp_path / "b")
    paths = {e.id: e.subtitles[0].path for e in out.episodes}
    assert paths["tmdb:1"] == "subs/tmdb-1.fr.srt"
    assert (tmp_path / "b" / "subs" / "tmdb-1.fr.srt").read_text().endswith("hi\n")


def test_lang_arg(tmp_path: Path):
    (tmp_path / "S01E03.srt").write_text("x")
    partial, _ = scan_folder(tmp_path, _bundle(), lang="de")
    assert partial.episodes[0].subtitles[0].lang == "de"
