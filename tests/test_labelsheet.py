from pathlib import Path

from showvenist.eval.labelsheet import Row, build_rows, hint_for


def _row(name, minutes, key, show="Show"):
    return Row(Path("/lib") / show / "Season1" / name, show, "S01", "1", int(name[-6:-4]),
               minutes, 1.0, "", key)


def test_hint_thresholds():
    assert hint_for(None) == "no-reference"
    assert hint_for(1.1) == "episode-length"
    assert hint_for(0.5) == "short" and hint_for(2.0) == "long"


def test_episode_length_first_and_duplicates_flagged():
    rips = [_row("d_t00.mkv", 5, "a"), _row("d_t01.mkv", 44, "b"), _row("d_t02.mkv", 90, "c"),
            _row("d_t03.mkv", 44, "b")]
    rows = build_rows(rips, {"Show": 45.0}, Path("/lib"))
    assert [r["hint"] for r in rows] == ["episode-length", "long", "short", "episode-length"]
    assert [bool(r["same_as"]) for r in rows] == [False, False, False, True]  # duplicates last
    assert rows[-1]["title_no"] == 3


def test_shows_without_named_files_use_their_own_rip_median():
    minutes = [2, 22, 23, 22, 60]
    rips = [_row(f"d_t0{i}.mkv", m, f"k{i}", show="Cartoon") for i, m in enumerate(minutes)]
    rows = build_rows(rips, {}, Path("/lib"))
    assert {r["ref_min"] for r in rows} == {"22"}
