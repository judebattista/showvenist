from pathlib import Path

import pytest

from showvenist.analysis.subtitles import normalize, read_text_cues
from showvenist.analysis.types import SubtitleTrack
from tests.test_probe import make_mkv


@pytest.mark.parametrize(
    "raw,want",
    [
        ("[door slams] Hello there.", "hello there"),
        ("(laughing) What?!", "what"),
        ("JOHN: Don't go.", "dont go"),
        ("<i>It's</i> {\\an8}fine, ok?", "its fine ok"),
        # Mixed-case "Name:" is dialogue unless the line is otherwise all caps (CC style).
        ("Caldwell: Sir.\nSAM: Yes.", "caldwell sir yes"),
        ("Caldwell: SIR.\nSAM: YES.", "sir yes"),
        ("CENTAURI 1:\nJump point forming.", "jump point forming"),
        ("Look: we go now.", "look we go now"),
        ("- Listen: he's gone.", "listen hes gone"),
        ("'Tis the rock 'n' roll", "tis the rock n roll"),
        ("It’s   a\n\ttest", "its a test"),
        ("Time is 10:30.", "time is 10 30"),
        # A stray bracket isn't an annotation: only its punctuation goes (review fix).
        ("[unterminated", "unterminated"),
        ("I (think we should go now, Mal.", "i think we should go now mal"),
        ("[DOOR SLAMS]\n(laughing) Hi.", "hi"),
        # Annotations don't span lines or run on for a whole cue.
        ("(" + "x" * 80 + ") go", "x" * 80 + " go"),
    ],
)
def test_normalize(raw, want):
    assert normalize(raw) == want


def test_normalize_stray_bracket_keeps_next_line():
    assert normalize("[door slams\nWe go now").endswith("we go now")
    assert "we go now" in normalize("(sighs\nWe go now)")


def test_read_text_cues_ass(tmp_path: Path):
    path = tmp_path / "t.mkv"
    cues = [(i * 10.0, i * 10.0 + 3, f"{{\\i1}}Line {i}{{\\i0}}\\Nsecond") for i in range(10)]
    make_mkv(path, cues=cues, seconds=100)
    track = SubtitleTrack(1, "ssa", "eng", events=10)
    got = read_text_cues(path, track, 25.0, 52.0)
    assert [c.text for c in got] == [f"Line {i}\nsecond" for i in (3, 4, 5)]
    assert all(c.end_s - c.start_s == pytest.approx(3.0) for c in got)
    assert read_text_cues(path, track, 0, 1000)[0].text == "Line 0\nsecond"
    assert len(read_text_cues(path, track, 0, 1000)) == 10
    assert read_text_cues(path, track, 500, 600) == []
