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
        ("Caldwell: Sir.\nSAM: Yes.", "sir yes"),
        ("'Tis the rock 'n' roll", "tis the rock n roll"),
        ("It’s   a\n\ttest", "its a test"),
        ("Time is 10:30.", "time is 10 30"),
        ("[unterminated", ""),
    ],
)
def test_normalize(raw, want):
    assert normalize(raw) == want


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
