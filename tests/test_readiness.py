import os
import shutil
from pathlib import Path

import pytest

from showvenist.hierarchy import readiness as rd
from showvenist.hierarchy.readiness import ReadinessChecker, matroska_complete
from tests.fixtures.textsubs import make_mkv_with_text_subs

NOW = 1_000_000.0


@pytest.fixture(scope="module")
def good_mkv(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("rd") / "good.mkv"
    make_mkv_with_text_subs(p, [(0, 2, "hi")], duration_s=10)
    return p


def _top_level(path: Path) -> dict[int, tuple[int, int]]:
    """id -> (offset, total length) of the Segment's top-level children."""
    out = {}
    with open(path, "rb") as f:
        h = rd._header(f)
        f.seek(h[2] + h[1])
        h = rd._header(f)
        pos = f.tell()
        end = pos + h[1]
        while pos < end:
            f.seek(pos)
            eid, size, hl = rd._header(f)
            out.setdefault(eid, (pos, hl + size))
            pos += hl + size
    return out


def _aged(p: Path) -> None:
    os.utime(p, (NOW - 3600, NOW - 3600))


def _checker(**kw) -> ReadinessChecker:
    return ReadinessChecker(60, clock=lambda: NOW, sleep=lambda s: None, **kw)


def test_complete(good_mkv):
    assert matroska_complete(good_mkv)
    assert rd.matroska_status(good_mkv) is None


def test_truncated_half(good_mkv, tmp_path):
    p = tmp_path / "half.mkv"
    p.write_bytes(good_mkv.read_bytes()[: good_mkv.stat().st_size // 2])
    assert not matroska_complete(p)
    assert rd.matroska_status(p) == "truncated"


def test_cues_removed(good_mkv, tmp_path):
    cues = _top_level(good_mkv)[rd._CUES][0]
    p = tmp_path / "nocues.mkv"
    p.write_bytes(good_mkv.read_bytes()[:cues])
    assert not matroska_complete(p)
    assert rd.matroska_status(p) in ("truncated", "no cues")


def test_unknown_size_segment_without_cues(good_mkv, tmp_path):
    data = bytearray(good_mkv.read_bytes())
    seg = data.index(b"\x18\x53\x80\x67")
    n = 9 - data[seg + 4].bit_length()
    data[seg + 4 : seg + 4 + n] = bytes([0xFF >> n | (1 << (8 - n))]) + b"\xff" * (n - 1)
    full = tmp_path / "unk.mkv"
    full.write_bytes(data)
    assert matroska_complete(full)
    cues = _top_level(good_mkv)[rd._CUES][0]
    cut = tmp_path / "unk_nocues.mkv"
    cut.write_bytes(bytes(data[:cues]))
    assert rd.matroska_status(cut) == "no cues"


def test_zero_duration(good_mkv, tmp_path):
    data = bytearray(good_mkv.read_bytes())
    info, length = _top_level(good_mkv)[rd._INFO]
    i = data.index(b"\x44\x89", info)
    assert i < info + length
    size = data[i + 2] & 0x7F
    data[i + 3 : i + 3 + size] = bytes(size)
    p = tmp_path / "nodur.mkv"
    p.write_bytes(data)
    assert rd.matroska_status(p) == "no duration"


def test_not_matroska(tmp_path):
    p = tmp_path / "x.mkv"
    p.write_bytes(b"hello world, definitely not ebml")
    assert rd.matroska_status(p) == "not matroska"
    assert rd.matroska_status(tmp_path / "missing.mkv") == "unreadable"


def test_ready(good_mkv, tmp_path):
    p = tmp_path / "e01.mkv"
    shutil.copy(good_mkv, p)
    _aged(p)
    assert _checker().check([p])[p].ready


def test_recent_mtime(good_mkv, tmp_path):
    p = tmp_path / "e01.mkv"
    shutil.copy(good_mkv, p)
    os.utime(p, (NOW - 12, NOW - 12))
    r = _checker().check([p])[p]
    assert not r.ready
    assert r.reason == "modified 12 s ago"


def test_unstable_size(good_mkv, tmp_path):
    p = tmp_path / "e01.mkv"
    shutil.copy(good_mkv, p)
    _aged(p)

    def grow(_s):
        with open(p, "ab") as f:
            f.write(b"x")
        _aged(p)

    r = ReadinessChecker(60, clock=lambda: NOW, sleep=grow).check([p])[p]
    assert not r.ready
    assert r.reason == "still growing"


def test_mtime_changed(good_mkv, tmp_path):
    p = tmp_path / "e01.mkv"
    shutil.copy(good_mkv, p)
    _aged(p)
    r = ReadinessChecker(
        60, clock=lambda: NOW, sleep=lambda s: os.utime(p, (NOW - 1000, NOW - 1000))
    ).check([p])[p]
    assert (r.ready, r.reason) == (False, "mtime changed")


def test_incomplete_mkv_not_ready(good_mkv, tmp_path):
    p = tmp_path / "e01.mkv"
    p.write_bytes(good_mkv.read_bytes()[:900])
    _aged(p)
    assert _checker().check([p])[p].reason == "truncated"


def test_other_container(tmp_path):
    import av

    p = tmp_path / "a.mp4"
    c = av.open(str(p), "w")
    s = c.add_stream("mpeg4", rate=1)
    s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
    import numpy as np

    for pkt in s.encode(av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), "uint8"), "rgb24")):
        c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    _aged(p)
    bad = tmp_path / "b.avi"
    bad.write_bytes(b"garbage" * 10)
    _aged(bad)
    res = _checker().check([p, bad])
    assert res[p].ready
    assert res[bad] == rd.Readiness(False, "unreadable")
