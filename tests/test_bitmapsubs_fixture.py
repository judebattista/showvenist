"""The bitmap-subtitle fixture generator produces streams ffmpeg decodes to the right bitmaps."""

from __future__ import annotations

import re
import shutil

import av
import numpy as np
import pytest
from PIL import Image

from tests.fixtures.bitmapsubs import make_mkv_with_bitmap_subs, render_text

CUES = [
    (1.0, 2.5, "Hello there friend"),
    (3.0, 4.5, "The quick brown fox\njumps over the dog"),
    (5.25, 7.0, "Where is the car"),
]
CODECS = {"pgs": "hdmv_pgs_subtitle", "vobsub": "dvd_subtitle"}
DECODERS = {"pgs": "pgssub", "vobsub": "dvdsub"}


def _decode(path):
    """[(pts_s, [rect, ...])] for subtitle packets that decode to bitmaps."""
    out = []
    with av.open(str(path)) as c:
        s = c.streams.subtitles[0]
        meta = {
            "lang": s.metadata.get("language"),
            "codec": s.codec_context.name,
            "extradata": s.codec_context.extradata,
            "duration": c.duration / av.time_base,
        }
        for pkt in c.demux(s):
            if pkt.pts is None or not pkt.size:
                continue
            rects = [r for r in pkt.decode() if r.width and r.height]
            if rects:
                out.append((float(pkt.pts * pkt.time_base), rects))
    return out, meta


def _ink(rect) -> np.ndarray:
    buf = np.frombuffer(bytes(rect.planes[0]), np.uint8)
    stride = len(buf) // rect.height
    return buf[: stride * rect.height].reshape(rect.height, stride)[:, : rect.width] > 0


def _ocr(ink: np.ndarray) -> str:
    pytesseract = pytest.importorskip("pytesseract")
    img = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8), "L")
    img = Image.fromarray(np.pad(np.asarray(img), 20, constant_values=255), "L")
    return pytesseract.image_to_string(img, lang="eng", config="--psm 6")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


@pytest.mark.parametrize("kind", ["pgs", "vobsub"])
def test_decodes_to_expected_bitmaps(tmp_path, kind):
    path = make_mkv_with_bitmap_subs(tmp_path / f"{kind}.mkv", CUES, kind, lang="eng")
    events, meta = _decode(path)

    assert meta["codec"] == DECODERS[kind]
    assert meta["lang"] == "eng"
    assert len(events) == len(CUES)
    for (pts, rects), (start, _end, text) in zip(events, CUES, strict=True):
        assert pts == pytest.approx(start, abs=0.001)
        assert len(rects) == 1
        r = rects[0]
        expected = render_text(text, (720, 480))
        assert (r.height, r.width) == expected.shape
        assert 0 <= r.x and r.x + r.width <= 720 and r.y + r.height <= 480
        # Pixel-exact round trip through the RLE encoder and ffmpeg's decoder.
        assert np.array_equal(_ink(r), expected.astype(bool))


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
@pytest.mark.parametrize("kind", ["pgs", "vobsub"])
def test_ocr_reads_cue_text(tmp_path, kind):
    path = make_mkv_with_bitmap_subs(tmp_path / f"{kind}.mkv", CUES, kind)
    events, _ = _decode(path)
    for (_pts, rects), (_s, _e, text) in zip(events, CUES, strict=True):
        assert _norm(_ocr(_ink(rects[0]))) == _norm(text)


def test_vobsub_extradata_has_idx_palette(tmp_path):
    path = make_mkv_with_bitmap_subs(tmp_path / "v.mkv", CUES[:1], "vobsub")
    _, meta = _decode(path)
    m = re.search(rb"palette:\s*([0-9a-fA-F, ]+)", meta["extradata"])
    assert m is not None
    cols = [x.strip() for x in m.group(1).decode().split(",")]
    assert len(cols) == 16 and cols[:2] == ["000000", "ffffff"]


def test_pgs_packets_are_bare_segments(tmp_path):
    """MKV PGS blocks start with a PCS segment, not the .sup "PG" header; each cue
    is a show display set plus a clearing one."""
    path = make_mkv_with_bitmap_subs(tmp_path / "p.mkv", CUES, "pgs")
    with av.open(str(path)) as c:
        datas = [bytes(p) for p in c.demux(c.streams.subtitles[0]) if p.size]
    assert len(datas) == 2 * len(CUES)
    assert all(d[0] == 0x16 and d[-3:] == b"\x80\x00\x00" for d in datas)


def test_long_sparse_duration(tmp_path):
    path = make_mkv_with_bitmap_subs(
        tmp_path / "long.mkv", [(600.0, 602.0, "Late line")], "vobsub", duration_s=1200
    )
    events, meta = _decode(path)
    assert meta["duration"] == pytest.approx(1200, abs=2)
    assert [round(t) for t, _ in events] == [600]
    assert path.stat().st_size < 2_000_000
