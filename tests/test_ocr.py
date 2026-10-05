"""OCR helpers on synthetic bitmaps and hand-built segment bytes (no real media)."""

from __future__ import annotations

import shutil
import struct

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from showvenist.analysis import ocr
from showvenist.analysis.ocr import Bitmap, PgsPalette, VobPalette, ocr_quality, render

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="no tesseract")


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _indexed_text(text: str, size: int, fill: int = 1) -> np.ndarray:
    """Palette-index bitmap like a disc subtitle: 0 background, `fill` for glyphs, no
    antialiasing (disc subtitles are a few palette entries, not greyscale)."""
    font = _font(size)
    w, h = (int(v) + size // 2 for v in font.getbbox(text)[2:])
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    draw.fontmode = "1"
    draw.text((size // 4, size // 4), text, font=font, fill=fill)
    return np.asarray(img, np.uint8).copy()


# --------------------------------------------------------------------------- PGS palette


def _seg(kind: int, body: bytes) -> bytes:
    return bytes([kind]) + struct.pack(">H", len(body)) + body


def _pds(pid: int, entries: list[tuple[int, int, int]]) -> bytes:
    body = bytes([pid, 0]) + b"".join(bytes([e, y, 128, 128, a]) for e, y, a in entries)
    return _seg(0x14, body)


def _pcs(pid: int) -> bytes:
    # width, height, framerate, comp number(2), comp state, palette update flag, palette id, ...
    body = struct.pack(">HHBHBBB", 1920, 1080, 0x10, 1, 0x80, 0, pid) + bytes([0])
    return _seg(0x16, body)


def test_pgs_palette_parses_pds_and_selects_by_pcs():
    pal = PgsPalette()
    pal.feed(_pcs(1) + _pds(0, [(1, 16, 255)]) + _pds(1, [(1, 235, 255), (2, 235, 128)]))
    lut = pal.lut()
    assert lut[1] == pytest.approx(255.0)
    assert lut[2] == pytest.approx(255 * 128 / 255)
    assert lut[0] == 0.0  # undefined entries are transparent
    pal.feed(_pcs(0))
    assert pal.lut()[1] == pytest.approx(0.0)  # palette 0: Y=16 is black


def test_pgs_palette_update_keeps_other_entries_and_skips_sup_framing():
    pal = PgsPalette()
    pal.feed(_pds(0, [(1, 235, 255), (3, 126, 255)]))
    pal.feed(b"PG" + bytes(8) + _pds(0, [(1, 16, 255)]))
    lut = pal.lut()
    assert lut[1] == pytest.approx(0.0)
    assert lut[3] == pytest.approx((126 - 16) * 255 / 219)


def test_pgs_palette_without_pds_is_identity():
    assert PgsPalette().lut()[200] == 200


# --------------------------------------------------------------------------- VobSub palette

IDX = (
    b"size: 720x480\npalette: 000000, ffffff, 808080, 101010, "
    + b", ".join([b"000000"] * 12)
    + b"\n"
)


def _vob_packet(colors: tuple[int, int, int, int], alphas: tuple[int, int, int, int]) -> bytes:
    """SPU: size(2), ctrl offset(2), 4 bytes of 'RLE', then one control block."""
    c = [*colors]
    a = [*alphas]
    ctrl = 8
    block = (
        struct.pack(">HH", 0, ctrl)  # delay, next = self (last block)
        + bytes([0x03, c[3] << 4 | c[2], c[1] << 4 | c[0]])
        + bytes([0x04, a[3] << 4 | a[2], a[1] << 4 | a[0]])
        + bytes([0x05])
        + bytes(6)
        + bytes([0x06])
        + bytes(4)
        + bytes([0x01, 0xFF])
    )
    body = bytes(4) + block
    return struct.pack(">HH", 4 + len(body), ctrl) + body


def test_vobsub_palette_from_idx_and_control_commands():
    pal = VobPalette(IDX)
    assert pal.luma is not None and pal.luma[1] == pytest.approx(255, abs=0.5)
    pal.feed(_vob_packet(colors=(0, 1, 2, 3), alphas=(0, 15, 15, 0)))
    lut = pal.lut()
    assert pal.colormap == [0, 1, 2, 3] and pal.alpha == [0, 15, 15, 0]
    assert lut[1] == pytest.approx(255, abs=0.5)
    assert lut[2] == pytest.approx(128, abs=1)
    assert lut[3] == 0.0  # transparent
    pal.feed(_vob_packet(colors=(0, 3, 1, 2), alphas=(0, 15, 15, 15)))
    assert pal.lut()[1] == pytest.approx(16, abs=0.5)  # code 1 now maps to the dark colour


def test_vobsub_palette_without_idx_assumes_code1_fill():
    lut = VobPalette(None).lut()
    assert list(lut[:4]) == [0, 255, 0, 128]


def test_vobsub_feed_tolerates_garbage():
    pal = VobPalette(IDX)
    pal.feed(b"\x00\x10\x00\x04\xff\xff")  # control offset pointing past the end
    pal.feed(b"\x00")
    assert pal.colormap == [0, 1, 2, 3]


# --------------------------------------------------------------------------- render


def test_render_composites_binarizes_and_inverts():
    lut = np.zeros(256, np.float32)
    lut[1] = 255
    a = np.zeros((10, 10), np.uint8)
    a[2:5, 2:5] = 1
    b = np.ones((5, 5), np.uint8)
    img = render([Bitmap(100, 50, a), Bitmap(120, 70, b)], lut)
    assert img is not None and img.mode == "L"
    arr = np.asarray(img)
    assert arr.shape == (25 + 40, 25 + 40)  # union of rects plus 20 px padding
    assert arr[20 + 3, 20 + 3] == 0  # ink is black
    assert arr[20 + 0, 20 + 0] == 255  # background is white
    assert arr[20 + 22, 20 + 22] == 0
    assert set(np.unique(arr)) <= {0, 255}


def test_render_blank_and_empty():
    lut = np.zeros(256, np.float32)
    assert render([], lut) is None
    assert render([Bitmap(0, 0, np.ones((4, 4), np.uint8))], lut) is None


def test_render_upscale():
    lut = np.zeros(256, np.float32)
    lut[1] = 255
    img = render([Bitmap(0, 0, np.ones((4, 6), np.uint8))], lut, upscale=3)
    assert img is not None and img.size == (6 * 3 + 40, 4 * 3 + 40)


class _FakeRect:
    def __init__(self, idx: np.ndarray, linesize: int):
        self.x, self.y = 3, 4
        self.height, self.width = idx.shape
        padded = np.zeros((self.height, linesize), np.uint8)
        padded[:, : self.width] = idx
        self.planes = [padded.tobytes()]


def test_bitmap_from_av_strips_linesize_padding():
    idx = np.arange(12, dtype=np.uint8).reshape(3, 4)
    bm = ocr.bitmap_from_av(_FakeRect(idx, linesize=8))
    assert bm is not None and (bm.x, bm.y) == (3, 4)
    assert np.array_equal(bm.indices, idx)


# --------------------------------------------------------------------------- OCR


@needs_tesseract
@pytest.mark.parametrize(
    "text,size,upscale,lut_entries",
    [
        ("Where were you last night?", 40, 1, {1: 235.0, 2: 20.0}),  # PGS-like, white on dark
        ("I told you, it's not safe.", 18, 3, {1: 255.0, 2: 0.0}),  # small VobSub-like, upscaled
    ],
)
def test_render_then_ocr_roundtrip(text, size, upscale, lut_entries):
    lut = np.zeros(256, np.float32)
    for k, v in lut_entries.items():
        lut[k] = v
    img = render([Bitmap(200, 900, _indexed_text(text, size))], lut, upscale)
    assert img is not None
    assert ocr.ocr_image(img, "en") == text


@needs_tesseract
def test_two_line_cue_and_pool():
    lut = np.zeros(256, np.float32)
    lut[1] = 255.0
    lines = ["We need to go.", "Now, Sam!"]
    top = Bitmap(100, 800, _indexed_text(lines[0], 40))
    bottom = Bitmap(100, 800 + top.height + 4, _indexed_text(lines[1], 40))
    img = render([top, bottom], lut)
    assert img is not None
    out = ocr.ocr_images([img, img], "en", workers=2)
    assert out == ["\n".join(lines)] * 2


def test_ocr_fixes_pipe_to_i(monkeypatch):
    monkeypatch.setattr(ocr.pytesseract, "image_to_string", lambda *a, **k: " | know.\n\n‘ok’ \n")
    assert ocr.ocr_image(Image.new("L", (1, 1)), "en") == "I know.\n'ok'"


# --------------------------------------------------------------------------- quality


def test_is_word_contractions_short_words_and_ocr_junk():
    for w in ("the", "theyre", "wouldnt", "youve", "maam", "howd", "okay", "gonna", "a", "i", "oh"):
        assert ocr.is_word(w), w
    for w in ("tbe", "rn", "ii", "lt", "th", "cl", "wliat", "qzxv", "w0rry"):
        assert not ocr.is_word(w), w
    assert ocr.is_word("1999")


def test_ocr_quality_uses_vocabulary_and_word_check():
    q = ocr_quality(["The car is red.", "Winchester qzxv"], {"winchester"})
    assert q == pytest.approx(5 / 6)
    assert ocr_quality([], set()) is None
    assert ocr_quality(["...", ""], set()) is None


def test_ocr_quality_junk_scores_low_and_real_dialogue_high():
    assert ocr_quality(["Don't leave tonight, they're coming."], set()) == 1.0
    assert ocr_quality(["tbe wliat rn ii lt qzxv"], set()) == 0.0
