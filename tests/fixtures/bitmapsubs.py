"""Synthetic bitmap-subtitle MKVs (PGS and VobSub) with known ground truth (spec §16.4).

`make_mkv_with_bitmap_subs` renders each cue's text with Pillow, encodes it as a
real bitmap subtitle stream and muxes it with a tiny black video via PyAV. No
external binaries are needed; ffmpeg's own decoders (via PyAV) read the result.

Format details relied on
========================

Timing: subtitle packets use a 1/1000 time base (Matroska's native one);
pts = dts = cue start, packet duration = cue length.

PGS (codec ``hdmv_pgs_subtitle``, Matroska ``S_HDMV/PGS``)
    In MKV each packet is one *display set* of segments with NO ``"PG"`` + pts/dts
    header (that framing exists only in raw ``.sup`` files). A segment is
    ``type:u8  size:u16be  body``.
    * PCS 0x16: width:u16 height:u16 frame_rate:u8(0x10) composition_number:u16
      composition_state:u8 (0x80 = epoch start, 0x00 = normal) palette_update:u8(0)
      palette_id:u8 object_count:u8, then per object
      object_id:u16 window_id:u8 crop_flag:u8(0) x:u16 y:u16.
    * WDS 0x17: window_count:u8, then id:u8 x:u16 y:u16 w:u16 h:u16.
    * PDS 0x14: palette_id:u8 version:u8, then entries id:u8 Y:u8 Cr:u8 Cb:u8 A:u8.
      We use entry 0 = transparent (Y16, A0) and entry 1 = opaque white (Y235).
    * ODS 0x15: object_id:u16 version:u8 seq_flag:u8 (0xC0 = first and last)
      data_len:u24 (RLE length + 4) width:u16 height:u16 RLE.
      One ODS per object; we refuse bitmaps whose RLE won't fit one segment.
    * END 0x80: empty.
    RLE per line: a non-zero byte C is one pixel of colour C; ``00`` escapes:
    ``00 00`` end of line, ``00 0LLLLLL`` L(1-63) pixels of 0,
    ``00 01LLLLLL LLLLLLLL`` L(<16384) pixels of 0, ``00 10LLLLLL C`` L(3-63)
    pixels of C, ``00 11LLLLLL LLLLLLLL C`` L pixels of C.
    Each cue = a display set at its start (epoch start, one object) plus a
    clearing display set at its end (PCS with zero objects, WDS, END). ffmpeg's
    decoder yields a subtitle with one rect for the first and none for the second.

VobSub (codec ``dvd_subtitle``, Matroska ``S_VOBSUB``)
    Stream extradata is the ``.idx`` header text ``palette: rrggbb, ...`` (16
    entries; this is what ffmpeg's decoder and spike/extract.py parse). We get
    it by opening the stream with ffmpeg's dvdsub *encoder*
    (``options={"palette": ...}``), whose init writes exactly that text; PyAV's
    add_mux_stream can't set extradata. The encoder only adds the optional
    ``size: WxH`` line when the context has dimensions, which PyAV's subtitle
    context doesn't expose, so it is absent. The packets themselves are ours.
    Each packet is one SPU: ``size:u16  ctrl_offset:u16``, then the RLE for the
    top field (even rows) followed by the bottom field (odd rows), then the
    control sequences. Each control sequence is ``date:u16 next:u16 commands``,
    date in units of 1024/90000 s:
      0x01 start display; 0x03 SET_COLOR (2 bytes, palette index per colour
      code, layout below); 0x04 SET_CONTR (alpha 0..15, same layout);
      0x05 SET_DAREA x1:12 x2:12 y1:12 y2:12 (inclusive); 0x06 SET_DSPXA
      top_field_offset:u16 bottom_field_offset:u16 (from SPU start); 0xFF end.
      A second sequence at date=duration holds 0x02 stop display and points
      ``next`` at itself.
    SET_COLOR/SET_CONTR byte layout (as ffmpeg reads it):
      byte0 = code3<<4 | code2, byte1 = code1<<4 | code0.
    We use code 0 = palette 0 alpha 0 (background), code 1 = palette 1 (white)
    alpha 15; codes 2 and 3 are unused and transparent.
    RLE (nibble-oriented, 2-bit colour code c, run n): n<4 -> 1 nibble ``nnc``;
    n<16 -> 2 nibbles; n<64 -> 3; n<256 -> 4 nibbles (value always n<<2|c);
    every row ends byte-aligned.
    ffmpeg's dvdsub decoder crops each rect to its non-transparent bounding box,
    so bitmaps are cropped to the ink here too: decoded rect size == bitmap size
    for both formats.
"""

from __future__ import annotations

import struct
from fractions import Fraction
from pathlib import Path
from typing import Literal

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

Cue = tuple[float, float, str]

FONT_SIZE = 40
VIDEO_FPS = 1
MS = Fraction(1, 1000)
# 16-entry .idx palette: 0 black, 1 white, rest grey (unused).
VOBSUB_PALETTE = ["000000", "ffffff"] + ["808080"] * 14


# --------------------------------------------------------------------------- rendering

def render_text(text: str, max_size: tuple[int, int]) -> np.ndarray:
    """Render text as a binary bitmap (uint8, 1 = ink, 0 = clear), cropped to the ink."""
    if not text.strip():
        raise ValueError("cue text must not be empty")
    font = ImageFont.load_default(size=FONT_SIZE)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    bbox = probe.multiline_textbbox((0, 0), text, font=font, align="center", spacing=8)
    x0, y0, x1, y1 = (int(v) for v in bbox)
    img = Image.new("L", (x1 - x0 + 10, y1 - y0 + 10), 0)
    ImageDraw.Draw(img).multiline_text((4 - x0, 4 - y0), text, fill=255, font=font,
                                       align="center", spacing=8)
    bits = (np.asarray(img) >= 128).astype(np.uint8)
    rows, cols = np.flatnonzero(bits.any(1)), np.flatnonzero(bits.any(0))
    bits = bits[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = bits.shape
    if w > max_size[0] or h > max_size[1]:
        raise ValueError(f"rendered cue {text!r} ({w}x{h}) exceeds canvas {max_size}")
    return bits


def _placement(bits: np.ndarray, size: tuple[int, int]) -> tuple[int, int]:
    """Bottom-centre position, like real subtitles."""
    h, w = bits.shape
    return (size[0] - w) // 2, max(0, size[1] - h - size[1] // 12)


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    """(length, value) runs of a row."""
    edges = np.flatnonzero(np.diff(row)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(row)]))
    return [(int(e - s), int(row[s])) for s, e in zip(starts, ends, strict=True)]


# --------------------------------------------------------------------------- PGS

def _pgs_rle(bits: np.ndarray) -> bytes:
    out = bytearray()
    for row in bits:
        for n, c in _runs(row):
            while n:
                k = min(n, 16383)
                n -= k
                if c == 0:
                    out += bytes([0, k]) if k < 64 else bytes([0, 0x40 | k >> 8, k & 0xFF])
                elif k <= 2:
                    out += bytes([c]) * k
                elif k < 64:
                    out += bytes([0, 0x80 | k, c])
                else:
                    out += bytes([0, 0xC0 | k >> 8, k & 0xFF, c])
        out += b"\x00\x00"
    return bytes(out)


def _seg(kind: int, body: bytes) -> bytes:
    return struct.pack(">BH", kind, len(body)) + body


def _pgs_pcs(size, comp_no, state, obj: tuple[int, int] | None) -> bytes:
    body = struct.pack(">HHBHBBBB", size[0], size[1], 0x10, comp_no, state, 0, 0,
                       1 if obj else 0)
    if obj:
        body += struct.pack(">HBBHH", 0, 0, 0, obj[0], obj[1])
    return _seg(0x16, body)


def _pgs_display_sets(bits: np.ndarray, size, comp_no: int) -> tuple[bytes, bytes]:
    h, w = bits.shape
    x, y = _placement(bits, size)
    wds = _seg(0x17, struct.pack(">BBHHHH", 1, 0, x, y, w, h))
    pds = _seg(0x14, bytes([0, 0]) + bytes([0, 16, 128, 128, 0]) + bytes([1, 235, 128, 128, 255]))
    rle = _pgs_rle(bits)
    ods_body = struct.pack(">HBB", 0, 0, 0xC0) + (len(rle) + 4).to_bytes(3, "big")
    ods_body += struct.pack(">HH", w, h) + rle
    if len(ods_body) > 0xFFFF:
        raise ValueError("PGS object too large for a single ODS segment")
    end = _seg(0x80, b"")
    show = _pgs_pcs(size, comp_no, 0x80, (x, y)) + wds + pds + _seg(0x15, ods_body) + end
    clear = _pgs_pcs(size, comp_no + 1, 0x00, None) + wds + end
    return show, clear


# --------------------------------------------------------------------------- VobSub

class _Nibbles:
    def __init__(self):
        self.nibs: list[int] = []

    def put(self, value: int, count: int) -> None:
        for i in reversed(range(count)):
            self.nibs.append(value >> (4 * i) & 0xF)

    def align(self) -> None:
        if len(self.nibs) % 2:
            self.nibs.append(0)

    def to_bytes(self) -> bytes:
        self.align()
        return bytes(self.nibs[i] << 4 | self.nibs[i + 1] for i in range(0, len(self.nibs), 2))


def _vob_field(rows: np.ndarray) -> bytes:
    nb = _Nibbles()
    for row in rows:
        for n, c in _runs(row):
            while n:
                k = min(n, 255)
                n -= k
                v = k << 2 | c
                nb.put(v, 1 if k < 4 else 2 if k < 16 else 3 if k < 64 else 4)
        nb.align()
    return nb.to_bytes()


def _vob_spu(bits: np.ndarray, size, duration_s: float) -> bytes:
    h, w = bits.shape
    x, y = _placement(bits, size)
    top, bottom = _vob_field(bits[0::2]), _vob_field(bits[1::2])
    top_off = 4
    bot_off = top_off + len(top)
    ctrl1 = 4 + len(top) + len(bottom)
    x2, y2 = x + w - 1, y + h - 1
    cmds1 = (b"\x01"
             + b"\x03" + bytes([0x00, 0x10])  # code1 -> palette 1, code0 -> palette 0
             + b"\x04" + bytes([0x00, 0xF0])  # code1 alpha 15, others 0
             + b"\x05" + bytes([x >> 4, (x & 0xF) << 4 | x2 >> 8, x2 & 0xFF,
                               y >> 4, (y & 0xF) << 4 | y2 >> 8, y2 & 0xFF])
             + b"\x06" + struct.pack(">HH", top_off, bot_off)
             + b"\xff")
    ctrl2 = ctrl1 + 4 + len(cmds1)
    stop_date = max(1, round(duration_s * 90000 / 1024))
    seq1 = struct.pack(">HH", 0, ctrl2) + cmds1
    seq2 = struct.pack(">HH", stop_date, ctrl2) + b"\x02\xff"
    body = top + bottom + seq1 + seq2
    total = 4 + len(body)
    if total > 0xFFFF:
        raise ValueError("VobSub SPU too large")
    return struct.pack(">HH", total, ctrl1) + body


# --------------------------------------------------------------------------- muxing

def make_mkv_with_bitmap_subs(
    path: str | Path,
    cues: list[Cue],
    kind: Literal["pgs", "vobsub"],
    lang: str = "eng",
    duration_s: float | None = None,
    size: tuple[int, int] = (720, 480),
) -> Path:
    """Write an MKV with black 1 fps video and one bitmap subtitle track of `cues`.

    cues are (start_s, end_s, text); text may contain newlines. The file lasts
    `duration_s` (default: one second past the last cue). Returns the path.
    """
    path = Path(path)
    cues = sorted(cues)
    for start, end, _ in cues:
        if not 0 <= start < end:
            raise ValueError(f"bad cue timing {start}..{end}")
    total = duration_s if duration_s is not None else (cues[-1][1] + 1 if cues else 1.0)
    w, h = size

    events: list[tuple[int, bytes, int]] = []  # (pts_ms, data, duration_ms)
    for i, (start, end, text) in enumerate(cues):
        bits = render_text(text, size)
        s_ms, e_ms = round(start * 1000), round(end * 1000)
        if kind == "pgs":
            show, clear = _pgs_display_sets(bits, size, 2 * i)
            events.append((s_ms, show, e_ms - s_ms))
            events.append((e_ms, clear, 0))
        elif kind == "vobsub":
            events.append((s_ms, _vob_spu(bits, size, end - start), e_ms - s_ms))
        else:
            raise ValueError(f"unknown kind {kind!r}")
    events.sort(key=lambda e: e[0])

    with av.open(str(path), "w", format="matroska") as out:
        vs = out.add_stream("mpeg4", rate=VIDEO_FPS)
        vs.width, vs.height, vs.pix_fmt = w - w % 2, h - h % 2, "yuv420p"
        vs.codec_context.gop_size = 600
        if kind == "pgs":
            ss = out.add_mux_stream("hdmv_pgs_subtitle")
        else:
            ss = out.add_stream("dvd_subtitle", options={"palette": ",".join(VOBSUB_PALETTE)})
        ss.time_base = MS
        ss.metadata["language"] = lang

        black = av.VideoFrame.from_ndarray(np.zeros((vs.height, vs.width, 3), np.uint8),
                                           format="rgb24")
        nframes = max(1, int(np.ceil(total * VIDEO_FPS)))
        ei = 0
        for f in range(nframes):
            t_ms = f * 1000 // VIDEO_FPS
            while ei < len(events) and events[ei][0] <= t_ms:
                _mux_sub(out, ss, *events[ei])
                ei += 1
            black.pts = f
            black.time_base = Fraction(1, VIDEO_FPS)
            for pkt in vs.encode(black):
                out.mux(pkt)
        for pkt in vs.encode(None):
            out.mux(pkt)
        for ev in events[ei:]:
            _mux_sub(out, ss, *ev)
    return path


def _mux_sub(out, stream, pts_ms: int, data: bytes, dur_ms: int) -> None:
    pkt = av.Packet(data)
    pkt.stream = stream
    pkt.time_base = MS
    pkt.pts = pkt.dts = pts_ms
    pkt.duration = dur_ms
    out.mux(pkt)
