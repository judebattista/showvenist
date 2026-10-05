"""OCR for image-based subtitles: PGS (Blu-ray) and VobSub (DVD) (spec §8.3).

Ported from the M0 spike (spike/extract.py), where it was validated on real rips.

Pipeline: demux the subtitle stream with PyAV from ``start_s``; feed every packet's raw bytes to
a palette tracker (PyAV decodes bitmaps to raw palette indices but doesn't expose the palette,
so it is parsed here); decode the packet to bitmap rects; ``render`` composites them into one
dark-on-white image (palette -> luminance, upscale for low-resolution VobSub, binarize, invert,
pad); Tesseract reads the images in a thread pool.

Threads, not processes: pytesseract runs each image in its own ``tesseract`` subprocess, so a
thread pool already uses every core without pickling images across processes.

Dictionary for ``ocr_quality``: the spike used ``wordfreq``. Here the word list comes from
Tesseract's own traineddata for the OCR language (its LSTM word DAWG, unpacked with
``combine_tessdata`` and ``dawg2wordlist``, both shipped in the ``tesseract-ocr`` package that
OCR already requires): no extra Python dependency and no system word list. English yields about
340k words. It is unpacked once per process into a temporary directory and held in memory. If
the tools or the traineddata are missing, quality falls back to the show vocabulary alone.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import av
import numpy as np
import pytesseract
from PIL import Image

from showvenist.analysis.types import Cue, SubtitleTrack
from showvenist.bundle.schema import OCR_LANG_FOR

TESS_CONFIG = "--psm 6"
VOBSUB_UPSCALE = 3  # DVD subtitles are ~720x480; Tesseract wants taller glyphs
MIN_PEAK = 40  # a canvas whose brightest pixel is below this is treated as blank
FALLBACK_CUE_S = 5.0  # cue length when neither the packet nor a following packet bounds it


# --------------------------------------------------------------------------- palettes


class PgsPalette:
    """Tracks PGS palettes from PDS segments (0x14); the PCS (0x16) selects the active one.

    Packets carry raw segments: ``type(1) size(2, BE) body``. Raw ``.sup`` framing
    (``PG`` + timestamps) is skipped too, though MKV doesn't use it."""

    def __init__(self) -> None:
        self.pals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.current = 0

    def feed(self, data: bytes) -> None:
        i = 0
        n = len(data)
        while i + 3 <= n:
            if data[i : i + 2] == b"PG":
                i += 10
                continue
            seg, size = data[i], int.from_bytes(data[i + 1 : i + 3], "big")
            body = data[i + 3 : i + 3 + size]
            if seg == 0x14 and len(body) >= 2:  # PDS: id, version, then 5-byte entries
                pid = body[0]
                y, a = self.pals.get(pid, (np.zeros(256, np.float32), np.zeros(256, np.float32)))
                y, a = y.copy(), a.copy()
                for j in range(2, len(body) - 4, 5):
                    eid, yy, _cr, _cb, aa = body[j : j + 5]
                    y[eid] = yy
                    a[eid] = aa
                self.pals[pid] = (y, a)
            elif seg == 0x16 and len(body) >= 10:  # PCS: palette id at offset 9
                self.current = body[9]
            i += 3 + size

    def lut(self) -> np.ndarray:
        """Palette index -> luminance composited over black (0..255)."""
        pal = self.pals.get(self.current) or next(iter(self.pals.values()), None)
        if pal is None:
            return np.arange(256, dtype=np.float32)
        y, a = pal
        yf = np.clip((y - 16) * 255 / 219, 0, 255)  # studio-range luma -> full range
        return yf * a / 255


class VobPalette:
    """VobSub: global 16-colour palette from the idx extradata, per-packet colour/alpha
    assignment from the SET_COLOR (0x03) and SET_CONTR (0x04) control commands."""

    def __init__(self, extradata: bytes | None):
        self.luma: np.ndarray | None = None
        if extradata:
            m = re.search(rb"palette:\s*([0-9a-fA-F, ]+)", extradata)
            if m:
                cols = [int(x, 16) for x in m.group(1).decode().replace(" ", "").split(",") if x]
                cols = cols[:16]
                lum = [
                    0.299 * (c >> 16 & 255) + 0.587 * (c >> 8 & 255) + 0.114 * (c & 255)
                    for c in cols
                ]
                self.luma = np.array(lum + [0.0] * (16 - len(cols)), np.float32)
        self.colormap = [0, 1, 2, 3]
        self.alpha = [0, 15, 15, 15]

    def feed(self, data: bytes) -> None:
        if len(data) < 4:
            return
        ctrl = int.from_bytes(data[2:4], "big")
        seen: set[int] = set()
        while ctrl + 4 <= len(data) and ctrl not in seen:
            seen.add(ctrl)
            nxt = int.from_bytes(data[ctrl + 2 : ctrl + 4], "big")
            p = ctrl + 4
            while p < len(data):
                cmd = data[p]
                p += 1
                if cmd in (0x00, 0x01, 0x02):  # forced start, start, stop display
                    continue
                if cmd == 0x03 and p + 2 <= len(data):
                    b0, b1 = data[p], data[p + 1]
                    self.colormap = [b1 & 15, b1 >> 4, b0 & 15, b0 >> 4]
                    p += 2
                elif cmd == 0x04 and p + 2 <= len(data):
                    b0, b1 = data[p], data[p + 1]
                    self.alpha = [b1 & 15, b1 >> 4, b0 & 15, b0 >> 4]
                    p += 2
                elif cmd == 0x05:  # SET_DAREA
                    p += 6
                elif cmd == 0x06:  # SET_DSPXA
                    p += 4
                else:  # 0xff end, or unknown/extended
                    break
            if nxt == ctrl:
                break
            ctrl = nxt

    def lut(self) -> np.ndarray:
        out = np.zeros(256, np.float32)
        if self.luma is None:  # no palette: assume code 1 is the fill (most common DVD layout)
            out[:4] = [0, 255, 0, 128]
            return out
        for code in range(4):
            out[code] = self.luma[self.colormap[code]] * self.alpha[code] / 15
        return out


# --------------------------------------------------------------------------- image


class Rect(Protocol):
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Bitmap:
    """A positioned rect of palette indices (``indices`` is height x width, uint8)."""

    x: int
    y: int
    indices: np.ndarray

    @property
    def width(self) -> int:
        return int(self.indices.shape[1])

    @property
    def height(self) -> int:
        return int(self.indices.shape[0])


def bitmap_from_av(rect: Any) -> Bitmap | None:
    """Convert a PyAV ``BitmapSubtitle`` (plane rows may be padded to a linesize)."""
    if rect.width <= 0 or rect.height <= 0 or not rect.planes:
        return None
    buf = np.frombuffer(bytes(rect.planes[0]), np.uint8)
    ls = len(buf) // rect.height
    if ls < rect.width:
        return None
    idx = buf[: ls * rect.height].reshape(rect.height, ls)[:, : rect.width]
    return Bitmap(rect.x, rect.y, idx)


def render(bitmaps: Sequence[Bitmap], lut: np.ndarray, upscale: int = 1) -> Image.Image | None:
    """Composite bitmaps into one canvas, palette -> grayscale (bright text over black),
    upscale, binarize, invert to dark text on white with padding. None if blank."""
    bitmaps = [b for b in bitmaps if b.width > 0 and b.height > 0]
    if not bitmaps:
        return None
    x0 = min(b.x for b in bitmaps)
    y0 = min(b.y for b in bitmaps)
    x1 = max(b.x + b.width for b in bitmaps)
    y1 = max(b.y + b.height for b in bitmaps)
    canvas = np.zeros((y1 - y0, x1 - x0), np.float32)
    for b in bitmaps:
        g = lut[b.indices]
        sub = canvas[b.y - y0 : b.y - y0 + b.height, b.x - x0 : b.x - x0 + b.width]
        np.maximum(sub, g, out=sub)
    if canvas.max() < MIN_PEAK:
        return None
    img = Image.fromarray(canvas.clip(0, 255).astype(np.uint8), "L")
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.Resampling.LANCZOS)
    g = np.asarray(img, np.float32)
    ink = g >= max(0.5 * g.max(), 90.0)
    out = np.pad(np.where(ink, 0, 255).astype(np.uint8), 20, constant_values=255)
    return Image.fromarray(out, "L")


# --------------------------------------------------------------------------- OCR

_OCR_FIXES = [
    (re.compile(r"(?<![A-Za-z])[|](?![A-Za-z])"), "I"),
    (re.compile(r"(?<=\s)[|]|^[|]", re.M), "I"),
    (re.compile(r"[‘’`´]"), "'"),
    (re.compile(r"[“”]"), '"'),
]


def ocr_image(img: Image.Image, language: str = "en") -> str:
    """Tesseract one rendered subtitle image; returns trimmed non-empty lines."""
    text = pytesseract.image_to_string(img, lang=OCR_LANG_FOR[language], config=TESS_CONFIG)
    for rx, rep in _OCR_FIXES:
        text = rx.sub(rep, text)
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


def _pool_size(workers: int) -> int:
    return workers if workers > 0 else (os.cpu_count() or 1)


def ocr_images(images: Sequence[Image.Image], language: str, workers: int = 0) -> list[str]:
    if not images:
        return []
    with ThreadPoolExecutor(min(_pool_size(workers), len(images))) as pool:
        return list(pool.map(lambda im: ocr_image(im, language), images))


@dataclass
class _Event:
    start_s: float
    end_s: float | None  # None until bounded by packet duration or the next packet
    image: Image.Image


def _decode_events(path: Path, track: SubtitleTrack, start_s: float, end_s: float) -> list[_Event]:
    kind = track.kind
    if kind not in ("pgs", "vobsub"):
        raise ValueError(f"not a bitmap subtitle track: {track.codec}")
    upscale = VOBSUB_UPSCALE if kind == "vobsub" else 1
    events: list[_Event] = []
    with av.open(str(path)) as c:
        stream = c.streams[track.index]
        pal: PgsPalette | VobPalette = (
            PgsPalette() if kind == "pgs" else VobPalette(stream.codec_context.extradata)
        )
        if start_s > 0:
            c.seek(int(start_s * av.time_base), backward=True, any_frame=False)
            try:
                stream.codec_context.flush_buffers()
            except Exception:
                pass
        for pkt in c.demux(stream):
            if pkt.pts is None:
                continue
            t = float(pkt.pts * pkt.time_base)
            if t > end_s:
                break
            # Any packet (an empty PGS clear included) ends the cue on screen before it.
            if events and (events[-1].end_s is None or events[-1].end_s > t):
                events[-1].end_s = max(t, events[-1].start_s)
            data = bytes(pkt)
            pal.feed(data)  # palette state must see packets before the window too
            try:
                rects = pkt.decode()
            except av.error.FFmpegError:
                continue
            if t < start_s:
                continue
            bitmaps = [b for b in map(bitmap_from_av, rects) if b is not None]
            img = render(bitmaps, pal.lut(), upscale)
            if img is None:
                continue
            dur = float(pkt.duration * pkt.time_base) if pkt.duration else 0.0
            events.append(_Event(t, t + dur if dur > 0 else None, img))
    return events


def ocr_cues(
    path: Path,
    track: SubtitleTrack,
    start_s: float,
    end_s: float,
    language: str,
    workers: int = 0,
) -> list[Cue]:
    """OCR the bitmap subtitle cues starting in [start_s, end_s]. Text is unnormalized;
    consecutive identical cues (PGS re-sends an epoch) are merged; empty OCR results dropped."""
    events = _decode_events(Path(path), track, start_s, end_s)
    texts = ocr_images([e.image for e in events], language, workers)
    cues: list[Cue] = []
    for ev, text in zip(events, texts, strict=True):
        if not text:
            continue
        end = ev.end_s if ev.end_s is not None else ev.start_s + FALLBACK_CUE_S
        if cues and cues[-1].text == text:
            cues[-1] = Cue(cues[-1].start_s, max(cues[-1].end_s, end), text)
            continue
        cues.append(Cue(ev.start_s, end, text))
    return cues


# --------------------------------------------------------------------------- quality

_WORD_PUNCT = re.compile(r"[^\w\s]|_")


def _fallback_normalize(text: str) -> str:
    return " ".join(_WORD_PUNCT.sub(" ", text.replace("'", "").replace("’", "").lower()).split())


def _normalize(text: str) -> str:
    """§8.3 normalization from analysis.subtitles, so OCR words match the vocabulary's form."""
    try:
        from showvenist.analysis.subtitles import normalize
    except ImportError:  # pragma: no cover - subtitles.normalize not available
        return _fallback_normalize(text)
    return normalize(text)


def _tessdata_dir() -> Path | None:
    env = os.environ.get("TESSDATA_PREFIX")
    if env:
        p = Path(env)
        return p / "tessdata" if (p / "tessdata").is_dir() else p
    try:
        out = subprocess.run(
            [pytesseract.pytesseract.tesseract_cmd, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r'"([^"]+)"', out)
    return Path(m.group(1)) if m else None


def parse_wordlist(lines: Iterable[str]) -> frozenset[str]:
    """Dictionary entries -> normalized form (lowercase, apostrophes dropped); entries with
    other non-letters (URLs, hyphenations, numbers) are skipped."""
    words = set()
    for line in lines:
        w = line.strip().lower().replace("'", "")
        if w.isalpha():
            words.add(w)
    return frozenset(words)


@functools.lru_cache(maxsize=8)
def dictionary(language: str) -> frozenset[str]:
    """Tesseract's word list for the language (see module docstring); empty if unavailable."""
    lang = OCR_LANG_FOR.get(language, language)
    tessdata = _tessdata_dir()
    combine, dawg2 = shutil.which("combine_tessdata"), shutil.which("dawg2wordlist")
    if tessdata is None or combine is None or dawg2 is None:
        return frozenset()
    traineddata = tessdata / f"{lang}.traineddata"
    if not traineddata.is_file():
        return frozenset()
    with tempfile.TemporaryDirectory(prefix="showvenist-dict-") as tmp:
        prefix = str(Path(tmp) / f"{lang}.")
        try:
            subprocess.run([combine, "-u", str(traineddata), prefix], capture_output=True,
                           check=True, timeout=60)
            out = Path(tmp) / "words.txt"
            subprocess.run([dawg2, prefix + "lstm-unicharset", prefix + "lstm-word-dawg",
                            str(out)], capture_output=True, check=True, timeout=60)
            return parse_wordlist(out.read_text(encoding="utf-8", errors="replace").splitlines())
        except (OSError, subprocess.SubprocessError):
            return frozenset()


def ocr_quality(texts: list[str], vocabulary: set[str], language: str = "en") -> float | None:
    """Share of the texts' normalized words found in vocabulary ∪ the language dictionary.
    None when there are no words. ``vocabulary`` holds normalized words (show subtitles and
    bundle text)."""
    words = [w for t in texts for w in _normalize(t).split()]
    if not words:
        return None
    dic = dictionary(language)
    return sum(w in vocabulary or w in dic for w in words) / len(words)
