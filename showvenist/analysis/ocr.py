"""OCR for image-based subtitles: PGS (Blu-ray) and VobSub (DVD) (spec §8.3).

Ported from the M0 spike (spike/extract.py), where it was validated on real rips.

Pipeline: demux the subtitle stream with PyAV from ``start_s``; feed every packet's raw bytes to
a palette tracker (PyAV decodes bitmaps to raw palette indices but doesn't expose the palette,
so it is parsed here); decode the packet to bitmap rects; ``render`` composites them into one
dark-on-white image (palette -> luminance, upscale for low-resolution VobSub, binarize, invert,
pad); Tesseract reads the images in a thread pool, a batch of images per ``tesseract`` process.

The two phases are separate so the M4 pipeline (analysis.pipeline) can run them in different
stages: ``decode_events`` is the I/O phase (demux, decode, render), ``ocr_events`` the compute
phase; ``ocr_cues`` is their composition.

Threads, not processes: each batch runs in its own ``tesseract`` subprocess, so a thread pool
already uses every core without pickling images across processes.

Word check for ``ocr_quality`` (``is_word``): ``wordfreq`` Zipf frequency >= WORD_ZIPF_FLOOR,
with apostrophes restored for contractions (our normalization drops them: "theyre" is checked as
"they're") and a whitelist for one- and two-letter words, because web frequencies rate OCR junk
such as "rn", "ii" and "lt" as common. Tuned on the M0 spike's OCR text (93k tokens): at floor 3.0
it accepts 97.8% of real dialogue tokens before any show vocabulary (the misses are names and
initials, which the vocabulary covers) and accepts 3.0% of synthetically OCR-corrupted words,
against 8.8% for Tesseract's own word list.
"""

from __future__ import annotations

import functools
import math
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import av
import numpy as np
import pytesseract
from PIL import Image

from showvenist.analysis.types import Cue, SubtitleReadError, SubtitleTrack
from showvenist.bundle.schema import ocr_language

# One tesseract subprocess per pool thread already uses every core; OpenMP threads inside each
# would oversubscribe the CPU. pytesseract's subprocesses inherit this environment.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

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


def _clean(text: str) -> str:
    for rx, rep in _OCR_FIXES:
        text = rx.sub(rep, text)
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())


def ocr_image(img: Image.Image, language: str = "en") -> str:
    """Tesseract one rendered subtitle image; returns trimmed non-empty lines."""
    return _clean(pytesseract.image_to_string(img, lang=ocr_language(language), config=TESS_CONFIG))


# Images per tesseract process. Startup (loading the model) costs about as much as reading
# three or four subtitle images, so one process per image wastes most of the CPU; a batch of
# BATCH_SIZE amortizes it while still spreading a window's images over the pool.
BATCH_SIZE = 16
_PAGE_SEP = "\f"  # tesseract's default page_separator, written after every page but the last


def ocr_batch(images: Sequence[Image.Image], language: str = "en") -> list[str]:
    """OCR several images in ONE tesseract process (a list file of PNGs: each image is a page,
    recognized independently; the LSTM engine keeps no state across pages, and on real PGS rips
    the text is identical to one process per image). Falls back to ``ocr_image`` per image if
    the process fails or the page count doesn't match."""
    if len(images) <= 1:
        return [ocr_image(im, language) for im in images]
    cmd = pytesseract.pytesseract.tesseract_cmd
    try:
        with tempfile.TemporaryDirectory(prefix="sv-ocr-") as tmp:
            names = []
            for i, im in enumerate(images):
                name = os.path.join(tmp, f"{i:05d}.png")
                im.save(name, format="PNG")
                names.append(name)
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w", encoding="utf-8") as f:
                f.write("".join(n + "\n" for n in names))
            out = os.path.join(tmp, "out")
            args = [cmd, lst, out, "-l", ocr_language(language), *shlex.split(TESS_CONFIG)]
            proc = subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace"))
            with open(out + ".txt", encoding="utf-8") as f:
                text = f.read()
    except (OSError, RuntimeError):
        return [ocr_image(im, language) for im in images]
    pages = text.split(_PAGE_SEP)
    if len(pages) == len(images) + 1 and not pages[-1].strip():
        pages.pop()
    if len(pages) != len(images):
        return [ocr_image(im, language) for im in images]
    return [_clean(t) for t in pages]


def batches(n: int, workers: int, size: int = BATCH_SIZE) -> list[range]:
    """Split ``n`` items into contiguous batches of at most ``size``, at least one per worker
    while there are fewer than ``workers`` batches (so a lone window still uses the pool)."""
    if n <= 0:
        return []
    k = max(math.ceil(n / size), min(workers, n))
    base, extra = divmod(n, k)
    out, pos = [], 0
    for i in range(k):
        m = base + (i < extra)
        out.append(range(pos, pos + m))
        pos += m
    return out


def _pool_size(workers: int) -> int:
    return workers if workers > 0 else (os.cpu_count() or 1)


def ocr_images(images: Sequence[Image.Image], language: str, workers: int = 0) -> list[str]:
    """OCR images in a thread pool of tesseract processes, batched (see ``ocr_batch``)."""
    if not images:
        return []
    parts = batches(len(images), _pool_size(workers))
    with ThreadPoolExecutor(min(_pool_size(workers), len(parts))) as pool:
        done = pool.map(lambda r: ocr_batch([images[i] for i in r], language), parts)
        return [t for texts in done for t in texts]


@dataclass
class BitmapEvent:
    """One decoded bitmap subtitle event, rendered and ready for OCR."""

    start_s: float
    end_s: float | None  # None until bounded by packet duration or the next packet
    image: Image.Image


_Event = BitmapEvent  # pre-M4 name


class BitmapDecodeError(Exception):
    """Decoding failed part way; ``events`` holds the events decoded before the failure."""

    def __init__(self, message: str, events: list[BitmapEvent]):
        super().__init__(message)
        self.events = events


_DecodeError = BitmapDecodeError  # pre-M4 name


def decode_events(
    path: Path, track: SubtitleTrack, start_s: float, end_s: float
) -> list[BitmapEvent]:
    """Decode phase of ``ocr_cues`` (I/O, no OCR): the bitmap events starting in
    [start_s, end_s], rendered to images. Raises BitmapDecodeError (with the events decoded so
    far) when the file can't be opened, seeked or demuxed."""
    kind = track.kind
    if kind not in ("pgs", "vobsub"):
        raise ValueError(f"not a bitmap subtitle track: {track.codec}")
    upscale = VOBSUB_UPSCALE if kind == "vobsub" else 1
    events: list[BitmapEvent] = []
    try:
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
                # Any packet (an empty PGS clear included) ends the cue on screen before it,
                # including the first packet past the window: close it before stopping.
                if events and (events[-1].end_s is None or events[-1].end_s > t):
                    events[-1].end_s = max(t, events[-1].start_s)
                if t > end_s:
                    break
                data = bytes(pkt)
                pal.feed(data)  # palette state must see packets before the window too
                try:
                    rects = pkt.decode()
                except av.error.FFmpegError:
                    continue
                if t < start_s:
                    continue
                # After a mid-file seek a PGS display set may reuse a palette defined before
                # the seek point; without it lut() is an identity ramp and renders garbage.
                if isinstance(pal, PgsPalette) and not pal.pals:
                    continue
                bitmaps = [b for b in map(bitmap_from_av, rects) if b is not None]
                img = render(bitmaps, pal.lut(), upscale)
                if img is None:
                    continue
                dur = float(pkt.duration * pkt.time_base) if pkt.duration else 0.0
                events.append(BitmapEvent(t, t + dur if dur > 0 else None, img))
    except (av.error.FFmpegError, OSError) as e:
        raise BitmapDecodeError(f"{type(e).__name__}: {e}", events) from e
    return events


_decode_events = decode_events  # pre-M4 name


def cues_from_texts(events: Sequence[BitmapEvent], texts: Sequence[str]) -> list[Cue]:
    """OCR phase result -> cues: empty texts dropped, consecutive identical texts merged (PGS
    re-sends an epoch), unbounded events given FALLBACK_CUE_S."""
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


def ocr_events(events: Sequence[BitmapEvent], language: str, workers: int = 0) -> list[Cue]:
    """OCR phase of ``ocr_cues``: images -> text -> cues (no I/O)."""
    return cues_from_texts(events, ocr_images([e.image for e in events], language, workers))


def ocr_cues(
    path: Path,
    track: SubtitleTrack,
    start_s: float,
    end_s: float,
    language: str,
    workers: int = 0,
) -> list[Cue]:
    """OCR the bitmap subtitle cues starting in [start_s, end_s]: ``decode_events`` then
    ``ocr_events``. Text is unnormalized; consecutive identical cues (PGS re-sends an epoch) are
    merged; empty OCR results dropped. Raises SubtitleReadError (with the cues OCR'd before the
    failure) when the file can't be opened, seeked or demuxed."""
    try:
        events = decode_events(Path(path), track, start_s, end_s)
    except BitmapDecodeError as e:
        raise SubtitleReadError(str(e), ocr_events(e.events, language, workers)) from e
    return ocr_events(events, language, workers)


_events_to_cues = ocr_events  # pre-M4 name


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


WORD_ZIPF_FLOOR = 3.0
_SHORT_WORDS = frozenset(
    "a i am an as at be by do go he hi if in is it me my no of oh ok on or so to up us we "
    "ya yo ah uh um mm ha hm eh em id im mr dr st".split()
)
_CONTRACTION_SUFFIXES = (("s", 1), ("t", 1), ("d", 1), ("ll", 2), ("re", 2), ("ve", 2), ("m", 1))


def _zipf(word: str, language: str) -> float:
    from wordfreq import zipf_frequency

    best = zipf_frequency(word, language)
    if language == "en":
        cands = []
        for suf, n in _CONTRACTION_SUFFIXES:
            if word.endswith(suf) and len(word) > n + 1:
                cands.append(f"{word[:-n]}'{suf}")
        if word.endswith("nt") and len(word) > 3:
            cands.append(f"{word[:-2]}n't")
        if word == "maam":
            cands.append("ma'am")
        best = max([best] + [zipf_frequency(c, language) for c in cands])
    return best


@functools.lru_cache(maxsize=65536)
def is_word(word: str, language: str = "en") -> bool:
    """True if `word` (normalized: lowercase, no apostrophes) is an ordinary word."""
    if word.isdigit():
        return True
    if len(word) <= 2:
        return language == "en" and word in _SHORT_WORDS
    return _zipf(word, language) >= WORD_ZIPF_FLOOR


def ocr_quality(texts: list[str], vocabulary: set[str], language: str = "en") -> float | None:
    """Share of the texts' normalized words found in vocabulary or accepted by is_word().
    None when there are no words. ``vocabulary`` holds normalized words (show subtitles and
    bundle text)."""
    words = [w for t in texts for w in _normalize(t).split()]
    if not words:
        return None
    return sum(w in vocabulary or is_word(w, language) for w in words) / len(words)
