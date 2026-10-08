"""Title-card frame reading (spec §8.4, decision 54). Contract; the body is task sv-8ue.

`read_cards` seeks to each configured window, decodes about one frame per second, keeps only
mostly-black frames when `cfg.background == "dark"` (every sampled frame when "any"), and
OCRs the kept frames with Tesseract. Results are cached per file in analysis.db under the
extractor "cardframes", keyed by `card_params(cfg)`; a cache hit reads nothing from disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
from PIL import Image

from showvenist.analysis.ocr import ocr_images
from showvenist.analysis.types import AnalysisCache, CardRead, params_hash
from showvenist.bundle.schema import TitleCardConfig

FRAMES_PER_SECOND = 1.0
OCR_WIDTH = 960
DARK_LUMA = 40  # luma below this (histogram bins 0..39 of 256) counts as near black
DARK_FRACTION = 0.9  # a frame is "dark" when at least this share of pixels is near black


def card_params(cfg: TitleCardConfig) -> str:
    """Cache key part: changes when the windows or background change. The pattern is not part
    of it (it only filters the cached OCR text), so editing the regex never re-reads video."""
    return params_hash(
        windows=[(w.edge, w.skip_seconds, w.length_seconds) for w in cfg.windows],
        background=cfg.background,
        fps=FRAMES_PER_SECOND,
        dark=DARK_FRACTION,
    )


def _spans(cfg: TitleCardConfig, duration_s: float) -> list[tuple[float, float]]:
    spans = []
    for w in cfg.windows:
        if w.edge == "start":
            a, b = w.skip_seconds, w.skip_seconds + w.length_seconds
        else:
            a, b = duration_s - w.skip_seconds - w.length_seconds, duration_s - w.skip_seconds
        a, b = max(a, 0.0), min(b, duration_s)
        if b > a:
            spans.append((a, b))
    return spans


def _grab(path: Path, spans: list[tuple[float, float]], dark: bool) -> tuple[list, bool]:
    """Sample frames over `spans`: (list of (offset_s, gray PIL image)), ok. `ok` is False when
    a media error cut the read short. Dark filtering happens here, before any OCR."""
    out: dict[float, Image.Image] = {}
    ok = True
    try:
        with av.open(str(path)) as c:
            stream = c.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)
            for a, b in spans:
                try:
                    c.seek(int(a / tb), stream=stream, backward=True)
                    stream.codec_context.flush_buffers()
                    nxt = a
                    for frame in c.decode(stream):
                        if frame.pts is None:
                            continue
                        t = float(frame.pts) * tb
                        if t >= b:
                            break
                        if t < nxt:
                            continue
                        nxt = max(nxt, t) + 1.0 / FRAMES_PER_SECOND
                        key = round(t, 3)
                        if key in out:
                            continue
                        w = min(frame.width, OCR_WIDTH)
                        h = max(2, round(frame.height * w / frame.width / 2) * 2)
                        img = Image.fromarray(
                            frame.reformat(width=w, height=h, format="gray").to_ndarray())
                        if dark and np.mean(np.asarray(img) < DARK_LUMA) < DARK_FRACTION:
                            continue
                        out[key] = img
                except (av.error.FFmpegError, OSError, ValueError):
                    ok = False
    except (av.error.FFmpegError, OSError, IndexError, ValueError):
        ok = False
    return sorted(out.items()), ok


def read_cards(
    path: Path,
    fingerprint: str,
    duration_s: float,
    cfg: TitleCardConfig,
    cache: AnalysisCache | None = None,
) -> list[CardRead]:
    """OCR text per sampled frame over `cfg.windows`, sorted by offset (frames with no text are
    left out). A window is clipped to the file (`duration_s`). Never raises on a read failure:
    returns what it could read."""
    params = card_params(cfg)
    if cache is not None:
        hit = cache.get(fingerprint, "cardframes", params)
        if hit is not None:
            try:
                return [CardRead(float(o), str(t)) for o, t in json.loads(hit)]
            except (ValueError, TypeError):
                pass
    frames, ok = _grab(Path(path), _spans(cfg, duration_s), cfg.background == "dark")
    reads: list[CardRead] = []
    try:
        texts = ocr_images([im for _, im in frames], "en")
        for (t, _), text in zip(frames, texts, strict=True):
            text = " ".join(text.split())
            if text:
                reads.append(CardRead(t, text))
    except Exception:  # tesseract missing or crashed: keep what we have, don't cache
        ok = False
    if ok and cache is not None:
        payload = json.dumps([[r.offset_s, r.text] for r in reads])
        cache.put(fingerprint, "cardframes", params, payload)
    return reads
