"""Title-card frame reading (spec §8.4, decision 54). Contract; the body is task sv-8ue.

`read_cards` seeks to each configured window, decodes about one frame per second, keeps only
mostly-black frames when `cfg.background == "dark"` (every sampled frame when "any"), and
OCRs the kept frames with Tesseract. Results are cached per file in analysis.db under the
extractor "cardframes", keyed by `card_params(cfg)`; a cache hit reads nothing from disk.
"""

from __future__ import annotations

from pathlib import Path

from showvenist.analysis.types import AnalysisCache, CardRead, params_hash
from showvenist.bundle.schema import TitleCardConfig

FRAMES_PER_SECOND = 1.0
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
    raise NotImplementedError("sv-8ue")
