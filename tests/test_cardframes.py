"""read_cards on a tiny synthetic video (sv-8ue): white 'EPISODE XXVII' on black near the end."""

import shutil
import time

import av
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from showvenist.analysis import cardframes
from showvenist.analysis.cardframes import card_params, read_cards
from showvenist.analysis.types import AnalysisCache
from showvenist.bundle.schema import TitleCardConfig

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")

W, H, FPS, DUR = 320, 180, 10, 30
CARD = (22, 26)  # card on screen from 22 s to 26 s


def font():
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, 28)
        except OSError:
            pass
    return ImageFont.load_default(28)


def card_frame():
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).text((W // 2, H // 2), "EPISODE XXVII", fill=255, anchor="mm", font=font())
    return np.asarray(img)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("v") / "card.mkv"
    rng = np.random.default_rng(1)
    card = card_frame()
    with av.open(str(path), "w") as out:
        s = out.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        s.options = {"g": "10", "preset": "ultrafast", "crf": "18"}
        for i in range(DUR * FPS):
            t = i / FPS
            if CARD[0] <= t < CARD[1]:
                g = card
            elif t >= 27:
                g = np.zeros((H, W), np.uint8)
            else:
                g = rng.integers(60, 255, (H, W), dtype=np.uint8)
            f = av.VideoFrame.from_ndarray(np.stack([g] * 3, -1), format="rgb24")
            for p in s.encode(f):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    return path


def cfg(windows, background="dark"):
    return TitleCardConfig.model_validate({
        "windows": windows, "pattern": r"EPISODE\s+(?P<ref>[IVXLC]+)", "kind": "roman",
        "background": background})


END = {"from": "end", "skip_seconds": 0, "length_seconds": 12}


def test_finds_card_in_end_window(video):
    reads = read_cards(video, "fp", DUR, cfg([END]))
    assert reads and all(CARD[0] - 1 <= r.offset_s < CARD[1] + 1 for r in reads)
    assert any("EPISODE" in r.text and "XXVII" in r.text for r in reads)
    assert [r.offset_s for r in reads] == sorted(r.offset_s for r in reads)


def test_card_free_window_is_empty(video):
    assert read_cards(video, "fp", DUR, cfg([{"from": "start", "skip_seconds": 2,
                                              "length_seconds": 10}])) == []


def test_dark_prefilter_sends_fewer_frames_to_ocr(video, monkeypatch):
    counts = {}
    real = cardframes.ocr_images

    def spy(images, lang, *a, **k):
        counts["n"] = len(images)
        return real(images, lang, *a, **k)

    monkeypatch.setattr(cardframes, "ocr_images", spy)
    read_cards(video, "a", DUR, cfg([END], "dark"))
    dark = counts["n"]
    read_cards(video, "b", DUR, cfg([END], "any"))
    assert 0 < dark < counts["n"]


def test_overlapping_windows_do_not_duplicate(video):
    one = read_cards(video, "a", DUR, cfg([END]))
    two = read_cards(video, "b", DUR, cfg([END, {"from": "end", "skip_seconds": 2,
                                                "length_seconds": 8}]))
    assert [r.offset_s for r in two] == [r.offset_s for r in one]


def test_cache_hit_reads_no_disk(video, tmp_path):
    cache = AnalysisCache(tmp_path / "a.db")
    first = read_cards(video, "fp", DUR, cfg([END]), cache)
    assert first
    again = read_cards(tmp_path / "gone.mkv", "fp", DUR, cfg([END]), cache)
    assert again == first
    empty_cfg = cfg([{"from": "start", "skip_seconds": 2, "length_seconds": 5}])
    assert read_cards(video, "fp", DUR, empty_cfg, cache) == []
    assert cache.get("fp", "cardframes", card_params(empty_cfg)) == "[]"
    assert read_cards(tmp_path / "gone.mkv", "fp", DUR, empty_cfg, cache) == []


def test_bad_file_returns_empty_and_caches_nothing(tmp_path):
    cache = AnalysisCache(tmp_path / "a.db")
    c = cfg([END])
    assert read_cards(tmp_path / "missing.mkv", "fp", DUR, c, cache) == []
    junk = tmp_path / "junk.mkv"
    junk.write_bytes(b"not a video" * 100)
    assert read_cards(junk, "fp2", DUR, c, cache) == []
    assert cache.get("fp", "cardframes", card_params(c)) is None
    assert cache.get("fp2", "cardframes", card_params(c)) is None


def test_windows_clipped_to_duration(video):
    wins = [{"from": "end", "skip_seconds": 0, "length_seconds": 500},
            {"from": "start", "skip_seconds": 29, "length_seconds": 50},
            {"from": "start", "skip_seconds": 40, "length_seconds": 5}]
    spans = cardframes._spans(cfg(wins), DUR)
    assert spans == [(0.0, 30.0), (29.0, 30.0)]
    t = time.time()
    assert read_cards(video, "fp", DUR, cfg([{"from": "end", "skip_seconds": 0,
                                              "length_seconds": 500}]))
    assert time.time() - t < 8
