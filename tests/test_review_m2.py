"""Regression tests for the M2 review fixes (windows, OCR edges, errors, cache)."""

from __future__ import annotations

import math
import os
import shutil
import threading
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from showvenist.analysis import ocr as ocr_mod
from showvenist.analysis import types
from showvenist.analysis import windows as W
from showvenist.analysis.probe import probe
from showvenist.analysis.types import AnalysisCache, Cue, SubtitleReadError, SubtitleTrack
from showvenist.config import AnalysisConfig
from tests.fixtures import bitmapsubs as B
from tests.fixtures.bitmapsubs import make_mkv_with_bitmap_subs
from tests.fixtures.textsubs import make_mkv_with_text_subs
from tests.test_probe import make_mkv

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="no tesseract")

WORDS = "hello there we should leave now the door is open".split()
CUES = [(3.0 * i, 3.0 * i + 2.0, f"{WORDS[i % 10]} {WORDS[(i + 3) % 10]}") for i in range(60)]
CFG = AnalysisConfig(window_min_lines=8, window_max_minutes=1)


class _Reads:
    def __init__(self, monkeypatch):
        self.n = 0
        real_t, real_o = W.read_text_cues, W.ocr_cues

        def t(*a, **k):
            self.n += 1
            return real_t(*a, **k)

        def o(*a, **k):
            self.n += 1
            return real_o(*a, **k)

        monkeypatch.setattr(W, "read_text_cues", t)
        monkeypatch.setattr(W, "ocr_cues", o)


# --------------------------------------------------------------------------- 2: stale cache


def test_bumping_text_extractor_version_rereads_windows(tmp_path, monkeypatch):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, duration_s=200)
    cache = AnalysisCache(tmp_path / "a.db")
    reads = _Reads(monkeypatch)
    W.extract_dialogue(p, "en", set(), CFG, cache)
    n = reads.n
    W.extract_dialogue(p, "en", set(), CFG, cache)
    assert reads.n == n
    monkeypatch.setitem(types.EXTRACTOR_VERSIONS, "text", types.EXTRACTOR_VERSIONS["text"] + 1)
    W.extract_dialogue(p, "en", set(), CFG, cache)
    assert reads.n > n
    # Bumping the OCR extractor doesn't touch text windows.
    n = reads.n
    monkeypatch.setitem(types.EXTRACTOR_VERSIONS, "ocr", types.EXTRACTOR_VERSIONS["ocr"] + 1)
    W.extract_dialogue(p, "en", set(), CFG, cache)
    assert reads.n == n
    cache.close()


# --------------------------------------------------------------------------- 1: one probe


def test_extract_dialogue_uses_given_probe_and_cached_probe(tmp_path, monkeypatch):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, duration_s=200)
    info = probe(p)
    calls = []
    monkeypatch.setattr(W, "probe", lambda *a, **k: calls.append(a) or info)
    d = W.extract_dialogue(p, "en", set(), CFG, None, probe_result=info)
    assert calls == [] and d.track is not None


def test_unknown_count_tracks_fall_back_on_stage2_lines(tmp_path, monkeypatch):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, duration_s=200)
    info = probe(p)
    real = info.subtitle_tracks[0]
    assert real.events == -1
    # A first (lower-index) unknown track that reads nothing: the real one must win.
    ghost = SubtitleTrack(0, "subrip", "eng", events=-1)
    info = types.MediaProbe(**{**info.__dict__, "subtitle_tracks": [ghost, real]})
    real_read = W.read_text_cues
    monkeypatch.setattr(
        W, "read_text_cues", lambda pa, tr, a, b: [] if tr is ghost else real_read(pa, tr, a, b)
    )
    d = W.extract_dialogue(p, "en", set(), CFG, None, probe_result=info)
    assert d.track == real and d.text
    assert d.note and "without an event count" in d.note


# --------------------------------------------------------------------------- 6: read errors


def test_truncated_file_windows_fail_soft_and_are_not_cached(tmp_path):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, duration_s=200)
    info = probe(p)
    data = p.read_bytes()
    p.write_bytes(data[:64])  # replaced by a broken file after probing
    cache = AnalysisCache(tmp_path / "a.db")
    d = W.extract_dialogue(p, "en", set(), CFG, cache, probe_result=info)
    assert d.track is not None and len(d.windows) == 3
    assert all(w.error and w.lines == 0 for w in d.windows)
    assert d.note and "read failed" in d.note
    fp = W.fingerprint(p)
    assert cache._db.execute(
        "SELECT count(*) FROM analysis WHERE fingerprint=? AND extractor='window'", (fp,)
    ).fetchone() == (0,)
    cache.close()


def test_mid_window_error_keeps_cues_read_so_far(tmp_path, monkeypatch):
    track = SubtitleTrack(2, "subrip", "eng", events=-1)
    calls = []

    def flaky(path, tr, a, b):
        calls.append(a)
        if len(calls) == 2:
            raise SubtitleReadError("InvalidDataError: boom", [Cue(a + 1, a + 2, "partial")])
        return [Cue(a + 1, a + 2, f"chunk {a}")]

    monkeypatch.setattr(W, "read_text_cues", flaky)
    w, _ = W._read_window(Path("x"), track, 0.0, 300.0, "en", CFG, 0)
    assert [c.text for c in w.cues] == ["chunk 0.0", "partial"]
    assert w.error == "InvalidDataError: boom" and len(calls) == 2


def test_ocr_cues_wraps_open_errors(tmp_path):
    p = tmp_path / "missing.mkv"
    with pytest.raises(SubtitleReadError):
        ocr_mod.ocr_cues(p, SubtitleTrack(1, "hdmv_pgs_subtitle", "eng"), 0, 10, "en")
    from showvenist.analysis.subtitles import read_text_cues

    with pytest.raises(SubtitleReadError):
        read_text_cues(p, SubtitleTrack(1, "subrip", "eng"), 0, 10)


# --------------------------------------------------------------------------- 8: OMP


def test_tesseract_env_single_threaded():
    assert os.environ.get("OMP_THREAD_LIMIT") is not None


# --------------------------------------------------------------------------- 9: chunk edges


def test_pgs_cue_straddling_chunk_edge_keeps_its_end(tmp_path, monkeypatch):
    p = tmp_path / "p.mkv"
    make_mkv_with_bitmap_subs(p, [(58.0, 62.0, "edge"), (80.0, 82.0, "b")], "pgs", duration_s=90)
    track = SubtitleTrack(1, "hdmv_pgs_subtitle", "eng")
    ev = ocr_mod._decode_events(p, track, 0.0, 60.0)
    assert [(e.start_s, e.end_s) for e in ev] == [(58.0, 62.0)]  # was 58 + 5 s fallback

    monkeypatch.setattr(ocr_mod, "ocr_images", lambda ims, *a, **k: ["Edge cue"] * len(ims))
    w, _ = W._read_window(p, track, 0.0, 90.0, "en", CFG, 0)
    edge = [c for c in w.cues if c.start_s == 58.0]
    assert len(edge) == 1 and edge[0].end_s == 62.0


def test_text_cue_straddling_chunk_edge_appears_once(tmp_path):
    p = tmp_path / "t.mkv"
    make_mkv(p, [(10.0, 11.0, "one"), (59.0, 61.5, "edge")], seconds=90)
    track = probe(p).subtitle_tracks[0]
    w, _ = W._read_window(p, track, 0.0, 90.0, "en", CFG, 0)
    assert [(c.start_s, c.end_s, c.text) for c in w.cues if c.text == "edge"] == [
        (59.0, 61.5, "edge")
    ]


def test_dedupe_tolerates_ocr_end_and_case_differences(monkeypatch):
    track = SubtitleTrack(1, "hdmv_pgs_subtitle", "eng")
    batches = iter([[Cue(59.0, 64.0, "Edge cue.")], [Cue(59.04, 62.0, "edge cue")]])
    monkeypatch.setattr(W, "ocr_cues", lambda *a, **k: next(batches))
    w, _ = W._read_window(Path("x"), track, 0.0, 120.0, "en", CFG, 0)
    assert len(w.cues) == 1


# --------------------------------------------------------------------------- 10: palettes


def _pgs_mkv_palette_only_at_start(path: Path, cues) -> None:
    """PGS MKV with a keyframe every second; only the first display set carries a PDS, as in
    an epoch whose later display sets reuse its palette."""
    size = (720, 480)
    events = []
    for i, (a, b, text) in enumerate(cues):
        show, clear = B._pgs_display_sets(B.render_text(text, size), size, 2 * i)
        if i:  # normal-state display set (not an epoch start) without a PDS
            pds_at = show.index(b"\x14\x00\x0c")  # PDS: type 0x14, 12-byte body
            show = show[:pds_at] + show[pds_at + 15 :]
            show = show[:10] + b"\x00" + show[11:]  # PCS composition_state
        events += [(round(a * 1000), show, round((b - a) * 1000)), (round(b * 1000), clear, 0)]
    with av.open(str(path), "w", format="matroska") as out:
        vs = out.add_stream("mpeg4", rate=1)
        vs.width, vs.height, vs.pix_fmt = 720, 480, "yuv420p"
        vs.codec_context.gop_size = 1
        ss = out.add_mux_stream("hdmv_pgs_subtitle")
        ss.time_base = B.MS
        ss.metadata["language"] = "eng"
        black = av.VideoFrame.from_ndarray(np.zeros((480, 720, 3), np.uint8), format="rgb24")
        ei = 0
        total = int(cues[-1][1]) + 2
        for f in range(total):
            while ei < len(events) and events[ei][0] <= f * 1000:
                B._mux_sub(out, ss, *events[ei])
                ei += 1
            black.pts, black.time_base = f, Fraction(1, 1)
            for pkt in vs.encode(black):
                out.mux(pkt)
        for pkt in vs.encode(None):
            out.mux(pkt)


def test_pgs_mid_file_seek_skips_events_until_a_palette(tmp_path, monkeypatch):
    p = tmp_path / "p.mkv"
    _pgs_mkv_palette_only_at_start(p, [(1.0, 2.0, "first"), (40.0, 41.0, "later")])
    track = SubtitleTrack(1, "hdmv_pgs_subtitle", "eng")
    luts = []
    real_render = ocr_mod.render
    monkeypatch.setattr(
        ocr_mod, "render", lambda bms, lut, up=1: luts.append(lut) or real_render(bms, lut, up)
    )
    # From the start the palette is seen and the later cue renders with it.
    assert len(ocr_mod._decode_events(p, track, 0.0, 60.0)) == 2
    luts.clear()
    # Seeking past the PDS: no palette yet, so no render with the identity ramp.
    assert ocr_mod._decode_events(p, track, 30.0, 60.0) == []
    assert luts == []


@needs_tesseract
def test_pgs_mid_file_window_with_palette_ocrs_correctly(tmp_path):
    p = tmp_path / "p.mkv"
    make_mkv_with_bitmap_subs(p, CUES, "pgs", duration_s=200)
    track = SubtitleTrack(1, "hdmv_pgs_subtitle", "eng")
    cues = ocr_mod.ocr_cues(p, track, 100.0, 115.0, "en")
    assert [c.text.lower() for c in cues] == [t for a, _, t in CUES if 100 <= a <= 115]


# --------------------------------------------------------------------------- 11: duration


def test_window_spans_without_duration_is_one_full_span():
    assert W.window_spans(0.0) == [(0.0, math.inf)]
    assert W.window_spans(None) == [(0.0, math.inf)]


def test_window_spans_stop_at_next_window():
    s = W.window_spans(200.0, max_s=300.0)
    assert s == [(40.0, 100.0), (100.0, 160.0), (160.0, 200.0)]


def test_no_duration_reads_whole_track(tmp_path):
    p = tmp_path / "t.mkv"
    make_mkv(p, CUES, seconds=200)
    info = probe(p)
    info = types.MediaProbe(**{**info.__dict__, "duration_s": 0.0})
    cache = AnalysisCache(tmp_path / "a.db")
    d = W.extract_dialogue(p, "en", set(), CFG, cache, probe_result=info)
    assert d.full and d.note and "duration unknown" in d.note
    assert [c.text for w in d.windows for c in w.cues] == [t for _, _, t in CUES]
    d2 = W.extract_dialogue(p, "en", set(), CFG, cache, probe_result=info)  # cached (inf JSON)
    assert [w.cues for w in d2.windows] == [w.cues for w in d.windows]
    cache.close()


def test_short_file_windows_do_not_double_count(tmp_path):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, duration_s=200)
    cfg = AnalysisConfig(window_min_lines=1000, window_max_minutes=5)  # read to the limit
    d = W.extract_dialogue(p, "en", set(), cfg)
    keys = [(c.start_s, c.text) for w in d.windows for c in w.cues]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- 14: threads


def test_cache_usable_across_threads(tmp_path):
    cache = AnalysisCache(tmp_path / "a.db")
    errors = []

    def work(i):
        try:
            for j in range(20):
                cache.put(f"fp{i}", "window", str(j), "{}")
                assert cache.get(f"fp{i}", "window", str(j)) == "{}"
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    cache.close()
