"""analysis.pipeline on synthetic MKVs (no real media): equivalence with sequential
extract_dialogue, order, cache, errors, the per-root I/O limit and stage overlap."""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest

from showvenist.analysis import pipeline as P
from showvenist.analysis import windows as W
from showvenist.analysis.cache import AnalysisCache
from showvenist.analysis.ocr import BitmapEvent
from showvenist.analysis.probe import probe
from showvenist.config import AnalysisConfig
from showvenist.state.fingerprint import content_key, fingerprint
from tests.fixtures.bitmapsubs import make_mkv_with_bitmap_subs
from tests.fixtures.textsubs import make_mkv_with_text_subs

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="no tesseract")

WORDS = "hello there we should leave now the door is open".split()
CFG = AnalysisConfig(window_min_lines=3, window_max_minutes=1, ocr_workers=4)


def _cues(seed: int, n: int = 60):
    return [
        (3.0 * i, 3.0 * i + 2.0, f"{WORDS[(i + seed) % 10]} {WORDS[(i * 3 + seed) % 10]}")
        for i in range(n)
    ]


def _text_files(tmp_path: Path, n: int) -> list[Path]:
    out = []
    for k in range(n):
        p = tmp_path / f"t{k}.mkv"
        make_mkv_with_text_subs(p, _cues(k), duration_s=200 + 10 * k)
        out.append(p)
    return out


def _dump(d):
    return (
        d.track,
        d.ocr_quality,
        d.full,
        d.note,
        [
            (w.start_s, w.end_s, w.error, [(c.start_s, c.end_s, c.text) for c in w.cues])
            for w in d.windows
        ],
    )


def _sequential(paths, cfg, cache=None):
    out = []
    for p in paths:
        info = probe(p, cache)
        out.append(W.extract_dialogue(p, "en", set(WORDS), cfg, cache, probe_result=info))
    return out


def _rows(cache: AnalysisCache):
    return sorted(
        cache._db.execute("SELECT fingerprint, extractor, params_hash, payload FROM analysis")
    )


@pytest.fixture
def decodes(monkeypatch):
    """Count chunk decodes in both the pipeline and the default reader."""
    calls = []
    real_dc, real_rt, real_oc = P.decode_chunk, W.read_text_cues, W.ocr_cues
    monkeypatch.setattr(P, "decode_chunk", lambda *a: calls.append(a) or real_dc(*a))
    monkeypatch.setattr(W, "read_text_cues", lambda *a: calls.append(a) or real_rt(*a))
    monkeypatch.setattr(W, "ocr_cues", lambda *a, **k: calls.append(a) or real_oc(*a, **k))
    return calls


def test_text_equivalence_order_and_cache(tmp_path, decodes):
    paths = _text_files(tmp_path, 4)
    seq_cache = AnalysisCache(tmp_path / "seq.db")
    seq = _sequential(paths, CFG, seq_cache)
    decodes.clear()

    cache = AnalysisCache(tmp_path / "pipe.db")
    seen = []
    res = P.analyze_files(
        paths,
        language="en",
        vocabulary=set(WORDS),
        cfg=CFG,
        cache=cache,
        progress=lambda *a: seen.append(a),
    )
    assert decodes, "the pipeline read the files"
    assert [r.path for r in res] == paths
    assert [_dump(r.dialogue) for r in res] == [_dump(d) for d in seq]
    assert [r.dialogue.text for r in res] == [d.text for d in seq]
    for r in res:
        assert r.error is None and r.duplicate_of is None
        assert r.probe == probe(r.path)
        assert r.fingerprint == fingerprint(r.path) and r.content_key == content_key(r.path)
    assert [s[0] for s in seen] == [1, 2, 3, 4] and {s[2] for s in seen} == set(paths)
    assert all(s[1] == 4 and s[3] is None for s in seen)
    # identical cache entries, and a second run reads nothing
    assert _rows(cache) == _rows(seq_cache)
    decodes.clear()
    again = P.analyze_files(paths, language="en", vocabulary=set(WORDS), cfg=CFG, cache=cache)
    assert decodes == []
    assert [_dump(r.dialogue) for r in again] == [_dump(d) for d in seq]
    _sequential(paths, CFG, cache)  # sequential calls hit the pipeline's entries too
    assert decodes == []
    cache.close()
    seq_cache.close()


@needs_tesseract
@pytest.mark.parametrize("kind", ["pgs", "vobsub"])
def test_bitmap_equivalence_with_sequential(tmp_path, kind):
    paths = []
    for k in range(3):
        p = tmp_path / f"{kind}{k}.mkv"
        make_mkv_with_bitmap_subs(p, _cues(k, 40), kind, duration_s=130)
        paths.append(p)
    seq = _sequential(paths, CFG)
    res = P.analyze_files(paths, language="en", vocabulary=set(WORDS), cfg=CFG, cache=None)
    assert [_dump(r.dialogue) for r in res] == [_dump(d) for d in seq]
    assert all(r.dialogue.ocr_quality and r.dialogue.ocr_quality > 0.8 for r in res)


def test_errors_and_duplicates_are_isolated(tmp_path):
    good = _text_files(tmp_path, 2)
    junk = tmp_path / "junk.mkv"
    junk.write_bytes(b"not a matroska file" * 5000)
    missing = tmp_path / "missing.mkv"
    dup = tmp_path / "dup.mkv"
    shutil.copy(good[0], dup)
    paths = [missing, good[0], junk, dup, good[1]]
    seen = []
    res = P.analyze_files(
        paths,
        language="en",
        vocabulary=set(),
        cfg=CFG,
        cache=None,
        progress=lambda *a: seen.append(a),
    )
    assert [r.path for r in res] == paths
    assert res[0].error and res[0].error.startswith("FileNotFoundError")
    assert res[2].error and res[2].dialogue is None
    assert res[3].duplicate_of == good[0] and res[3].dialogue is None and res[3].error is None
    assert res[1].dialogue.windows and res[4].dialogue.windows
    assert sorted(s[0] for s in seen) == [1, 2, 3, 4, 5]
    errs = {s[2]: s[3] for s in seen}
    assert errs[missing] == res[0].error and errs[junk] == res[2].error and errs[good[1]] is None

    known = {res[1].content_key: Path("/elsewhere/first.mkv")}
    res2 = P.analyze_files(
        [good[0]], language="en", vocabulary=set(), cfg=CFG, cache=None, known_keys=known
    )
    assert res2[0].duplicate_of == Path("/elsewhere/first.mkv") and res2[0].dialogue is None


def _fake_bitmap_stage(monkeypatch, decode_s: float, ocr_s: float):
    """Slow fake decode (one event per chunk) and slow fake OCR, recording intervals."""
    log: list[tuple[str, Path, float, float]] = []
    lock = threading.Lock()
    active: dict[object, int] = {}
    peak = [0]

    def decode(path, track, a, b):
        root = P.storage_root(path)
        with lock:
            active[root] = active.get(root, 0) + 1
            peak[0] = max(peak[0], active[root])
        t0 = time.monotonic()
        time.sleep(decode_s)
        with lock:
            active[root] -= 1
            log.append(("decode", path, t0, time.monotonic()))
        ev = BitmapEvent(a + 1.0, a + 2.0, (path.name, a))
        return W.DecodedChunk(track, events=[ev])

    def ocr(images, language):
        t0 = time.monotonic()
        time.sleep(ocr_s)
        with lock:
            log.append(("ocr", images[0][0], t0, time.monotonic()))
        return [f"{name} {WORDS[int(a) % 10]} {a}" for name, a in images]

    monkeypatch.setattr(P, "decode_chunk", decode)
    monkeypatch.setattr(P, "ocr_batch", ocr)
    return log, peak


def _pgs_files(tmp_path, n):
    paths = []
    for k in range(n):
        p = tmp_path / f"p{k}.mkv"
        make_mkv_with_bitmap_subs(p, [(1.0, 2.0, "x")], "pgs", duration_s=200)
        paths.append(p)
    return paths


@pytest.mark.parametrize("slots", [1, 2])
def test_io_limit_per_root(tmp_path, monkeypatch, slots):
    paths = _pgs_files(tmp_path, 4)
    assert len({P.storage_root(p) for p in paths}) == 1
    log, peak = _fake_bitmap_stage(monkeypatch, 0.03, 0.03)
    res = P.analyze_files(
        paths, language="en", vocabulary=set(), cfg=CFG, cache=None, io_concurrency=slots
    )
    assert all(r.error is None and len(r.dialogue.windows) == 3 for r in res)
    assert sum(1 for e in log if e[0] == "decode") == 12  # 3 one-chunk windows per file
    assert peak[0] == slots if slots == 1 else 1 <= peak[0] <= slots


def test_stages_overlap(tmp_path, monkeypatch):
    """Decode and OCR take 0.1 s per chunk: done one after the other, 4 files x 3 chunks take
    2.4 s; pipelined, OCR hides behind the next file's reads (~1.3 s)."""
    paths = _pgs_files(tmp_path, 4)
    log, peak = _fake_bitmap_stage(monkeypatch, 0.1, 0.1)
    t0 = time.monotonic()
    res = P.analyze_files(paths, language="en", vocabulary=set(), cfg=CFG, cache=None)
    wall = time.monotonic() - t0
    assert all(r.error is None for r in res) and peak[0] == 1
    serial = sum(e[3] - e[2] for e in log)
    assert serial >= 2.3
    assert wall < 0.75 * serial
    # some file's OCR ran while a different file was being read
    decodes = [e for e in log if e[0] == "decode"]
    ocrs = [e for e in log if e[0] == "ocr"]
    assert any(o[1] != d[1].name and o[2] < d[3] and d[2] < o[3] for o in ocrs for d in decodes)
    # the windows came out as the fake OCR said, in input order
    for p, r in zip(paths, res, strict=True):
        assert all(c.text.startswith(p.name) for w in r.dialogue.windows for c in w.cues)


def test_empty_input():
    assert P.analyze_files([], language="en", vocabulary=set(), cfg=CFG, cache=None) == []
