import pytest

from showvenist.analysis import ocr as ocr_mod
from showvenist.analysis import windows as W
from showvenist.analysis.cache import AnalysisCache
from showvenist.analysis.types import Window
from showvenist.config import AnalysisConfig
from tests.fixtures.bitmapsubs import make_mkv_with_bitmap_subs
from tests.fixtures.textsubs import make_mkv_with_text_subs

WORDS = "hello there we should leave now the door is open".split()
CUES = [
    (3.0 * i, 3.0 * i + 2.0, f"{WORDS[i % len(WORDS)]} {WORDS[(i + 3) % len(WORDS)]}")
    for i in range(60)
]
CFG = AnalysisConfig(window_min_lines=8, window_max_minutes=1)


@pytest.fixture
def counted(monkeypatch):
    calls = {"text": 0, "ocr": 0}
    rt, ro = W.read_text_cues, W.ocr_cues

    def t(*a, **k):
        calls["text"] += 1
        return rt(*a, **k)

    def o(*a, **k):
        calls["ocr"] += 1
        return ro(*a, **k)

    monkeypatch.setattr(W, "read_text_cues", t)
    monkeypatch.setattr(W, "ocr_cues", o)
    return calls


def test_window_spans():
    s = W.window_spans(1000.0)
    assert [a for a, _ in s] == [200.0, 500.0, 800.0]
    assert s[2][1] == 1000.0 and s[0][1] == 500.0


def test_text_windows_cache_and_full(tmp_path, counted):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, duration_s=200)
    cache = AnalysisCache(tmp_path / "a.db")
    d = W.extract_dialogue(p, "en", set(), CFG, cache)
    assert d.track is not None and len(d.windows) == 3
    assert all(w.lines >= 5 for w in d.windows)
    assert d.ocr_quality == 1.0 and not d.full
    n = counted["text"]
    assert n >= 3
    d2 = W.extract_dialogue(p, "en", set(), CFG, cache)
    assert counted["text"] == n
    assert [w.cues for w in d2.windows] == [w.cues for w in d.windows]
    assert isinstance(d2.windows[0], Window)

    f = W.extract_dialogue(p, "en", set(), CFG, cache, full=True)
    assert f.full and len(f.windows) > 3
    texts = [(c.start_s, c.end_s, c.text) for w in f.windows for c in w.cues]
    assert len(texts) == len(set(texts))
    assert len(texts) >= 55
    n2 = counted["text"]
    W.extract_dialogue(p, "en", set(), CFG, cache, full=True)
    assert counted["text"] == n2
    cache.close()


def test_pgs_windows(tmp_path, counted):
    p = tmp_path / "p.mkv"
    cues = CUES
    make_mkv_with_bitmap_subs(p, cues, "pgs", duration_s=200)
    cache = AnalysisCache(tmp_path / "a.db")
    vocab = set(WORDS)
    d = W.extract_dialogue(
        p, "en", vocab, AnalysisConfig(window_min_lines=3, window_max_minutes=1), cache
    )
    assert len(d.windows) == 3 and all(w.lines >= 3 for w in d.windows)
    assert d.ocr_quality is not None and d.ocr_quality > 0.8
    n = counted["ocr"]
    assert n >= 3
    W.extract_dialogue(
        p, "en", vocab, AnalysisConfig(window_min_lines=3, window_max_minutes=1), cache
    )
    assert counted["ocr"] == n
    cache.close()


def test_no_track(tmp_path):
    p = tmp_path / "n.mkv"
    make_mkv_with_text_subs(p, CUES, lang="fre", duration_s=200)
    d = W.extract_dialogue(p, "en", set(), CFG)
    assert d.track is None and d.windows == [] and d.ocr_quality is None
    assert ocr_mod  # imported for patch parity


# --------------------------------------------------------------------------- phase split


def test_decode_finish_split_matches_default_reader(tmp_path):
    from showvenist.analysis.probe import probe

    t = tmp_path / "t.mkv"
    make_mkv_with_text_subs(t, CUES, duration_s=200)
    track = probe(t).subtitle_tracks[0]
    d = W.decode_chunk(t, track, 30.0, 90.0)
    assert d.error is None and d.events == [] and d.images == []
    assert W.finish_chunk(d) == W._read(t, track, 30.0, 90.0, "en", CFG)

    p = tmp_path / "p.mkv"
    make_mkv_with_bitmap_subs(p, CUES[:20], "pgs", duration_s=70)
    track = probe(p).subtitle_tracks[0]
    d = W.decode_chunk(p, track, 0.0, 30.0)
    assert d.error is None and len(d.images) == 11 and d.cues == []  # 0 s .. 30 s inclusive
    texts = ocr_mod.ocr_images(d.images, "en")
    assert W.finish_chunk(d, texts) == W._read(p, track, 0.0, 30.0, "en", CFG)


def test_decode_chunk_records_errors_and_finish_raises(tmp_path):
    from showvenist.analysis.types import SubtitleReadError, SubtitleTrack

    for track in (SubtitleTrack(1, "subrip", "eng"), SubtitleTrack(1, "hdmv_pgs_subtitle", "eng")):
        d = W.decode_chunk(tmp_path / "missing.mkv", track, 0.0, 60.0)
        assert d.error
        with pytest.raises(SubtitleReadError):
            W.finish_chunk(d, [])


def test_extract_dialogue_uses_injected_reader(tmp_path):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, duration_s=200)
    calls = []

    def reader(path, track, a, b, language, cfg):
        calls.append((a, b))
        return W.finish_chunk(W.decode_chunk(path, track, a, b))

    d = W.extract_dialogue(p, "en", set(), CFG, reader=reader)
    assert calls and len(d.windows) == 3
    plain = W.extract_dialogue(p, "en", set(), CFG)
    assert [w.cues for w in d.windows] == [w.cues for w in plain.windows]
