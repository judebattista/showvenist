from showvenist.analysis.types import AnalysisCache, SubtitleTrack, params_hash


def test_track_kind():
    assert SubtitleTrack(1, "hdmv_pgs_subtitle", "eng").kind == "pgs"
    assert SubtitleTrack(2, "dvd_subtitle", "eng").kind == "vobsub"
    assert SubtitleTrack(3, "subrip", "eng").kind == "text"


def test_cache_roundtrip_and_version_invalidation(tmp_path, monkeypatch):
    from showvenist.analysis import types

    c = AnalysisCache(tmp_path / "a.db")
    p = params_hash(start=0)
    c.put("fp", "ocr", p, "{}")
    assert c.get("fp", "ocr", p) == "{}"
    monkeypatch.setitem(types.EXTRACTOR_VERSIONS, "ocr", 2)
    assert c.get("fp", "ocr", p) is None
