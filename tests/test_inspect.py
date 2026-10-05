from showvenist.analysis.inspect import inspect_file
from showvenist.analysis.types import AnalysisCache
from showvenist.config import AnalysisConfig
from tests.fixtures.textsubs import make_mkv_with_text_subs

CUES = [(3.0 * i, 3.0 * i + 2.0, f"line number {i}") for i in range(60)]
CFG = AnalysisConfig(window_min_lines=8, window_max_minutes=1)


def test_inspect_report(tmp_path):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, title="My Title", duration_s=200)
    cache = AnalysisCache(tmp_path / "a.db")
    r = inspect_file(p, "en", set(), CFG, cache)
    for s in (
        "My Title",
        "fingerprint:",
        "content key:",
        "lines",
        "ocr quality:  1.00",
        "* #",
        "line number",
    ):
        assert s in r
    cache.close()


def test_inspect_no_track(tmp_path):
    p = tmp_path / "t.mkv"
    make_mkv_with_text_subs(p, CUES, lang="fre", duration_s=200)
    r = inspect_file(p, "en", set(), CFG, None)
    assert "no usable" in r and "* #" not in r
