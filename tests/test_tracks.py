from showvenist.analysis.tracks import select_track
from showvenist.analysis.types import SubtitleTrack as T


def test_select_prefers_most_events_and_language():
    ts = [
        T(1, "subrip", "eng", events=300),
        T(2, "hdmv_pgs_subtitle", "eng", events=900),
        T(3, "hdmv_pgs_subtitle", "fre", events=2000),
    ]
    assert select_track(ts, "en").index == 2


def test_drops_forced_and_sparse():
    ts = [
        T(1, "hdmv_pgs_subtitle", "eng", forced=True, events=900),
        T(2, "hdmv_pgs_subtitle", "eng", events=10),
    ]
    assert select_track(ts, "en") is None
    assert select_track(ts, "en", min_events=5).index == 2


def test_none_cases_and_sdh_ok():
    assert select_track([], "en") is None
    assert select_track([T(1, "subrip", None, events=500)], "en") is None
    assert select_track([T(1, "subrip", "eng", events=500)], "fr") is None
    sdh = T(1, "dvd_subtitle", "eng", sdh=True, events=800)
    assert select_track([sdh, T(2, "subrip", "eng", events=100)], "en") is sdh


def test_tie_prefers_text_and_unreadable_dropped():
    ts = [T(1, "hdmv_pgs_subtitle", "eng", events=100), T(2, "ass", "eng", events=100),
          T(3, "eia_608", "eng", events=5000)]
    assert select_track(ts, "en").index == 2
