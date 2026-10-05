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
    ts = [
        T(1, "hdmv_pgs_subtitle", "eng", events=100),
        T(2, "ass", "eng", events=100),
        T(3, "eia_608", "eng", events=5000),
    ]
    assert select_track(ts, "en").index == 2


def test_pgs_packets_count_double():
    """Review fix: a PGS cue is two packets (show + clear), so equal cues means text wins."""
    pgs = T(1, "hdmv_pgs_subtitle", "eng", events=800)
    text = T(2, "subrip", "eng", events=400)
    assert select_track([pgs, text], "en") is text
    assert select_track([pgs, T(3, "subrip", "eng", events=300)], "en") is pgs
    assert select_track([T(1, "hdmv_pgs_subtitle", "eng", events=60)], "en") is None  # 30 cues


def test_unknown_counts_are_eligible_after_known():
    from showvenist.analysis.tracks import rank_tracks

    unk_sdh = T(1, "subrip", "eng", sdh=True, events=-1)
    unk = T(2, "mov_text", "eng", events=-1)
    unk_forced = T(3, "subrip", "eng", forced=True, events=-1)
    known = T(4, "subrip", "eng", events=500)
    assert select_track([unk_sdh, unk, unk_forced], "en") is unk  # non-SDH first
    assert rank_tracks([unk_sdh, unk, unk_forced, known], "en") == [known, unk, unk_sdh]
    assert select_track([T(1, "subrip", "fre", events=-1)], "en") is None
