from fractions import Fraction
from pathlib import Path

import av
import pytest

from showvenist.analysis.probe import probe
from showvenist.analysis.types import UNKNOWN_EVENTS, AnalysisCache, SubtitleTrack

ASS_HEADER = (
    b"[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize\n"
    b"Style: Default,Arial,20\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, "
    b"MarginR, MarginV, Effect, Text\n"
)


def make_mkv(
    path: Path, cues, seconds: int = 4, tracks=(("eng", "", False),), title=None, frames_tag=False
):
    """Tiny MKV: black video + one ASS stream per entry in `tracks` (lang, title, forced).
    PyAV can mux ASS but not subrip, so cues are written as ASS events. `frames_tag` writes
    MakeMKV's NUMBER_OF_FRAMES-eng statistics tag with the cue count."""
    out = av.open(str(path), "w", format="matroska")
    if title:
        out.metadata["title"] = title
    v = out.add_stream("mpeg4", rate=1)
    v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
    subs = []
    for lang, stitle, forced in tracks:
        st = out.add_stream("ass")
        st.codec_context.extradata = ASS_HEADER
        st.metadata["language"] = lang
        if stitle:
            st.metadata["title"] = stitle
        if forced:
            st.disposition = av.stream.Disposition.forced
        if frames_tag:
            st.metadata["NUMBER_OF_FRAMES-eng"] = str(len(cues))
        st.time_base = Fraction(1, 1000)
        subs.append(st)
    for i in range(seconds):
        f = av.VideoFrame(64, 48, "yuv420p")
        f.pts = i
        for pk in v.encode(f):
            out.mux(pk)
    for pk in v.encode():
        out.mux(pk)
    for n, (a, b, text) in enumerate(cues):
        for st in subs:
            pk = av.Packet(f"{n},0,Default,,0,0,0,,{text}".encode())
            pk.stream, pk.pts, pk.dts = st, int(a * 1000), int(a * 1000)
            pk.duration = int((b - a) * 1000)
            pk.time_base = Fraction(1, 1000)
            out.mux(pk)
    out.close()


def test_probe_basic(tmp_path):
    p = tmp_path / "a.mkv"
    cues = [(i * 0.5, i * 0.5 + 0.4, f"Line {i}") for i in range(7)]
    make_mkv(p, cues, title="Pilot", tracks=(("eng", "SDH", False), ("fre", "Forced", True)))
    pr = probe(p)
    assert pr.container == "matroska"
    assert pr.title_tag == "Pilot"
    assert pr.size == p.stat().st_size
    assert pr.duration_s > 1
    assert pr.video_codec == "mpeg4" and (pr.width, pr.height) == (64, 48)
    assert pr.chapters == 0
    a, b = pr.subtitle_tracks
    # No NUMBER_OF_FRAMES tag (ffmpeg-written MKV): count unknown, not demuxed.
    assert (a.lang, a.events, a.sdh, a.forced, a.kind) == ("eng", -1, True, False, "text")
    assert (b.lang, b.events, b.forced) == ("fre", -1, True)


def test_probe_no_title_no_subs(tmp_path):
    p = tmp_path / "b.mkv"
    make_mkv(p, [], tracks=())
    pr = probe(p)
    assert pr.title_tag is None
    assert pr.subtitle_tracks == []


def test_tag_lookup_accepts_makemkv_language_suffix():
    from showvenist.analysis.probe import _tag

    assert _tag({"NUMBER_OF_FRAMES-eng": "1096"}, "NUMBER_OF_FRAMES") == "1096"
    assert _tag({"NUMBER_OF_FRAMES": "5"}, "NUMBER_OF_FRAMES") == "5"
    assert _tag({"NUMBER_OF_BYTES-eng": "5"}, "NUMBER_OF_FRAMES") is None


class _CountingContainer:
    """Proxy for an av InputContainer that counts demuxed packets."""

    def __init__(self, inner, counter):
        self._inner, self._counter = inner, counter

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._inner.close()

    def demux(self, *args, **kwargs):
        for pkt in self._inner.demux(*args, **kwargs):
            self._counter["packets"] += 1
            yield pkt


@pytest.fixture
def demux_counter(monkeypatch):
    import showvenist.analysis.probe as probe_mod

    counter = {"packets": 0, "opens": 0}
    real_open = av.open

    def counting_open(*args, **kwargs):
        if "w" in args[1:2] or kwargs.get("mode") == "w":  # fixture writes
            return real_open(*args, **kwargs)
        counter["opens"] += 1
        return _CountingContainer(real_open(*args, **kwargs), counter)

    monkeypatch.setattr(probe_mod.av, "open", counting_open)
    return counter


def test_probe_uses_frame_tags_without_demuxing(tmp_path, demux_counter):
    p = tmp_path / "tagged.mkv"
    make_mkv(p, [(i * 0.5, i * 0.5 + 0.4, f"Line {i}") for i in range(7)], frames_tag=True)
    pr = probe(p)
    assert demux_counter["opens"] == 1
    assert demux_counter["packets"] == 0
    assert pr.subtitle_tracks[0].events == 7


def test_probe_untagged_is_unknown_without_demuxing(tmp_path, demux_counter):
    """Review fix: an untagged track used to be counted by demuxing the whole file."""
    p = tmp_path / "untagged.mkv"
    make_mkv(p, [(i * 0.5, i * 0.5 + 0.4, f"Line {i}") for i in range(7)])
    pr = probe(p)
    assert demux_counter["packets"] == 0
    assert pr.subtitle_tracks[0].events == UNKNOWN_EVENTS == -1


def test_probe_cached_by_fingerprint(tmp_path, demux_counter):
    p = tmp_path / "c.mkv"
    make_mkv(p, [(0.5, 1.0, "Hi")], title="Pilot", frames_tag=True)
    cache = AnalysisCache(tmp_path / "a.db")
    first = probe(p, cache)
    assert demux_counter["opens"] == 1
    again = probe(p, cache)
    assert demux_counter["opens"] == 1  # served from analysis.db
    assert again == first and isinstance(again.subtitle_tracks[0], SubtitleTrack)
    assert again == probe(p)  # uncached probe agrees
    cache.close()
