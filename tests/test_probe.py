from fractions import Fraction
from pathlib import Path

import av

from showvenist.analysis.probe import probe

ASS_HEADER = (
    b"[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize\n"
    b"Style: Default,Arial,20\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, "
    b"MarginR, MarginV, Effect, Text\n"
)


def make_mkv(path: Path, cues, seconds: int = 4, tracks=(("eng", "", False),), title=None):
    """Tiny MKV: black video + one ASS stream per entry in `tracks` (lang, title, forced).
    PyAV can mux ASS but not subrip, so cues are written as ASS events."""
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
    assert (a.lang, a.events, a.sdh, a.forced, a.kind) == ("eng", 7, True, False, "text")
    assert (b.lang, b.events, b.forced) == ("fre", 7, True)


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
