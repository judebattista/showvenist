import pytest

from showvenist.analysis.ocr import ocr_cues
from showvenist.analysis.probe import probe
from tests.fixtures.bitmapsubs import make_mkv_with_bitmap_subs

CUES = [(1.0, 3.0, "Hello there"), (5.0, 7.0, "We should leave now")]


@pytest.mark.parametrize("kind", ["pgs", "vobsub"])
def test_ocr_end_to_end_on_generated_disc_subs(tmp_path, kind):
    p = tmp_path / f"{kind}.mkv"
    make_mkv_with_bitmap_subs(p, CUES, kind)
    track = probe(p).subtitle_tracks[0]
    cues = ocr_cues(p, track, 0.0, 10.0, "en")
    assert [c.text.strip() for c in cues] == [c[2] for c in CUES]
