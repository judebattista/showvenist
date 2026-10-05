import math

import numpy as np

from showvenist.scoring.duration import DurationScorer
from tests.synth import make_bundle, make_ctx, make_file


def run(minutes, runtimes):
    ctx = make_ctx(make_bundle(len(runtimes), runtimes=runtimes))
    return DurationScorer().score([make_file(minutes=m) for m in minutes], ctx)


def test_flags():
    s = DurationScorer()
    assert s.name == "duration" and s.can_be_negative


def test_match_positive_and_extras_negative():
    r = run([44, 3, 90], [44, 44, 46, 44, 42, 44])
    assert r.available.all()
    assert 2.0 < r.llr[0, 0] < 3.0
    assert r.llr[1].max() < -8 and r.llr[2].max() < -8
    assert np.isfinite(r.llr).all() and np.abs(r.llr).max() <= 50


def test_pal_speed():
    r = run([42.2, 44.0], [44] * 3 + [50, 46, 44])
    assert r.llr[0, 0] > 1.5  # 44*0.959 = 42.196
    assert r.llr[0, 0] > r.llr[1, 3] - 100 and r.llr[0, 3] < r.llr[0, 0]


def test_uniform_runtimes_sigma_4():
    uni = run([50.0], [44] * 4)
    var = run([50.0], [44, 44, 44, 45])
    # 6 min off: sigma 4 is far more forgiving than sigma 2
    assert uni.llr[0, 0] > var.llr[0, 0]
    at_rt = run([44.0], [44] * 4).llr[0, 0]
    uni = run([46.0], [44] * 4)
    assert math.isclose(uni.llr[0, 0], at_rt - 0.5 * (2 / 4) ** 2, abs_tol=0.01)


def test_missing_runtime_uses_median():
    ctx = make_ctx(make_bundle(4, runtimes=[44, 44, 50, 44]))
    ctx.candidates[0].episode.runtime_min = None
    r = DurationScorer().score([make_file(minutes=44)], ctx)
    ref = run([44], [44, 44, 50, 44])
    assert math.isclose(r.llr[0, 0], ref.llr[0, 0])
    assert r.available[0]


def test_no_runtimes_unavailable():
    ctx = make_ctx(make_bundle(3))
    for c in ctx.candidates:
        c.episode.runtime_min = None
    r = DurationScorer().score([make_file(minutes=44)], ctx)
    assert not r.available.any() and (r.llr == 0).all()
