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
    r = run([44, 3, 150], [44, 44, 46, 44, 42, 44])
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
    # a lone file is scored by the accurate/slot mixture: close to the sharp Gaussian shape
    assert math.isclose(uni.llr[0, 0], at_rt - 0.5 * (2 / 4) ** 2, abs_tol=0.15)


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


def test_slot_length_runtimes_are_calibrated_from_the_batch():
    # Hulk-like: TMDB says 60 min (the slot), files are ~48.6 min; the pilot is 95 vs 97.9
    from tests.synth import make_bundle, make_ctx, make_file

    ctx = make_ctx(make_bundle(6, runtimes=[95, 60, 60, 60, 60, 60]))
    files = [make_file(f"t0{i}.mkv", m) for i, m in enumerate([97.9, 48.6, 48.7, 48.6, 49.6, 3.0])]
    llr = DurationScorer().score(files, ctx).llr
    assert llr[0, 0] > 1.5  # pilot matches its own runtime
    assert llr[1, 1] > 1.5 and llr[2, 2] > 1.5  # episode-like files are not rejected
    assert llr[5].max() < -8  # a 3-minute extra still is


def test_single_file_with_slot_length_runtime_is_not_rejected_hard():
    from tests.synth import make_bundle, make_ctx, make_file

    ctx = make_ctx(make_bundle(6, runtimes=[60] * 6))
    llr = DurationScorer().score([make_file("t00.mkv", 48.6)], ctx).llr
    assert llr.min() > -3  # no batch to calibrate on: wide sigma, weak evidence only
    llr3 = DurationScorer().score([make_file("t00.mkv", 3.0)], ctx).llr
    assert llr3.max() < -8


def test_double_length_file_is_not_rejected_as_an_extra():
    # two-part episode muxed as one file (spec §14): ~2 x 44 min must not look like an extra
    ctx = make_ctx(make_bundle(6, runtimes=[44, 44, 45, 44, 44, 44]))
    files = [make_file(f"t0{i}.mkv", m) for i, m in enumerate([88.0, 44.0, 44.5, 43.8])]
    llr = DurationScorer().score(files, ctx).llr
    assert llr[0, 0] > -1  # the double file is near-neutral for the first part
    assert llr[0, 0] < llr[1, 1]  # but a normal match is still worth more
    one = DurationScorer().score([make_file("x.mkv", 3.0)], ctx).llr
    assert one.max() < -8
