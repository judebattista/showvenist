import random

import numpy as np

from showvenist.bundle.schema import SubtitleRef
from showvenist.scoring.refsubs import RefsubsScorer, cross_check, dialogue_factor, ocr_factor
from tests.synth import make_bundle, make_ctx, make_file

VOCAB = [f"w{n}x" for n in range(3000)]
COMMON = ["the", "theme", "song", "plays", "all", "night", "long", "again"]


def gen_lines(seed: int, n: int = 120, common: bool = False) -> list[str]:
    rng = random.Random(seed)
    lines = [" ".join(rng.choice(VOCAB) for _ in range(8)) for _ in range(n)]
    if common:
        lines = [" ".join(COMMON)] + lines
    return lines


def write_srt(path, lines):
    parts = []
    for k, t in enumerate(lines):
        parts.append(f"{k + 1}\n00:{k // 60:02d}:{k % 60:02d},000 --> 00:{k // 60:02d}:"
                     f"{k % 60:02d},900\n{t}\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def setup(tmp_path, n=4, skip=(), lang="en", common=False):
    b = make_bundle(n)
    refs = {}
    for k, ep in enumerate(b.episodes):
        lines = gen_lines(k, common=common)
        refs[k] = lines
        if k in skip:
            continue
        p = tmp_path / "subs" / f"{k}.srt"
        p.parent.mkdir(exist_ok=True)
        write_srt(p, lines)
        ep.subtitles.append(SubtitleRef(lang=lang, path=f"subs/{k}.srt", source="localsubs"))
    return make_ctx(b, bundle_dir=tmp_path), refs


def test_matching_file_ranks_first(tmp_path):
    ctx, refs = setup(tmp_path)
    f = make_file(lines=refs[2])
    r = RefsubsScorer().score([f], ctx)
    assert r.available[0]
    assert int(np.argmax(r.llr[0])) == 2
    assert r.llr[0, 2] > 10
    assert all(r.llr[0, j] < 0 for j in (0, 1, 3))
    assert r.notes[(0, 2)]["containment"] > 0.9
    assert r.notes[(0, 2)]["top_shingles"]
    assert r.window_llr.shape == (1, 1, 4)
    assert r.window_llr[0, 0, 2] > 10


def test_bad_ocr_suppresses_negatives_only(tmp_path):
    ctx, refs = setup(tmp_path)
    good = make_file(lines=refs[1], ocr_quality=1.0)
    bad = make_file("b.mkv", lines=refs[1], ocr_quality=0.5)
    unrelated_bad = make_file("c.mkv", lines=gen_lines(99), ocr_quality=0.5)
    r = RefsubsScorer().score([good, bad, unrelated_bad], ctx)
    assert r.llr[1, 1] == r.llr[0, 1] > 10
    assert (r.llr[0] [[0, 2, 3]] < 0).all()
    assert (r.llr[1][[0, 2, 3]] == 0).all()
    assert (r.llr[2] == 0).all()
    assert r.available[2]


def test_unrelated_is_negative(tmp_path):
    ctx, _ = setup(tmp_path)
    r = RefsubsScorer().score([make_file(lines=gen_lines(500))], ctx)
    assert (r.llr[0] < -3).all()


def test_missing_and_other_language(tmp_path):
    ctx, refs = setup(tmp_path, skip=(3,))
    r = RefsubsScorer().score([make_file(lines=refs[0])], ctx)
    assert r.llr[0, 3] == 0 and r.llr[0, 0] > 10
    (tmp_path / "x").mkdir()
    ctx2, refs2 = setup(tmp_path / "x", lang="fr")
    r2 = RefsubsScorer().score([make_file(lines=refs2[0])], ctx2)
    assert not r2.available[0] and (r2.llr == 0).all()
    r3 = RefsubsScorer().score([make_file(lines=None)], ctx)
    assert not r3.available[0] and (r3.llr == 0).all()


def test_639_2_tag_accepted(tmp_path):
    ctx, refs = setup(tmp_path, lang="eng")
    r = RefsubsScorer().score([make_file(lines=refs[0])], ctx)
    assert r.llr[0, 0] > 10


def test_little_dialogue_scales_down(tmp_path):
    ctx, refs = setup(tmp_path)
    r = RefsubsScorer().score([make_file(lines=refs[0][:15])], ctx)
    assert 0 < r.llr[0, 0] < 6
    assert dialogue_factor(0) == 0 and dialogue_factor(300) == 1 and dialogue_factor(900) == 1
    assert ocr_factor(None) == 1 and ocr_factor(0.7) == 0 and ocr_factor(0.9) == 1


def test_common_line_is_discounted(tmp_path):
    ctx, refs = setup(tmp_path, common=True)
    # file with only the shared theme line and nothing else matching
    lines = [" ".join(COMMON)] * 1 + gen_lines(777, 120)
    r = RefsubsScorer().score([make_file(lines=lines)], ctx)
    assert (r.llr[0] < -3).all()
    assert r.notes[(0, 0)]["containment"] < 0.02


def test_cross_check_finds_duplicate(tmp_path):
    ctx, refs = setup(tmp_path)
    assert cross_check(ctx) == []
    write_srt(tmp_path / "subs" / "3.srt", refs[1])
    pairs = cross_check(ctx)
    assert len(pairs) == 1
    a, b, c = pairs[0]
    assert {a, b} == {ctx.candidates[1].id, ctx.candidates[3].id} and c > 0.9
