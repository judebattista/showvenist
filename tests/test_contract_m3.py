import numpy as np

from showvenist.config import load_config
from showvenist.scoring.candidates import build_candidates
from showvenist.scoring.types import (
    DEFAULT_CLAMPS,
    REGIME_SCORERS,
    ScoreResult,
    level_of,
)
from tests.synth import make_bundle, make_ctx, make_file


def test_regime_scorers_have_defaults_and_llm_replaces_dialogue_scorers():
    for names in REGIME_SCORERS.values():
        assert set(names) <= set(DEFAULT_CLAMPS)
    assert not {"synopsis", "guests", "cues"} & set(REGIME_SCORERS["llm"])


def test_levels():
    assert level_of(4.6) == "high" and level_of(4.59) == "medium"
    assert level_of(2.2) == "medium" and level_of(2.19) == "low"


def test_candidates_filters():
    b = make_bundle(5)
    assert [c.label for c in build_candidates(b)] == [f"S01E0{i}" for i in range(1, 6)]
    assert len(build_candidates(b, season=2)) == 0
    assert [c.number for c in build_candidates(b, episodes={(1, 2), (1, 4)})] == [2, 4]
    b.episodes[0].excluded = True
    assert len(build_candidates(b)) == 4 and build_candidates(b)[0].index == 0


def test_synth_helpers_and_empty_result():
    ctx = make_ctx()
    f = make_file(lines=["hello there", "general kenobi"])
    assert ctx.n_candidates == 6 and f.dialogue.windows[0].lines == 2
    r = ScoreResult.empty(2, 6)
    assert r.llr.shape == (2, 6) and not r.available.any() and np.all(r.llr == 0)


def test_weights_config_nesting(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[weights]\nduration = 0.7\n[weights.llm]\nllm = 1.4\n")
    cfg = load_config(p, env={})
    assert cfg.weights.duration == 0.7 and cfg.weights.llm.llm == 1.4
    assert cfg.weights.synopsis == 1.0
