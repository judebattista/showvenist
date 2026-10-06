from showvenist.bundle.schema import Bundle, Episode, ShowInfo
from showvenist.config import load_config


def _bundle(title="Ep"):
    return Bundle(show=ShowInfo(title="X"), episodes=[Episode(id="tmdb:1", title=title)])


def test_revision_is_stable_and_content_sensitive():
    a, b = _bundle().with_revision(), _bundle().with_revision()
    assert a.bundle_revision == b.bundle_revision and a.bundle_revision.startswith("b3:")
    assert _bundle("Other").compute_revision() != a.bundle_revision


def test_roundtrip():
    b = _bundle().with_revision()
    assert Bundle.model_validate_json(b.model_dump_json()) == b


def test_config_env_overrides_file(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[tmdb]\napi_key = "file"\n[analysis]\nocr_workers = 2\n')
    cfg = load_config(
        f, env={"SHOWVENIST_TMDB_API_KEY": "env", "SHOWVENIST_ANALYSIS_OCR_WORKERS": "5"}
    )
    assert cfg.tmdb.api_key == "env" and cfg.analysis.ocr_workers == 5
    assert load_config(tmp_path / "none.toml", env={}).llm.mode == "off"


def test_float_settings_can_be_overridden_by_env():
    cfg = load_config(None, env={"SHOWVENIST_BLOCKS_ORDERING_THRESHOLD": "12.5"})
    assert cfg.blocks.ordering_threshold == 12.5
