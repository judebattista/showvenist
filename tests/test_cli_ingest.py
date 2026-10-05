from typer.testing import CliRunner

from showvenist import cli
from showvenist.bundle.store import load_bundle
from tests.test_tmdb import adapter

runner = CliRunner()


def test_ingest_then_check(tmp_path, monkeypatch):
    ad, _ = adapter()
    monkeypatch.setattr(cli, "_tmdb", lambda: ad)
    r = runner.invoke(cli.app, ["ingest", "Firefly", "--tmdb-id", "1437", "-o", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "show.json" in r.output and "Season" in r.output
    slug = next(tmp_path.iterdir()).name
    assert load_bundle(slug, tmp_path).show.title == "Firefly"
    r = runner.invoke(cli.app, ["bundle", "check", slug, "-o", str(tmp_path)])
    assert r.exit_code == 0 and "Season" in r.output


def test_ingest_without_key_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("SHOWVENIST_TMDB_API_KEY", raising=False)
    r = runner.invoke(cli.app, ["ingest", "X"])
    assert r.exit_code == 1
