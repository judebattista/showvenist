"""Tests for the CLI."""

from typer.testing import CliRunner

from showvenist.cli import app

runner = CliRunner()


def test_version():
    """Test that --version prints the version and exits with 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "showvenist" in result.stdout
