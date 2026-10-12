import pytest


@pytest.fixture(autouse=True)
def _no_secondary_sources(monkeypatch):
    """`ingest` fetches tmdb,tvmaze by default; tests must never reach the network, so they
    start with TMDB only and opt into TVmaze with --source and a fake adapter."""
    monkeypatch.setenv("SHOWVENIST_INGEST_SOURCES", "tmdb")
