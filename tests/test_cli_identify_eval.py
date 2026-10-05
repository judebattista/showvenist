"""identify and eval through the CLI on tiny synthetic MKVs (no real media, no network)."""

import json

import pytest
from typer.testing import CliRunner

from showvenist import cli
from showvenist.bundle.store import bundle_path, save_bundle
from showvenist.scoring import llm as llm_mod
from tests.test_session import (
    FakeLlmClient,
    dialogue_for,
    make_episode_mkv,
    synth_bundle,
    write_refs,
)

runner = CliRunner()


def setup(tmp_path, monkeypatch):
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    root = tmp_path / "bundles"
    bundle = synth_bundle()
    slug_dir = bundle_path("synth-show-2000", root)
    slug_dir.mkdir(parents=True)
    save_bundle(write_refs(bundle, slug_dir), root)
    rips = tmp_path / "rips" / "Season 1"
    rips.mkdir(parents=True)
    runtimes = [8, 11]
    for k in range(2):
        make_episode_mkv(rips / f"s01e0{k + 1}.mkv", runtimes[k], dialogue_for(k))
    return root, rips


def test_identify_json_and_exit_code(tmp_path, monkeypatch):
    root, rips = setup(tmp_path, monkeypatch)
    r = runner.invoke(cli.app, ["identify", str(rips), "--show", "synth-show-2000", "-o",
                                str(root), "--json"])
    assert r.exit_code in (0, 2), r.output
    report = json.loads(r.stdout[r.stdout.index("{"):])
    assert report["report_version"] == 1 and len(report["files"]) == 2
    got = {
        f["path"].rsplit("/", 1)[-1]: f["decision"]["episode"]["episode"]
        for f in report["files"]
    }
    assert got == {"s01e01.mkv": 1, "s01e02.mkv": 2}


def test_identify_unknown_show_fails(tmp_path, monkeypatch):
    root, rips = setup(tmp_path, monkeypatch)
    r = runner.invoke(cli.app, ["identify", str(rips), "--show", "nope", "-o", str(root)])
    assert r.exit_code == 1


def test_eval_prints_metrics_and_fit(tmp_path, monkeypatch):
    root, rips = setup(tmp_path, monkeypatch)
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "path,show,label,order,disc\n"
        f"{rips}/s01e01.mkv,synth-show-2000,S01E01,aired,\n"
        f"{rips}/s01e02.mkv,synth-show-2000,S01E02,aired,\n"
    )
    r = runner.invoke(cli.app, ["eval", str(manifest), "-o", str(root)])
    assert r.exit_code == 0, r.output
    assert "OVERALL" in r.output and "top-1" in r.output


# ---- review fixes: identify flags ---------------------------------------------------------------


@pytest.fixture
def no_real_llm(monkeypatch):
    """The real client factory must never run in tests."""
    def boom():
        raise AssertionError("real Anthropic client requested")

    monkeypatch.setattr(llm_mod, "default_client_factory", boom)
    monkeypatch.setattr(cli, "LLM_CLIENT_FACTORY", None)


def _ident(root, rips, *extra):
    return runner.invoke(cli.app, ["identify", str(rips), "--show", "synth-show-2000", "-o",
                                   str(root), "--json", *extra])


def _episodes(r):
    report = json.loads(r.stdout[r.stdout.index("{"):])
    return {f["path"].rsplit("/", 1)[-1]: (f["decision"]["episode"] or {}).get("episode")
            for f in report["files"]}


def test_parse_episodes():
    assert cli._parse_episodes("5-8", 2) == {(2, 5), (2, 6), (2, 7), (2, 8)}
    assert cli._parse_episodes("S02E05,s02e09", None) == {(2, 5), (2, 9)}
    assert cli._parse_episodes("3, S01E07", 1) == {(1, 3), (1, 7)}
    for bad, season in (("5-8", None), ("8-5", 1), ("x", 1), (",", 1), ("S2", 1)):
        with pytest.raises(ValueError):
            cli._parse_episodes(bad, season)


def test_identify_episode_ranges_and_errors(tmp_path, monkeypatch, no_real_llm):
    root, rips = setup(tmp_path, monkeypatch)
    r = _ident(root, rips, "--season", "1", "--episodes", "1-2")
    assert r.exit_code in (0, 2), r.output
    assert _episodes(r) == {"s01e01.mkv": 1, "s01e02.mkv": 2}
    r = _ident(root, rips, "--episodes", "S01E01,S01E02")
    assert _episodes(r) == {"s01e01.mkv": 1, "s01e02.mkv": 2}
    for args in (["--episodes", "1-2"], ["--episodes", "zz"],
                 ["--season", "1", "--episodes", "9-12"],  # empty scope: Session ValueError
                 ["--season", "7"]):
        r = _ident(root, rips, *args)
        assert r.exit_code == 1, (args, r.output)
        assert isinstance(r.exception, SystemExit), r.exception  # no traceback


def test_identify_report_and_signals(tmp_path, monkeypatch, no_real_llm):
    root, rips = setup(tmp_path, monkeypatch)
    base = ["identify", str(rips), "--show", "synth-show-2000", "-o", str(root)]
    r = runner.invoke(cli.app, [*base, "--report"])
    assert r.exit_code in (0, 2), r.output
    assert "s01e01.mkv" in r.stdout and not r.stdout.lstrip().startswith("{")
    assert runner.invoke(cli.app, [*base, "--report", "--json"]).exit_code == 1
    r = _ident(root, rips, "--signals", "duration")
    assert r.exit_code in (0, 2), r.output
    report = json.loads(r.stdout[r.stdout.index("{"):])
    used = {sg["name"] for f in report["files"] for sg in f["signals"]}
    assert used == {"duration"}, used
    r = _ident(root, rips, "--signals", "-cues,-guests")
    assert r.exit_code in (0, 2)
    for bad in ("bogus", "duration,-cues", ""):
        assert _ident(root, rips, "--signals", bad).exit_code == 1


def test_identify_llm_needs_yes(tmp_path, monkeypatch, no_real_llm):
    root, rips = setup(tmp_path, monkeypatch)
    # config default is off: no client is ever built
    r = _ident(root, rips)
    assert r.exit_code in (0, 2) and "LLM" not in r.stderr
    # with --llm all but without --yes: the estimate is printed, nothing is sent
    r = _ident(root, rips, "--llm", "all")
    assert r.exit_code == 1 and "estimated $" in r.stderr and "--yes" in r.stderr
    clients = []
    monkeypatch.setattr(cli, "LLM_CLIENT_FACTORY",
                        lambda: clients.append(FakeLlmClient()) or clients[-1])
    r = _ident(root, rips, "--llm", "all")
    assert r.exit_code == 1 and not clients
    r = _ident(root, rips, "--llm", "all", "--yes")
    assert r.exit_code in (0, 2), r.output
    assert len(clients) == 1 and len(clients[0].calls) == 4  # two files, both orders
    assert "LLM spend" in r.stderr
    assert _episodes(r) == {"s01e01.mkv": 1, "s01e02.mkv": 2}
    # --signals without llm turns the LLM off whatever --llm says
    r = _ident(root, rips, "--llm", "all", "--signals", "-llm")
    assert r.exit_code in (0, 2) and len(clients) == 1
    assert _ident(root, rips, "--llm", "maybe").exit_code == 1


# ---- review fixes: eval flags -----------------------------------------------------------


def setup2(tmp_path, monkeypatch):
    """Two seasons, two episodes and an extra each."""
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    root = tmp_path / "bundles"
    slug_dir = bundle_path("synth-show-2000", root)
    slug_dir.mkdir(parents=True)
    save_bundle(write_refs(synth_bundle(seasons=2), slug_dir), root)
    lines = ["path,show,label,order,disc"]
    for s in (1, 2):
        d = tmp_path / "rips" / f"Season {s}"
        for k, minutes in ((0, 8), (1, 11)):
            p = make_episode_mkv(d / f"e{k}.mkv", minutes, dialogue_for(k, s))
            lines.append(f"{p},synth-show-2000,S0{s}E0{k + 1},aired,")
        x = make_episode_mkv(d / "x.mkv", 2, ["behind the scenes with the cast"] * 10)
        lines.append(f"{x},synth-show-2000,extra,aired,")
    manifest = tmp_path / "m.csv"
    manifest.write_text("\n".join(lines) + "\n")
    return root, manifest


def test_eval_ablate_times_and_fit_keeps_llm_weights(tmp_path, monkeypatch, no_real_llm):
    import tomllib

    root, manifest = setup2(tmp_path, monkeypatch)
    cfg = tmp_path / "XDG_CONFIG_HOME" / "showvenist" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("[weights]\nduration = 1.0\n\n[weights.llm]\nllm = 2.5\nrefsubs = 0.5\n")
    r = runner.invoke(cli.app, ["eval", str(manifest), "-o", str(root), "--ablate", "--fit",
                                "--write"])
    assert r.exit_code == 0, r.output
    out = r.stdout
    assert "time per season" in out and "synth-show-2000-S01" in out and "analyze" in out
    assert "without refsubs" in out and "without duration" in out and "all signals" in out
    assert "held-out seasons, base weights" in out and "fitted (season held out)" in out
    assert "left unchanged" in out
    w = tomllib.loads(cfg.read_text())["weights"]
    assert w["llm"] == {"llm": 2.5, "refsubs": 0.5}  # hand-set LLM weights survive the fit
    assert set(w) >= {"duration", "title", "refsubs", "synopsis", "guests", "cues"}


def test_eval_extras_share_and_signals(tmp_path, monkeypatch, no_real_llm):
    from showvenist.eval import fit as fit_mod

    root, manifest = setup2(tmp_path, monkeypatch)
    shares = []
    real = fit_mod.fit_loso

    def spy(items, regime, **kw):
        shares.append(kw.get("extras_share"))
        return real(items, regime, **kw)

    monkeypatch.setattr(fit_mod, "fit_loso", spy)
    args = ["eval", str(manifest), "-o", str(root), "--fit"]
    assert runner.invoke(cli.app, args).exit_code == 0
    assert runner.invoke(cli.app, [*args, "--extras-share", "0.3"]).exit_code == 0
    assert runner.invoke(cli.app, [*args, "--extras-share", "0"]).exit_code == 0
    assert shares == [0.15, 0.3, None]  # π_x by default when the manifest has extras
    assert runner.invoke(cli.app, [*args, "--extras-share", "1.5"]).exit_code == 1
    r = runner.invoke(cli.app, ["eval", str(manifest), "-o", str(root), "--signals",
                                "duration", "--ablate"])
    assert r.exit_code == 0 and "without duration" in r.stdout
    assert "without refsubs" not in r.stdout


def test_eval_llm_guard_and_cost(tmp_path, monkeypatch, no_real_llm):
    root, manifest = setup2(tmp_path, monkeypatch)
    clients = []
    monkeypatch.setattr(cli, "LLM_CLIENT_FACTORY",
                        lambda: clients.append(FakeLlmClient()) or clients[-1])
    r = runner.invoke(cli.app, ["eval", str(manifest), "-o", str(root), "--llm", "all"])
    assert r.exit_code == 1 and "--yes" in r.stderr and "estimated $" in r.stderr
    assert not clients  # the estimate never builds a client
    r = runner.invoke(cli.app, ["eval", str(manifest), "-o", str(root), "--llm", "all",
                                "--yes", "--fit"])
    assert r.exit_code == 0, r.output
    assert sum(len(c.calls) for c in clients) > 0
    assert "LLM cost: $" in r.stdout
