"""identify and eval through the CLI on tiny synthetic MKVs (no real media, no network)."""

import json

from typer.testing import CliRunner

from showvenist import cli
from showvenist.bundle.store import bundle_path, save_bundle
from tests.test_session import dialogue_for, make_episode_mkv, synth_bundle, write_refs

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
