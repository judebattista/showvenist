"""identify opens the review TUI on a terminal (§4.3, §12.2) with the already-built sessions;
report mode, --plan and --apply are unchanged."""

from __future__ import annotations

from typer.testing import CliRunner

from showvenist import cli
from showvenist.tui import app as tui_app
from tests.test_cli_identify_eval import setup

runner = CliRunner()


def _identify(root, rips, *args):
    return runner.invoke(cli.app, ["identify", str(rips), "--show", "synth-show-2000", "-o",
                                   str(root), *args])


def test_identify_opens_the_tui_on_a_terminal(tmp_path, monkeypatch):
    root, rips = setup(tmp_path, monkeypatch)
    seen = {}

    def fake_run(seasons, opts):
        seen["seasons"], seen["opts"] = seasons, opts
        return 0

    monkeypatch.setattr(tui_app, "run_review", fake_run)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr(cli, "_use_tui", lambda: True)
    r = _identify(root, rips, "--budget", "1.5")
    assert r.exit_code == 0, r.output
    assert "Decision" not in r.stdout  # no report printed
    (season,) = seen["seasons"]
    assert season.title == "Season 1"
    sess = season.session
    assert sess.files == [] and sess.solution is None  # the TUI analyzes in the background
    assert sess.season == 1 and len(sess.season_node.loose) + sum(
        len(d.files) for d in sess.season_node.discs) == 2
    opts = seen["opts"]
    assert opts.budget == 1.5 and opts.llm_mode == "off" and opts.llm_factory is not None
    assert opts.show_title == "Synth Show" and opts.plan_options.exclude == set()
    assert isinstance(season.notices(sess), list)


def test_output_flags_never_open_the_tui(tmp_path, monkeypatch):
    root, rips = setup(tmp_path, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("TUI opened")

    monkeypatch.setattr(tui_app, "run_review", boom)
    monkeypatch.setattr(cli, "_use_tui", lambda: True)
    for args in (["--report"], ["--json"], ["--plan"]):
        monkeypatch.setattr(cli, "_interactive", lambda: True)
        r = _identify(root, rips, *args)
        assert r.exit_code in (0, 2), (args, r.output)
    # without a terminal: the report, as before
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    r = _identify(root, rips)
    assert r.exit_code in (0, 2) and "t01.mkv" in r.stdout
