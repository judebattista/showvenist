# ruff: noqa: F811
"""Consent names what it approves (decision 60): plan tokens for identify --apply and --budget
for paid LLM calls. Synthetic media in tmp_path only; no network."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from showvenist import cli
from showvenist.rename.token import (
    normalize_token,
    plan_token,
    read_last_token,
    save_last_token,
)
from showvenist.rename.types import Plan, PlannedMove
from tests.test_cli_identify_eval import _ident, no_real_llm, setup  # noqa: F401
from tests.test_rename_cli import AFTER, files_under, identify, runner, two_discs  # noqa: F401
from tests.test_rename_plan import snapshot
from tests.test_session import FakeLlmClient


def mv(old="a.mkv", new="b.mkv", fp="f1", kind="rename"):
    return PlannedMove(kind=kind, old=Path(old), new=Path(new), fingerprint=fp, show_slug="s")


def plan(*moves):
    return Plan(show_slug="s", naming_order="aired", moves=list(moves))


def test_token_form_and_stability():
    t = plan_token([plan(mv(), mv("c", "d", "f2"))])
    assert re.fullmatch(r"[0-9a-f]{4}-[0-9a-f]{4}", t)
    assert t == plan_token([plan(mv(), mv("c", "d", "f2"))])


def test_token_sensitive_to_order_and_content():
    a, b = mv(), mv("c", "d", "f2")
    base = plan_token([plan(a, b)])
    assert plan_token([plan(b, a)]) != base
    assert plan_token([plan(mv(new="x.mkv"), b)]) != base
    assert plan_token([plan(mv(old="x.mkv"), b)]) != base
    assert plan_token([plan(mv(fp="f9"), b)]) != base
    assert plan_token([plan(mv(kind="move"), b)]) != base
    assert plan_token([plan(a, b)]) != plan_token([plan(a)])


def test_normalize_token():
    assert normalize_token("7F3A-9c1e") == "7f3a9c1e"
    assert normalize_token(" 7f3a 9c1e ") == "7f3a9c1e"
    for bad in ("", "7f3a", "7f3a-9c1g", "7f3a-9c1e0"):
        with pytest.raises(ValueError):
            normalize_token(bad)


def test_last_token_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert read_last_token() is None
    save_last_token("7f3a-9c1e")
    assert read_last_token() == "7f3a-9c1e"
    save_last_token("0000-ffff")
    assert read_last_token() == "0000-ffff"


def plan_token_of(root, show):
    r = identify(root, show, "--plan")
    assert r.exit_code == 0, r.output
    m = re.search(r"Plan token: ([0-9a-f]{4}-[0-9a-f]{4})  \(apply with --apply --confirm \1, "
                  r"or --confirm last\)", r.stdout)
    assert m, r.stdout
    return m.group(1)


def test_plan_prints_and_saves_token(two_discs):
    root, show, s = two_discs
    t = plan_token_of(root, show)
    assert read_last_token() == t


def test_plan_without_moves_prints_no_token(two_discs):
    root, show, s = two_discs
    assert identify(root, show, "--apply", "--confirm", plan_token_of(root, show)).exit_code == 0
    r = identify(root, show, "--plan")
    assert "No moves." in r.stdout and "Plan token" not in r.stdout


def test_apply_with_token(two_discs):
    root, show, s = two_discs
    t = plan_token_of(root, show)
    r = identify(root, show, "--apply", "--confirm", t.upper().replace("-", ""))
    assert r.exit_code == 0, r.output
    assert "Moved 7 file(s)" in r.stdout
    assert files_under(s) == AFTER


def test_apply_with_wrong_token_moves_nothing(two_discs):
    root, show, s = two_discs
    t = plan_token_of(root, show)
    wrong = "0000-0000" if t != "0000-0000" else "0000-0001"
    before = snapshot(show)
    r = identify(root, show, "--apply", "--confirm", wrong)
    assert r.exit_code == 1
    assert f"the plan changed since the token was made (token now {t})" in r.output
    assert snapshot(show) == before
    assert identify(root, show, "--apply", "--confirm", "nonsense").exit_code == 1
    assert snapshot(show) == before


def test_confirm_last(two_discs):
    root, show, s = two_discs
    plan_token_of(root, show)
    r = identify(root, show, "--apply", "--confirm", "last")
    assert r.exit_code == 0, r.output
    assert files_under(s) == AFTER


def test_confirm_last_without_a_saved_plan(two_discs):
    root, show, s = two_discs
    before = snapshot(show)
    r = identify(root, show, "--apply", "--confirm", "last")
    assert r.exit_code == 1 and "no saved plan token: run --plan first" in r.output
    assert snapshot(show) == before


def test_stale_last_refuses_after_the_plan_changed(two_discs):
    root, show, s = two_discs
    plan_token_of(root, show)
    (s / "Disc 2" / "t03.mkv").unlink()  # one fewer copy: a different plan
    before = snapshot(show)
    r = identify(root, show, "--apply", "--confirm", "last")
    assert r.exit_code == 1 and "the plan changed" in r.output
    assert snapshot(show) == before


def test_non_tty_without_confirm_refuses(two_discs):
    root, show, s = two_discs
    before = snapshot(show)
    r = identify(root, show, "--apply")
    assert r.exit_code == 1 and "no TTY" in r.output and "--confirm" in r.output
    assert snapshot(show) == before


def test_confirm_without_apply_errors(two_discs):
    root, show, s = two_discs
    for flag in ("--plan", "--report"):
        r = identify(root, show, flag, "--confirm", "last")
        assert r.exit_code == 1 and "--confirm only goes with --apply" in r.output
    r = identify(root, show, "--confirm", "last")
    assert r.exit_code == 1


def test_yes_is_gone_from_identify_and_eval(two_discs):
    root, show, s = two_discs
    assert identify(root, show, "--yes").exit_code == 2
    r = runner.invoke(cli.app, ["eval", "m.csv", "--yes"])
    assert r.exit_code == 2


def test_budget_zero_refuses_and_names_a_budget(tmp_path, monkeypatch, no_real_llm):  # noqa: F811
    root, rips = setup(tmp_path, monkeypatch)
    clients = []
    monkeypatch.setattr(cli, "LLM_CLIENT_FACTORY",
                        lambda: clients.append(FakeLlmClient()) or clients[-1])
    r = _ident(root, rips, "--llm", "all")
    assert r.exit_code == 1 and not clients
    m = re.search(r"Not sending: estimated \$(\d+\.\d{4}) exceeds --budget 0\. "
                  r"Re-run with --budget (\d+\.\d\d) to allow these paid API calls\.", r.stderr)
    assert m, r.stderr
    assert float(m.group(2)) >= float(m.group(1))
    # a budget below the estimate still refuses; the suggested one proceeds
    assert _ident(root, rips, "--llm", "all", "--budget", "0.0000001").exit_code == 1
    assert not clients
    r = _ident(root, rips, "--llm", "all", "--budget", m.group(2))
    assert r.exit_code in (0, 2), r.output
    assert len(clients) == 1 and "LLM spend" in r.stderr


def test_over_budget_rounds_the_suggestion_up():
    msg = cli._over_budget(0.4201, 0)
    assert "estimated $0.4201 exceeds --budget 0." in msg and "--budget 0.43 " in msg
    assert "--budget 0.42 " in cli._over_budget(0.42, 0.1)


# ---- --budget covers the whole invocation (M6 review) ------------------------------------------


@pytest.fixture
def fixed_cost(monkeypatch):
    """Every season's LLM pass is estimated at, and costs, $0.60 (the fake client's own usage
    is tiny, which would hide a per-season check)."""
    from showvenist.scoring.llm import LlmScorer

    real_score = LlmScorer.score

    def score(self, files, ctx):
        r = real_score(self, files, ctx)
        self.total_cost_usd = 0.6
        return r

    monkeypatch.setattr(LlmScorer, "estimate_cost", lambda self, files, ctx: 0.6)
    monkeypatch.setattr(LlmScorer, "score", score)
    clients = []
    monkeypatch.setattr(cli, "LLM_CLIENT_FACTORY",
                        lambda: clients.append(FakeLlmClient()) or clients[-1])
    return clients


def _two_season_rips(tmp_path, monkeypatch):
    from showvenist.bundle.store import bundle_path, save_bundle
    from tests.test_cli_identify_eval import age
    from tests.test_session import dialogue_for, make_episode_mkv, synth_bundle, write_refs

    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    monkeypatch.setattr(cli, "READINESS_SETTLE_SECONDS", 0.0)
    root = tmp_path / "bundles"
    slug_dir = bundle_path("synth-show-2000", root)
    slug_dir.mkdir(parents=True)
    save_bundle(write_refs(synth_bundle(seasons=2), slug_dir), root)
    show = tmp_path / "rips" / "Synth Show"
    lines = ["path,show,label,order,disc"]
    for s in (1, 2):
        for k, minutes in ((0, 8), (1, 11)):
            p = make_episode_mkv(show / f"Season {s}" / f"t0{k}.mkv", minutes,
                                 dialogue_for(k, s))
            age(p)
            lines.append(f"{p},synth-show-2000,S0{s}E0{k + 1},aired,")
    manifest = tmp_path / "m.csv"
    manifest.write_text("\n".join(lines) + "\n")
    return root, show, manifest


def test_identify_budget_is_cumulative_across_seasons(tmp_path, monkeypatch, no_real_llm,
                                                      fixed_cost):
    root, show, _ = _two_season_rips(tmp_path, monkeypatch)
    r = runner.invoke(cli.app, ["identify", str(show), "--show", "synth-show-2000", "-o",
                                str(root), "--json", "--llm", "all", "--budget", "1"])
    # season 1 fits ($0.60 of $1); season 2's $0.60 does not fit the $0.40 left
    assert r.exit_code == 1, r.output
    assert r.stderr.count("LLM spend") == 1
    assert "exceeds the $0.4000 that --budget 1 leaves after $0.6000 already spent" in r.stderr
    assert "Re-run with --budget 1.20 " in r.stderr
    assert len(fixed_cost) == 1
    r = runner.invoke(cli.app, ["identify", str(show), "--show", "synth-show-2000", "-o",
                                str(root), "--json", "--llm", "all", "--budget", "1.2"])
    assert r.exit_code in (0, 2), r.output
    assert r.stderr.count("LLM spend") == 2


def test_eval_budget_is_cumulative_across_seasons(tmp_path, monkeypatch, no_real_llm,
                                                  fixed_cost):
    root, _, manifest = _two_season_rips(tmp_path, monkeypatch)
    r = runner.invoke(cli.app, ["eval", str(manifest), "-o", str(root), "--llm", "all",
                                "--budget", "1"])
    assert r.exit_code == 1, r.output
    assert "LLM not run for 1 season(s)" in r.stderr
    assert "exceeds the $0.4000 that --budget 1 leaves after $0.6000 already spent" in r.stderr
    assert "--budget 1.20 " in r.stderr
    assert len(fixed_cost) == 1
