"""Review TUI (spec §12.2, §12.4) driven with Textual's Pilot on a tiny synthetic season.

No real media, no network: synthetic MKVs under tmp_path, XDG_* pointing into tmp_path (the
state.db written by the apply is a temporary one), and a fake LLM client."""

from __future__ import annotations

import asyncio
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

from showvenist.analysis.cache import AnalysisCache
from showvenist.config import Config, HierarchyConfig
from showvenist.hierarchy.parse import build_tree
from showvenist.rename.types import PlanOptions
from showvenist.scoring.llm import LlmScorer
from showvenist.session import Session
from showvenist.state.db import StateDb, state_db_path
from showvenist.tui.app import MainScreen, ReviewApp, ReviewOptions, SeasonInput
from showvenist.tui.dialogs import ConfirmDialog, EpisodeEntry, QuitDialog, two_numbers
from showvenist.tui.evidence import EvidencePane
from showvenist.tui.plan import PlanScreen, SummaryScreen
from showvenist.tui.tree import GLYPH_PINNED, SeasonTree
from tests.test_session import (
    FILLER,
    FakeLlmClient,
    dialogue_for,
    make_episode_mkv,
    synth_bundle,
    write_refs,
)

WIDE = (140, 50)


def filler(seed: int) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choices(FILLER, k=8)) for _ in range(200)]


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """Synth Show/Season 1: Disc 1 = E1 (+ sidecar), E2; Disc 2 = E3, two files with only
    filler dialogue (they need review), and an exact copy of E1. Season 2: one E1."""
    base = tmp_path_factory.mktemp("media")
    bdir = base / "bundle"
    bundle = write_refs(synth_bundle(seasons=2), bdir)
    show = base / "Synth Show"
    s1 = show / "Season 1"
    make_episode_mkv(s1 / "Disc 1" / "t00.mkv", 8, dialogue_for(0))
    make_episode_mkv(s1 / "Disc 1" / "t01.mkv", 11, dialogue_for(1))
    make_episode_mkv(s1 / "Disc 2" / "t00.mkv", 14, dialogue_for(2))
    make_episode_mkv(s1 / "Disc 2" / "t01.mkv", 16, filler(5))
    make_episode_mkv(s1 / "Disc 2" / "t02.mkv", 9, filler(6))
    shutil.copyfile(s1 / "Disc 1" / "t00.mkv", s1 / "Disc 2" / "t03.mkv")
    (s1 / "Disc 1" / "t00.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n")
    make_episode_mkv(show / "Season 2" / "Disc 1" / "t00.mkv", 8, dialogue_for(0, 2))
    then = time.time() - 3600
    for p in show.rglob("*"):
        if p.is_file():
            import os

            os.utime(p, (then, then))
    return bundle, bdir, show


@pytest.fixture
def work(media, tmp_path, monkeypatch):
    """A private copy of the media (the apply moves files) and XDG dirs under tmp_path."""
    for var in ("XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME"):
        monkeypatch.setenv(var, str(tmp_path / var))
    bundle, bdir, show = media
    copy = tmp_path / "Synth Show"
    shutil.copytree(show, copy)
    return bundle, bdir, copy


def make_app(work, *, seasons=(1,), budget=0.0, client=None, history=None,
             plan_kwargs=None) -> ReviewApp:
    bundle, bdir, show = work
    tree = build_tree(show, config=HierarchyConfig(), tag_of=lambda p: None)
    cache = AnalysisCache(show.parent / "analysis.db")
    inputs = []
    for snode in tree.seasons:
        if snode.number not in seasons:
            continue
        sub = replace(tree, seasons=[snode])
        sess = Session(bundle, bdir, Config(), season=snode.number, tree=sub, cache=cache,
                       history=history)
        inputs.append(SeasonInput(f"Season {snode.number}", sess,
                                  lambda s: ["a notice from the CLI"]))
    factory = None
    if client is not None:
        factory = lambda: LlmScorer(client)  # noqa: E731
    opts = ReviewOptions(bundle=bundle, show_title="Synth Show", plan_options=PlanOptions(),
                         history=history, budget=budget, llm_factory=factory,
                         command_args="identify test", plan_kwargs=plan_kwargs or {})
    return ReviewApp(inputs, opts)


async def ready(pilot, app: ReviewApp, timeout: float = 60.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        await pilot.pause(0.05)
        if all(s.status in ("ready", "failed") for s in app.seasons) and app.worker is None:
            await pilot.pause()
            return
    raise AssertionError("analysis did not finish")


async def until(pilot, cond, timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        await pilot.pause(0.05)
        if cond():
            return
    raise AssertionError("condition not reached")


def tree(app) -> SeasonTree:
    return app.main.query_one(SeasonTree)


def names(app) -> list[str]:
    return [f"{r.path.parent.name}/{r.path.name}" for r in tree(app).rows]


def cur(app) -> str:
    r = tree(app).current
    return f"{r.path.parent.name}/{r.path.name}"


def status(app) -> str:
    return str(app.main.query_one("#status").render())


def run(coro):
    return asyncio.run(coro)


REVIEW = {"Disc 2/t01.mkv", "Disc 2/t02.mkv"}


# ---- tests ------------------------------------------------------------------------------------


def test_two_numbers():
    for text in ("S05E03", "5x03", "5-3", "5 3", "season 5 episode 3"):
        assert two_numbers(text) == (5, 3)
    for text in ("", "5", "S01E02E03", "abc"):
        assert two_numbers(text) is None


def test_open_analysis_and_first_solve(work):
    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            assert isinstance(app.screen, MainScreen)
            await ready(pilot, app)
            st = app.seasons[0]
            assert st.status == "ready" and st.done == st.total == 6
            # the default view lists only the files that need review, cursor on the least
            # confident one
            assert set(names(app)) == REVIEW
            low = min((r for r in st.rows if r.in_queue), key=lambda r: r.delta)
            assert tree(app).current.path == low.path
            title = str(app.main.query_one("#title").render())
            assert "Synth Show · Season 1" in title and "2 to review" in title
            ev = app.main.query_one(EvidencePane).text
            assert "Chosen:" in ev and "a)" in ev and "Evidence" in ev and "duration" in ev
            assert "a notice from the CLI" in ev
            # f shows all files; named-free season: the ✓ ones are listed too
            await pilot.press("f")
            assert {"Disc 1/t00.mkv", "Disc 1/t01.mkv", "Disc 2/t00.mkv", "Disc 2/t03.mkv"} \
                <= set(names(app))
            glyphs = {f"{r.path.parent.name}/{r.path.name}": r.glyph for r in tree(app).rows}
            assert glyphs["Disc 1/t00.mkv"] == "✓" and glyphs["Disc 2/t03.mkv"] == "="
            await pilot.press("f")
            assert set(names(app)) == REVIEW
            # n / N walk the review queue
            first = cur(app)
            await pilot.press("n")
            assert cur(app) != first and cur(app) in REVIEW
            await pilot.press("N")
            assert cur(app) == first
    run(go())


def test_pin_resolves_and_jumps(work):
    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            sess = app.seasons[0].session
            first = tree(app).current.path
            await pilot.press("x")  # extra
            assert sess.pins == {first: None}
            assert app.seasons[0].report is not None
            row = next(r for r in app.seasons[0].rows if r.path == first)
            assert row.glyph == GLYPH_PINNED and not row.in_queue
            # the cursor jumped to the other file that needs review
            other = tree(app).current.path
            assert other != first and other.name in {"t01.mkv", "t02.mkv"}
            # a: accept the top candidate of the file under the cursor
            top = sess.explain(other).candidates[0]
            await pilot.press("a")
            assert sess.pins[other] == top.episode_id
            assert "nothing left to review" in status(app)
            # u unpins the file under the cursor (pinned files stay listed)
            tree(app).move_to(other)
            await pilot.press("u")
            assert other not in sess.pins and first in sess.pins
            # Esc undoes the last change (the unpin): the pin is back
            await pilot.press("escape")
            assert sess.pins[other] == top.episode_id
            await pilot.press("escape")  # one level only
            assert "Nothing to undo" in status(app)
            assert sess.pins[other] == top.episode_id
    run(go())


def test_episode_entry(work):
    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            sess = app.seasons[0].session
            path = tree(app).current.path
            await pilot.press("e")
            assert isinstance(app.screen, EpisodeEntry)
            assert "S01 1–4, S02 1–4" in str(app.screen.query_one("#entry-range").render())
            for bad, why in (("hello", "two numbers"), ("S01E09", "not an episode"),
                             ("0 1", "Season 0")):
                inp = app.screen.query_one("#entry-input")
                inp.value = bad
                await pilot.press("enter")
                assert isinstance(app.screen, EpisodeEntry), bad
                assert why in str(app.screen.query_one("#entry-error").render())
            # an episode of the ordering that is not a candidate of this season: the engine's
            # refusal is shown and the entry stays open
            app.screen.query_one("#entry-input").value = "S02E01"
            await pilot.press("enter")
            assert isinstance(app.screen, EpisodeEntry)
            assert "not a candidate" in str(app.screen.query_one("#entry-error").render())
            assert not sess.pins
            app.screen.query_one("#entry-input").value = "1x4"
            await pilot.press("enter")
            assert isinstance(app.screen, MainScreen)
            assert sess.pins == {path: "tmdb:103"}
            # Esc in the entry cancels
            await pilot.press("e")
            await pilot.press("escape")
            assert isinstance(app.screen, MainScreen) and sess.pins == {path: "tmdb:103"}
    run(go())


def test_llm_refused_at_budget_zero_and_sent_within_budget(work):
    async def go():
        client = FakeLlmClient()
        app = make_app(work, client=client)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("l")
            text = status(app)
            assert "Not sending" in text and "--budget 0 " in text and "Re-run with --budget" \
                in text
            assert client.calls == [] and app.worker is None
        client = FakeLlmClient()
        app = make_app(work, client=client, budget=5.0)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            path = tree(app).current.path
            await pilot.press("l")
            await until(pilot, lambda: app.worker is None)
            assert client.calls and app.llm_spent > 0
            assert "LLM answered" in status(app)
            assert app.seasons[0].session.solution.regimes[
                [f.path for f in app.seasons[0].session.files].index(path)] == "llm"
    run(go())


def test_narrow_layout(work):
    async def go():
        app = make_app(work)
        async with app.run_test(size=(80, 40)) as pilot:
            await ready(pilot, app)
            main = app.main.query_one("#main")
            assert main.has_class("narrow")
            t, ev = tree(app).region, app.main.query_one(EvidencePane).region
            assert ev.y > t.y and ev.x == t.x
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            assert not app.main.query_one("#main").has_class("narrow")
            t, ev = tree(app).region, app.main.query_one(EvidencePane).region
            assert ev.x > t.x and ev.y == t.y
    run(go())


def test_quit_with_pending_pins(work):
    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("x")
            await pilot.press("q")
            assert isinstance(app.screen, QuitDialog)
            assert "1 pin" in str(app.screen.query_one("#quit-question").render())
            await pilot.press("s")
            assert isinstance(app.screen, MainScreen) and app.is_running
            await pilot.press("q")
            await pilot.press("a")  # apply: opens the plan
            assert isinstance(app.screen, PlanScreen)
            await pilot.press("escape")
            await pilot.press("q")
            assert "pin" in str(app.screen.query_one("#quit-question").render())
            assert "unapplied plan" in str(app.screen.query_one("#quit-question").render())
            await pilot.press("d")
            await pilot.pause()
            assert not app.is_running
        assert app.return_value == 0
        # nothing pending: q quits at once
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("q")
            await pilot.pause()
            assert not app.is_running
    run(go())


def _opt_ids(screen: PlanScreen) -> list[str]:
    from textual.widgets import OptionList

    return [o.id or "" for o in screen.query_one(OptionList).options]


def _highlight(screen: PlanScreen, oid: str) -> None:
    from textual.widgets import OptionList

    ol = screen.query_one(OptionList)
    ol.highlighted = ol.get_option_index(oid)


def test_plan_untick_confirm_apply_summary(work):
    _, _, show = work
    s1 = show / "Season 1"

    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            sess = app.seasons[0].session
            low = tree(app).current.path
            # override the tool: it says extra (low), the user says E4
            await pilot.press("e")
            app.screen.query_one("#entry-input").value = "S01E04"
            await pilot.press("enter")
            assert sess.pins == {low: "tmdb:103"}
            await pilot.press("p")
            screen = app.screen
            assert isinstance(screen, PlanScreen)
            ids = _opt_ids(screen)
            e1 = s1 / "Disc 1" / "t00.mkv"
            dupe = s1 / "Disc 2" / "t03.mkv"
            assert f"m:{e1}" in ids and f"m:{dupe}" in ids and f"m:{low}" in ids
            footer = str(screen.query_one("#plan-footer").render())
            assert "Plan token" in footer and "1 copies" in footer and "1 sidecars" in footer
            token = screen.token
            # untick the primary: its copy goes too, the plan is rebuilt
            _highlight(screen, f"m:{e1}")
            await pilot.press("space")
            ids = _opt_ids(screen)
            assert f"x:{e1}" in ids and f"m:{e1}" not in ids and f"m:{dupe}" not in ids
            assert {e1, dupe} <= app.exclude and screen.token != token
            assert "0 sidecars" in str(screen.query_one("#plan-footer").render())
            # tick it again: both come back
            _highlight(screen, f"x:{e1}")
            await pilot.press("space")
            ids = _opt_ids(screen)
            assert f"m:{e1}" in ids and f"m:{dupe}" in ids and screen.token == token
            # untick E2: it stays where it is
            e2 = s1 / "Disc 1" / "t01.mkv"
            _highlight(screen, f"m:{e2}")
            await pilot.press("space")
            # Enter: y/n confirm; n moves nothing
            await pilot.press("enter")
            assert isinstance(app.screen, ConfirmDialog)
            await pilot.press("n")
            assert isinstance(app.screen, PlanScreen)
            assert "Nothing was moved" in str(screen.query_one("#plan-status").render())
            assert (s1 / "Disc 1" / "t00.mkv").exists()
            await pilot.press("enter")
            await pilot.press("y")
            await until(pilot, lambda: isinstance(app.screen, SummaryScreen))
            text = app.summary
            assert "Run 1." in text and "showvenist undo --last" in text
            assert "moved 6 file(s), 0 failed" in text
            # the files really moved; E2 (unticked) stayed in its disc folder
            assert sorted(p.relative_to(s1).as_posix() for p in s1.rglob("*")
                          if p.is_file()) == [
                "Disc 1/t01.mkv",
                "Disc 2/synth_show_s01e01_dupe_1.mkv",
                "extras/Disc 2__t02.mkv",  # high-confidence extra once E4 was pinned
                "synth_show_s01e01.en.srt",
                "synth_show_s01e01.mkv",
                "synth_show_s01e03.mkv",
                "synth_show_s01e04.mkv",
            ]
            # one run; the override recorded as `rejected` (decision 61)
            with StateDb.open_default() as db:
                runs = list(db.conn.execute("select id, command from run"))
                assert [tuple(r) for r in runs] == [(1, "identify")]
                rej = list(db.conn.execute(
                    "select episode_id, run_id from match where status='rejected'"))
                assert [tuple(r) for r in rej] == [(None, 1)]  # the tool said extra
                assert list(db.conn.execute(
                    "select count(*) from rename_op where status='done'"))[0][0] == 6
            await pilot.press("enter")
            assert isinstance(app.screen, MainScreen)
            assert "Applied" in status(app)
            # after the apply nothing is pending: q quits without asking
            await pilot.press("q")
            await pilot.pause()
            assert not app.is_running
        assert state_db_path().exists()
    run(go())


def test_two_season_tabs(work):
    async def go():
        app = make_app(work, seasons=(1, 2))
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            assert [s.status for s in app.seasons] == ["ready", "ready"]
            await pilot.press("right")
            await pilot.pause()
            assert app.active == 1
            assert "Season 2" in str(app.main.query_one("#title").render())
            await pilot.press("f")
            assert names(app) == ["Disc 1/t00.mkv"]
            await pilot.press("left")
            await pilot.pause()
            assert app.active == 0
    run(go())


def test_keys_wait_while_a_season_is_analyzed(work):
    """While a season reads, its keys are disabled with the reason in the status line."""
    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            assert app.seasons[0].status in ("queued", "analyzing")
            assert app.blocked(0, "accept")
            assert "still being analyzed" in status(app)
            assert app.pin(0, Path("x"), "extra") is not None
            await ready(pilot, app)
            assert not app.blocked(0, "accept")
    run(go())


def test_cross_filesystem_moves_need_a_second_confirm(work):
    _, _, show = work
    s1 = show / "Season 1"
    # pretend Disc 2 lives on another filesystem than the season folder
    dev = {"device": lambda p: 2 if "Disc 2" in str(p) else 1}

    async def go():
        app = make_app(work, plan_kwargs=dev)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("p")
            screen = app.screen
            assert isinstance(screen, PlanScreen)
            assert any("cross-filesystem" in str(o.prompt) for o in
                       screen.query_one("OptionList").options)
            await pilot.press("enter")
            await pilot.press("y")
            assert isinstance(app.screen, ConfirmDialog)
            assert "cross filesystems" in str(app.screen.query_one("#confirm-question").render())
            await pilot.press("n")  # the rest still moves; the crossing moves fail
            await until(pilot, lambda: isinstance(app.screen, SummaryScreen))
            assert "crosses filesystems: not confirmed" in app.summary
            assert (s1 / "Disc 2" / "t00.mkv").exists()  # E3 stayed
            assert (s1 / "synth_show_s01e01.mkv").exists()  # E1 moved
    run(go())


def test_recovery_runs_first(work):
    from showvenist.state.history import History

    _, _, show = work
    old = show / "Season 1" / "Disc 1" / "t01.mkv"
    with StateDb.open_default() as db:
        run_id = History(db).begin_run("identify --apply", "")
        with db.transaction() as c:
            c.execute("insert into rename_op (run_id, fingerprint, kind, old_path, new_path,"
                      " status, cross_fs) values (?,?,?,?,?,'planned',0)",
                      (run_id.id, "fp", "episode", str(old), str(old.with_name("x.mkv"))))

    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            assert "1 rolled back" in status(app)
            await ready(pilot, app)
    run(go())
    with StateDb.open_default() as db:
        assert list(db.conn.execute("select count(*) from rename_op"))[0][0] == 0


def test_quit_during_analysis_and_help(work):
    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await pilot.press("question_mark")
            from showvenist.tui.dialogs import HelpScreen

            assert isinstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await pilot.press("q")
            await pilot.pause()
            assert app._cancel.is_set() and not app.is_running
    run(go())


# ---- M6 review fixes ----------------------------------------------------------------------------


class SlowLlmClient(FakeLlmClient):
    def parse(self, **kw):
        time.sleep(1.5)
        return super().parse(**kw)


def _spy_session(monkeypatch, app_ref):
    """Record every Session.explain / solve call made on the UI (main) thread with the worker
    that was running at the time."""
    import threading

    seen = []
    orig_ex, orig_solve = Session.explain, Session.solve

    def ex(self, *a, **k):
        if threading.current_thread() is threading.main_thread():
            seen.append(("explain", app_ref[0].worker))
        return orig_ex(self, *a, **k)

    def sv(self, *a, **k):
        if threading.current_thread() is threading.main_thread():
            seen.append(("solve", app_ref[0].worker))
        return orig_solve(self, *a, **k)

    monkeypatch.setattr(Session, "explain", ex)
    monkeypatch.setattr(Session, "solve", sv)
    return seen


def test_ui_never_calls_session_while_the_llm_runs(work, monkeypatch):
    app_ref = [None]
    seen = _spy_session(monkeypatch, app_ref)

    async def go():
        app = make_app(work, client=SlowLlmClient(), budget=5.0)
        app_ref[0] = app
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("f")  # all files: rows never explained before
            await pilot.press("l")
            await pilot.pause(0.05)
            assert app.worker == "llm"
            for k in "jjjjkkjj":
                await pilot.press(k)
                await pilot.pause(0.02)
            assert app.worker == "llm"
            text = app.main.query_one(EvidencePane).text
            assert "Waiting for the LLM call" in text or "Chosen:" in text
            await until(pilot, lambda: app.worker is None)
            assert "Chosen:" in app.main.query_one(EvidencePane).text
    run(go())
    assert not [s for s in seen if s[1] in ("llm", "apply")], seen


def test_plan_screen_stays_and_ui_leaves_session_alone_during_apply(work, monkeypatch):
    from showvenist.rename import movelog

    app_ref = [None]
    seen = _spy_session(monkeypatch, app_ref)
    real = movelog.apply_plan

    def slow(*a, **k):
        time.sleep(2.0)
        return real(*a, **k)

    monkeypatch.setattr(movelog, "apply_plan", slow)

    async def go():
        app = make_app(work)
        app_ref[0] = app
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("x")
            await pilot.press("p")
            await pilot.press("enter")
            await pilot.press("y")
            await pilot.pause(0.05)
            assert app.worker == "apply"
            for k in ("escape", "q", "space", "ctrl+q"):
                await pilot.press(k)
                await pilot.pause(0.02)
                assert isinstance(app.screen, PlanScreen), k
                assert app.is_running
            assert "wait for the summary" in str(app.screen.query_one("#plan-status").render())
            await until(pilot, lambda: isinstance(app.screen, SummaryScreen))
    run(go())
    assert not [s for s in seen if s[1] in ("llm", "apply")], seen


def test_ctrl_q_asks_about_pins_and_cancels_analysis(work):
    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("x")
            await pilot.press("ctrl+q")
            assert isinstance(app.screen, QuitDialog)
            await pilot.press("ctrl+q")  # no second dialog
            assert isinstance(app.screen, QuitDialog)
            await pilot.press("s")
            assert isinstance(app.screen, MainScreen) and app.is_running
        app = make_app(work, seasons=(1, 2))
        async with app.run_test(size=WIDE) as pilot:
            await pilot.press("ctrl+q")  # nothing pending, analysis running: cancelled
            await pilot.pause()
            assert app._cancel.is_set() and not app.is_running
    run(go())


def test_auto_llm_budget_covers_both_seasons(work, monkeypatch):
    """--llm all: each season costs $0.60 against --budget 1. The second season must see the
    first one's spend even before the UI thread has handled its SeasonReady."""
    class Fixed(LlmScorer):
        def estimate_cost(self, files, ctx):
            return 0.6

        def score(self, files, ctx):
            r = super().score(files, ctx)
            self.total_cost_usd = 0.6
            return r

    # warm the analysis cache so the second season is analyzed at once
    async def warm():
        app = make_app(work, seasons=(1, 2))
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
    run(warm())

    real_ready = ReviewApp.on_season_ready

    def late_ready(self, m):
        if m.idx == 0:
            time.sleep(1.0)  # the UI thread is slow: the worker goes on to season 2
        return real_ready(self, m)

    monkeypatch.setattr(ReviewApp, "on_season_ready", late_ready)

    async def go():
        app = make_app(work, seasons=(1, 2), budget=1.0)
        app.opts.llm_mode = "all"
        client = FakeLlmClient()
        app.opts.llm_factory = lambda: Fixed(client)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            assert app.llm_spent == pytest.approx(0.6)
            assert "sent" in " ".join(app.seasons[0].llm_notices)
            assert "Not sending" in " ".join(app.seasons[1].llm_notices)
    run(go())


def test_run_review_returns_1_on_a_crash(work, monkeypatch):
    from showvenist.tui import app as tui_app

    def boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(MainScreen, "on_mount", boom)
    app = make_app(work)
    seasons = [s.inp for s in app.seasons]
    assert tui_app.run_review(seasons, app.opts, headless=True) == 1


def _rejected():
    with StateDb.open_default() as db:
        return [tuple(r) for r in db.conn.execute(
            "select episode_id, run_id from match where status='rejected'")]


def test_overrides_recorded_once_and_only_for_applied_files(work, monkeypatch):
    from showvenist.rename import movelog

    real = movelog.apply_plan
    calls = []

    def refused_twice(*a, **k):
        calls.append(1)
        if len(calls) <= 2:
            raise movelog.PlanRefused("refused for the test")
        return real(*a, **k)

    monkeypatch.setattr(movelog, "apply_plan", refused_twice)

    async def apply(pilot, app):
        await pilot.press("enter")
        await pilot.press("y")
        await until(pilot, lambda: isinstance(app.screen, SummaryScreen))
        text = app.summary
        await pilot.press("enter")
        return text

    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            low = tree(app).current.path
            await pilot.press("e")
            app.screen.query_one("#entry-input").value = "S01E04"  # the tool says extra
            await pilot.press("enter")
            await pilot.press("p")
            _highlight(app.screen, f"m:{low}")
            await pilot.press("space")  # unticked: its override is not recorded
            assert "refused for the test" in await apply(pilot, app)
            assert _rejected() == []
            await pilot.press("p")
            _highlight(app.screen, f"x:{low}")
            await pilot.press("space")  # ticked again: recorded with the (refused) run
            assert "refused for the test" in await apply(pilot, app)
            assert _rejected() == [(None, 1)]
            await pilot.press("p")
            assert "Run 1." in await apply(pilot, app)  # the retry records nothing more
            assert _rejected() == [(None, 1)]
    run(go())


def test_escape_at_cross_filesystem_confirm_cancels_everything(work):
    _, _, show = work
    s1 = show / "Season 1"
    dev = {"device": lambda p: 2 if "Disc 2" in str(p) else 1}

    async def go():
        app = make_app(work, plan_kwargs=dev)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("p")
            await pilot.press("enter")
            await pilot.press("y")
            assert isinstance(app.screen, ConfirmDialog)
            hint = str(app.screen.query_one("#confirm-hint").render())
            assert "n skip those, move the rest" in hint and "Esc cancel" in hint
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, PlanScreen) and app.worker is None
            assert "Nothing was moved" in str(app.screen.query_one("#plan-status").render())
            assert (s1 / "Disc 1" / "t00.mkv").exists()  # same-fs moves did not run either
            assert not app.applied
    run(go())


def test_reopened_plan_reticks_copies_with_their_primary(work):
    _, _, show = work
    s1 = show / "Season 1"
    e1, dupe = s1 / "Disc 1" / "t00.mkv", s1 / "Disc 2" / "t03.mkv"

    async def go():
        app = make_app(work)
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("p")
            _highlight(app.screen, f"m:{e1}")
            await pilot.press("space")
            assert {e1, dupe} <= app.exclude
            await pilot.press("escape")
            await pilot.press("p")
            screen = app.screen
            assert isinstance(screen, PlanScreen)
            _highlight(screen, f"x:{e1}")
            await pilot.press("space")
            ids = _opt_ids(screen)
            assert f"m:{e1}" in ids and f"m:{dupe}" in ids and not app.exclude
    run(go())


def test_escape_undo_switches_to_the_pins_tab(work):
    async def go():
        app = make_app(work, seasons=(1, 2))
        async with app.run_test(size=WIDE) as pilot:
            await ready(pilot, app)
            await pilot.press("x")
            assert app.seasons[0].session.pins
            await pilot.press("right")
            await pilot.pause()
            assert app.active == 1
            await pilot.press("escape")
            await pilot.pause()
            assert app.active == 0 and not app.seasons[0].session.pins
            assert "Season 1" in str(app.main.query_one("#title").render())
            assert "Undid the last change" in status(app)
    run(go())


def test_summary_shows_recovery_and_cancelled():
    from showvenist.rename.types import ApplyResult, RecoveryReport
    from showvenist.tui.plan import summary_text

    rec = RecoveryReport(removed_partials=[Path("/x/a.showvenist-partial")])
    text = summary_text([("Season 1", ApplyResult(run_id=0, recovered=rec, cancelled=True)),
                         ("Season 2", ApplyResult(run_id=3))])
    assert "Season 1: Recovered from an earlier interrupted run" in text
    assert "Season 1: cancelled, nothing moved" in text
    assert "Season 2: moved 0 file(s), 0 failed" in text and "Run 3." in text
