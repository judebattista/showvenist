"""The review TUI (spec §12.2, §12.4, decision 61): App, season tabs and workers.

Threads: `analyze` (every season, one after another, in tree order), `ask_llm` and `apply`
run as thread workers, one at a time. The UI never calls a Session while a worker is using
it: a season that is queued or reading is not touched at all, and while the LLM or an apply
runs every key that would change a session is disabled (the reason is in the status line
above the footer). `pin`, `unpin`, `solve` and `plan` are fast and run inline. Workers talk to
the UI only through posted messages.

No matching or planning logic lives here: the app calls `Session`, `build_plan`,
`apply_plan`, `recover`, `record_overrides` and `History`.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, OptionList, ProgressBar, Static, Tab, Tabs
from textual.worker import get_current_worker

from showvenist.bundle.schema import Bundle
from showvenist.rename.types import ApplyResult, Plan, PlanOptions
from showvenist.report_schema import Report
from showvenist.session import Session
from showvenist.state.db import StateDb
from showvenist.state.history import History
from showvenist.tui.dialogs import (
    EpisodeEntry,
    HelpScreen,
    QuitDialog,
    episode_in_ordering,
    ordering_range,
    two_numbers,
)
from showvenist.tui.evidence import EvidencePane, render_detail
from showvenist.tui.plan import Plans, PlanScreen, SummaryScreen, summary_text
from showvenist.tui.tree import (
    NOT_READY,
    READY,
    UNREADABLE,
    Row,
    SeasonTree,
    build_rows,
    least_confident,
    progress_rows,
)

NARROW = 100  # below this many columns the evidence pane moves under the tree


@dataclass
class SeasonInput:
    """One season tab: its (not yet analyzed) Session and the notices the CLI adds."""

    title: str
    session: Session
    notices: Callable[[Session], list[str]] = lambda s: []


@dataclass
class ReviewOptions:
    bundle: Bundle
    show_title: str
    plan_options: PlanOptions
    history: History | None = None  # read-only history used by the sessions and build_plan
    budget: float = 0.0  # --budget USD (decision 60)
    llm_mode: str = "off"  # --llm: auto / all run after each season's first solve
    llm_factory: Callable[[], Any] | None = None  # makes an LlmScorer
    command_args: str = ""  # recorded with the run
    open_state_db: Callable[[], StateDb] = StateDb.open_default
    state_db_exists: Callable[[], bool] | None = None
    plan_kwargs: dict[str, Any] = field(default_factory=dict)  # build_plan's injectables (tests)


@dataclass
class SeasonState:
    inp: SeasonInput
    status: str = "queued"  # queued | analyzing | ready | failed | applied
    done: int = 0
    total: int = 0
    files: dict[Path, str] = field(default_factory=dict)  # analysis state per file
    report: Report | None = None
    rows: list[Row] = field(default_factory=list)
    cursor: Path | None = None
    pre_pin: dict[Path, str] = field(default_factory=dict)  # level before the user's pin
    error: str | None = None
    llm_notices: list[str] = field(default_factory=list)
    # copies of Session state the UI reads while a worker may be using the Session
    pinned: set[Path] = field(default_factory=set)
    details: dict[Path, Any] = field(default_factory=dict)  # explain() results of this solve
    # read once, before any worker runs: the UI never reads the Session while it is analyzed
    season_node: Any = None
    ignored: list[Path] = field(default_factory=list)
    not_ready: set[Path] = field(default_factory=set)
    ordering: str = ""

    def __post_init__(self) -> None:
        sess = self.inp.session
        self.season_node = sess.season_node
        self.ordering = sess.ordering
        node = self.season_node
        if node is not None:
            files = [f for d in node.discs for f in d.files] + list(node.loose)
            self.total = len(files)
            self.not_ready = {Path(f.path) for f in files if not f.ready}
            if sess.tree is not None and node.folder is not None:
                self.ignored = [p for p in sess.tree.ignored if node.folder in p.parents]

    @property
    def session(self) -> Session:
        return self.inp.session

    @property
    def finished(self) -> bool:
        return self.status == "ready"


# ---- worker -> UI messages ---------------------------------------------------------------


class SeasonStarted(Message):
    def __init__(self, idx: int, total: int):
        super().__init__()
        self.idx, self.total = idx, total


class FileDone(Message):
    def __init__(self, idx: int, done: int, total: int, path: Path, error: str | None):
        super().__init__()
        self.idx, self.done, self.total, self.path, self.error = idx, done, total, path, error


class SeasonReady(Message):
    def __init__(self, idx: int, error: str | None, notices: list[str], spent: float):
        super().__init__()
        self.idx, self.error, self.notices, self.spent = idx, error, notices, spent


class LlmDone(Message):
    def __init__(self, idx: int, spent: float, error: str | None):
        super().__init__()
        self.idx, self.spent, self.error = idx, spent, error


class ApplyDone(Message):
    def __init__(self, results: list[tuple[str, ApplyResult]], error: str | None,
                 run_id: int | None):
        super().__init__()
        self.results, self.error, self.run_id = results, error, run_id


class _Cancelled(Exception):
    pass


def over_budget(est: float, spent: float, budget: float) -> str:
    need = math.ceil((spent + est) * 100 - 1e-9) / 100
    left = max(0.0, budget - spent)
    return (f"Not sending: estimated ${est:.4f} exceeds what --budget {budget:g} leaves "
            f"(${left:.4f}). Re-run with --budget {need:.2f} to allow it.")


# ---- the main screen ---------------------------------------------------------------------


class MainScreen(Screen[None]):
    BINDINGS = [
        Binding("n", "next_review", "next"),
        Binding("N", "prev_review", "prev", show=False),
        Binding("a", "accept(0)", "a/b/c accept"),
        Binding("b", "accept(1)", "b", show=False),
        Binding("c", "accept(2)", "c", show=False),
        Binding("e", "enter_episode", "e S01E##"),
        Binding("x", "extra", "x extra"),
        Binding("u", "unpin", "u unpin"),
        Binding("escape", "undo_pin", "Esc undo pin"),
        Binding("l", "ask_llm", "l ask LLM"),
        Binding("f", "filter", "f filter"),
        Binding("p", "plan", "p plan"),
        Binding("left", "season(-1)", "season", show=False),
        Binding("right", "season(1)", "season", show=False),
        Binding("question_mark", "help", "? help"),
        Binding("q", "quit_review", "q quit"),
    ]
    DEFAULT_CSS = """
    #title { height: 1; padding: 0 1; background: $primary; color: $text; }
    #progress { height: 1; padding: 0 1; }
    #main { height: 1fr; layout: horizontal; }
    #tree { width: 45%; height: 100%; border-right: solid $primary; }
    #evidence { width: 1fr; height: 100%; padding: 0 1; }
    #main.narrow { layout: vertical; }
    #main.narrow #tree { width: 100%; height: 50%; border-right: none;
                         border-bottom: solid $primary; }
    #main.narrow #evidence { width: 100%; height: 1fr; }
    #status { height: auto; padding: 0 1; color: $warning; }
    """

    # actions that change a session
    MUTATING = {"accept", "enter_episode", "extra", "unpin", "undo_pin", "ask_llm"}

    def __init__(self, app_: ReviewApp):
        super().__init__()
        self.r = app_

    def compose(self) -> ComposeResult:
        yield Static("", id="title")
        yield Tabs(*[Tab(s.inp.title, id=f"season-{i}") for i, s in enumerate(self.r.seasons)],
                   id="tabs")
        yield ProgressBar(id="progress", show_eta=False)
        with Horizontal(id="main"):
            yield SeasonTree(id="tree")
            yield EvidencePane(id="evidence")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#tabs", Tabs).can_focus = False
        self.query_one(SeasonTree).focus()
        self._layout()
        self.refresh_view()

    def on_resize(self) -> None:
        self._layout()

    def _layout(self) -> None:
        self.query_one("#main").set_class(self.size.width < NARROW, "narrow")

    # ---- state shortcuts ------------------------------------------------------------------

    @property
    def st(self) -> SeasonState:
        return self.r.seasons[self.r.active]

    @property
    def tree(self) -> SeasonTree:
        return self.query_one(SeasonTree)

    def status(self, text: str) -> None:
        self.r.message = text
        self.update_status()

    def update_status(self) -> None:
        busy = self.r.busy_reason(self.r.active)
        parts = [p for p in (busy, self.r.message) if p]
        self.query_one("#status", Static).update(" · ".join(parts))

    # ---- rendering ------------------------------------------------------------------------

    def refresh_view(self, cursor: Path | None = None) -> None:
        st = self.st
        if cursor is not None:
            st.cursor = cursor
        if st.finished or st.status == "applied":
            rows = st.rows
            show_all = self.r.show_all
        else:
            rows = progress_rows(st.season_node, st.files, st.status == "queued") \
                if st.season_node is not None else []
            show_all = True
        self.tree.show_rows(rows, show_all, st.cursor, st.ignored)
        self.update_title()
        self.update_evidence()
        self.update_status()
        self.refresh_bindings()

    def update_title(self) -> None:
        st = self.st
        ordering = st.ordering  # refreshed after each solve: the Session may be busy
        o = self.r.opts.bundle.orderings.get(ordering)
        order = (o.name if o is not None and o.name else ordering) + " order"
        state = {
            "queued": "queued",
            "analyzing": f"analyzing {st.done}/{st.total}",
            "failed": f"analysis failed: {st.error}",
            "applied": "applied",
        }.get(st.status, "")
        if st.finished and st.report is not None:
            q = sum(r.in_queue for r in st.rows)
            state = f"{q} to review" if q else "nothing to review"
        filt = "all files" if self.r.show_all else "needs review"
        self.query_one("#title", Static).update(
            f"{self.r.opts.show_title} · {st.inp.title} · {order} · {filt}"
            + (f" ── {state}" if state else ""))
        bar = self.query_one(ProgressBar)
        bar.display = st.status in ("queued", "analyzing")
        bar.update(total=max(st.total, 1), progress=st.done)

    def update_evidence(self) -> None:
        pane = self.query_one(EvidencePane)
        st = self.st
        row = self.tree.current
        notices = list(st.report.notices) if st.report is not None else []
        if row is None:
            pane.show(render_detail(None, None, Path("(no file)"), "No file selected.",
                                    notices))
            return
        if not st.finished:
            pane.show(render_detail(None, None, row.path, row.inert_reason))
            return
        detail, msg = None, row.inert_reason
        if row.actionable:
            detail, msg = self.r.explain(self.r.active, row.path)
        pane.show(render_detail(row.report, detail, row.path, msg, notices))

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list is not self.tree:
            return
        row = self.tree.current
        if row is not None:
            self.st.cursor = row.path
        self.update_evidence()

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab is None or event.tab.id is None:
            return
        idx = int(event.tab.id.split("-")[1])
        if idx != self.r.active:
            self.r.active = idx
            self.r.message = ""
        self.refresh_view()

    # ---- bindings -------------------------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in self.MUTATING and self.r.blocked(self.r.active, action):
            return None  # shown greyed; the reason is in the status line
        if action == "plan" and self.r.worker in ("llm", "apply"):
            return None
        return True

    def _actionable(self) -> Row | None:
        row = self.tree.current
        if row is None:
            self.status("No file under the cursor.")
            return None
        if not row.actionable:
            self.status(row.inert_reason or "Nothing can be done with this file.")
            return None
        return row

    def action_next_review(self) -> None:
        self._step(1)

    def action_prev_review(self) -> None:
        self._step(-1)

    def _step(self, d: int) -> None:
        rows = self.tree.rows
        queue = [k for k, r in enumerate(rows) if r.in_queue]
        if not queue:
            self.status("Nothing to review in this season.")
            return
        cur = self.tree.current
        pos = rows.index(cur) if cur in rows else -1
        if d > 0:
            nxt = next((k for k in queue if k > pos), queue[0])
        else:
            nxt = next((k for k in reversed(queue) if k < pos), queue[-1])
        self.tree.move_to(rows[nxt].path)

    def action_season(self, d: int) -> None:
        n = len(self.r.seasons)
        if n > 1:
            self.query_one("#tabs", Tabs).active = f"season-{(self.r.active + d) % n}"

    def action_accept(self, k: int) -> None:
        row = self._actionable()
        if row is None:
            return
        if self.r.blocked(self.r.active, "accept"):
            self.status(self.r.busy_reason(self.r.active) or "Not now.")
            return
        detail, msg = self.r.explain(self.r.active, row.path)
        if detail is None:
            self.status(msg or "No candidates.")
            return
        cands = detail.candidates
        if k >= len(cands):
            self.status(f"There is no candidate {'abc'[k]}.")
            return
        err = self.r.pin(self.r.active, row.path, cands[k].episode_id)
        if err:
            self.status(err)

    def action_extra(self) -> None:
        row = self._actionable()
        if row is not None and (err := self.r.pin(self.r.active, row.path, "extra")):
            self.status(err)

    def action_unpin(self) -> None:
        row = self._actionable()
        if row is not None and (err := self.r.unpin(self.r.active, row.path)):
            self.status(err)

    def action_undo_pin(self) -> None:
        err = self.r.undo_last()
        if err:
            self.status(err)

    def action_enter_episode(self) -> None:
        row = self._actionable()
        if row is None:
            return
        sess = self.st.session
        idx, path = self.r.active, row.path
        bundle = self.r.opts.bundle
        rng = ordering_range(bundle, sess.ordering, sess.include_specials)

        def submit(text: str) -> str | None:
            nums = two_numbers(text)
            if nums is None:
                return "Type two numbers, season then episode (S01E05, 1x5, 1-5, 1 5)."
            s, e = nums
            if s == 0 and not sess.include_specials:
                return f"Season 0 (specials) is not in scope here. Valid: {rng}"
            eid = episode_in_ordering(bundle, sess.ordering, s, e)
            if eid is None:
                return f"S{s:02d}E{e:02d} is not an episode of the {sess.ordering} order. " \
                       f"Valid: {rng}"
            return self.r.pin(idx, path, eid)

        self.app.push_screen(EpisodeEntry(path.name, rng, submit))

    def action_ask_llm(self) -> None:
        row = self._actionable()
        if row is not None and (err := self.r.ask_llm(self.r.active, row.path)):
            self.status(err)

    def action_filter(self) -> None:
        self.r.show_all = not self.r.show_all
        self.refresh_view()

    def action_plan(self) -> None:
        self.r.open_plan()

    def action_help(self) -> None:
        self.app.push_screen(HelpScreen())

    def action_quit_review(self) -> None:
        self.r.request_quit()


# ---- the app -----------------------------------------------------------------------------


class ReviewApp(App[int]):
    TITLE = "showvenist"
    # every way out goes through request_quit (pending pins, the running apply, cancelling the
    # analysis): Textual's own ctrl+q binding calls action_quit, overridden below

    def __init__(self, seasons: list[SeasonInput], opts: ReviewOptions):
        super().__init__()
        self.opts = opts
        self.seasons = [SeasonState(s) for s in seasons]
        self.active = 0
        self.show_all = False
        self.message = ""
        self.worker: str | None = None  # analysis | llm | apply
        self.analyzing: int | None = None
        # LLM spend of this invocation, against --budget: written by the workers that spent it
        # (in their thread, under the lock), so a later season's check sees it at once
        self._spend_lock = threading.Lock()
        self._llm_spent = 0.0
        self.last_pin: tuple[int, Path, bool, str | None] | None = None
        self.exclude: set[Path] = set()
        self.plan_pending = False  # a plan with moves was shown and not applied
        self.applied = False
        self.run_id: int | None = None
        self.overrides_recorded: set[Path] = set()  # pins already written as `rejected`
        self.recovery = ""
        self.summary = ""
        self._cancel = threading.Event()
        self.main: MainScreen | None = None

    # ---- lifecycle ------------------------------------------------------------------------

    def on_mount(self) -> None:
        self.main = MainScreen(self)
        self.push_screen(self.main)
        self._recover()
        self.worker = "analysis"
        self._analyze_all()

    def _recover(self) -> None:
        """Leftovers of an interrupted run are reconciled first (§13.4), as `identify
        --apply` does: only when state.db exists (nothing is created just to look)."""
        from showvenist.rename.movelog import recover
        from showvenist.rename.render import format_recovery
        from showvenist.state.db import state_db_path

        exists = self.opts.state_db_exists or (lambda: state_db_path().exists())
        if not exists():
            return
        try:
            db = self.opts.open_state_db()
        except Exception as e:  # noqa: BLE001 - shown, never fatal
            self.message = f"state.db could not be opened for recovery: {e}"
            return
        try:
            self.recovery = format_recovery(recover(db))
        finally:
            db.close()
        if self.recovery:
            self.message = self.recovery.strip().splitlines()[0]

    @property
    def llm_spent(self) -> float:
        with self._spend_lock:
            return self._llm_spent

    def _spend(self, usd: float) -> None:
        with self._spend_lock:
            self._llm_spent += usd

    def _view(self) -> MainScreen | None:
        return self.main if self.main is not None and self.main.is_mounted else None

    # ---- analysis worker ------------------------------------------------------------------

    @work(thread=True, exclusive=True, group="session", exit_on_error=False)
    def _analyze_all(self) -> None:
        worker = get_current_worker()
        for idx, st in enumerate(self.seasons):
            if self._cancel.is_set() or worker.is_cancelled:
                return
            sess = st.session
            self.post_message(SeasonStarted(idx, st.total))

            def progress(done, total, path, error, idx=idx):
                if self._cancel.is_set():
                    raise _Cancelled
                self.post_message(FileDone(idx, done, total, Path(path), error))

            notices: list[str] = []
            spent = 0.0
            try:
                sess.analyze(None, progress=progress)
                if sess.files or sess.named_rows:
                    sess.score()
                    sess.solve()
                    notices, spent = self._auto_llm(sess)
            except _Cancelled:
                return
            except Exception as e:  # noqa: BLE001 - shown on the tab, the next season goes on
                self.post_message(SeasonReady(idx, f"{type(e).__name__}: {e}", [], 0.0))
                continue
            self.post_message(SeasonReady(idx, None, notices, spent))

    def _auto_llm(self, sess: Session) -> tuple[list[str], float]:
        """--llm auto/all after the first solve, within --budget (as the CLI does)."""
        mode = self.opts.llm_mode
        if mode == "off" or not sess.files or self.opts.llm_factory is None:
            return [], 0.0
        scorer = self.opts.llm_factory()
        scorer.send = set(sess.llm_targets(mode))
        if not scorer.send:
            return [], 0.0
        est = scorer.estimate_cost(sess.files, sess.context())
        spent = self.llm_spent  # includes every earlier season: added below, in this thread
        if est > self.opts.budget - spent + 1e-12:
            return [f"LLM ({mode}): " + over_budget(est, spent, self.opts.budget)], 0.0
        try:
            sess.apply_llm(scorer)
        finally:
            self._spend(scorer.total_cost_usd)
        return [f"LLM ({mode}): sent {len(scorer.send)} file(s), spent "
                f"${scorer.total_cost_usd:.4f}"], scorer.total_cost_usd

    def on_season_started(self, m: SeasonStarted) -> None:
        st = self.seasons[m.idx]
        st.status, st.total, st.done = "analyzing", m.total, 0
        self.analyzing = m.idx
        self._refresh_if(m.idx)

    def on_file_done(self, m: FileDone) -> None:
        st = self.seasons[m.idx]
        st.done = max(st.done, m.done)
        st.total = m.total
        if m.error:
            st.files[m.path] = UNREADABLE
        elif m.path in st.not_ready:
            st.files[m.path] = NOT_READY
        else:
            st.files[m.path] = READY
        self._refresh_if(m.idx)

    def on_season_ready(self, m: SeasonReady) -> None:
        st = self.seasons[m.idx]
        st.llm_notices = m.notices  # the spend was counted by the worker
        if m.error is not None:
            st.status, st.error = "failed", m.error
        else:
            st.status = "ready"
            self._rebuild(m.idx)
            low = least_confident(st.rows)
            st.cursor = low.path if low is not None else None
        if m.idx == len(self.seasons) - 1:
            self.worker = None
            self.analyzing = None
        self._refresh_if(m.idx, always_status=True)

    def _refresh_if(self, idx: int, always_status: bool = False) -> None:
        v = self._view()
        if v is None:
            return
        if idx == self.active:
            v.refresh_view()
        elif always_status:
            v.update_status()
            v.refresh_bindings()

    # ---- session calls (UI thread, only on finished seasons) ------------------------------

    def busy_reason(self, idx: int) -> str:
        st = self.seasons[idx]
        if self.worker == "apply":
            return "Applying the plan: keys wait until it ends."
        if self.worker == "llm":
            return "Asking the LLM: keys that change pins wait until it answers."
        if st.status in ("queued", "analyzing"):
            return f"{st.inp.title} is still being analyzed: pins and the LLM wait for it."
        if st.status == "applied":
            return "Applied: run identify again to review what is left."
        if self.worker == "analysis" and self.analyzing is not None:
            return f"{self.seasons[self.analyzing].inp.title} is being analyzed: l and the " \
                   "apply wait for it."
        return ""

    def session_busy(self, idx: int) -> bool:
        """True while a worker may be using season `idx`'s Session: the UI must not call it."""
        st = self.seasons[idx]
        return (self.worker in ("llm", "apply") or not st.finished
                or (self.worker == "analysis" and self.analyzing == idx))

    def explain(self, idx: int, path: Path) -> tuple[Any, str | None]:
        """(FileDetail or None, message) for the evidence pane. Each solve's details are
        cached; while a worker runs, only the cache is read (never the Session)."""
        st = self.seasons[idx]
        got = st.details.get(path)
        if got is not None:
            return (None, got) if isinstance(got, str) else (got, None)
        if self.session_busy(idx):
            what = {"llm": "the LLM call", "apply": "the apply"}.get(self.worker or "",
                                                                     "the analysis")
            return None, f"Waiting for {what} to finish: the evidence shows when it ends."
        try:
            detail = st.session.explain(path)
        except (ValueError, KeyError) as e:
            st.details[path] = str(e)
            return None, str(e)
        st.details[path] = detail
        return detail, None

    def blocked(self, idx: int, action: str) -> bool:
        st = self.seasons[idx]
        if self.worker in ("llm", "apply") or not st.finished:
            return True
        return action == "ask_llm" and self.worker is not None

    def _report(self, idx: int) -> Report:
        st = self.seasons[idx]
        rep = st.session.report()
        rep.notices += st.inp.notices(st.session) + st.llm_notices
        return rep

    def _rebuild(self, idx: int) -> None:
        st = self.seasons[idx]
        st.report = self._report(idx)
        st.pinned = set(st.session.pins)
        st.ordering = st.session.ordering
        st.details.clear()
        st.rows = build_rows(st.report, st.pre_pin, st.pinned)

    def _primary(self, sess: Session, path: Path) -> Path:
        return sess.duplicates.get(path, path)

    def _after_change(self, idx: int) -> None:
        self._rebuild(idx)
        st = self.seasons[idx]
        low = least_confident(st.rows)
        st.cursor = low.path if low is not None else st.cursor
        self.message = "" if low is not None else \
            f"{st.inp.title}: nothing left to review. p shows the plan."
        if self._view() is not None and idx == self.active:
            self.main.refresh_view()  # type: ignore[union-attr]

    def pin(self, idx: int, path: Path, episode: str) -> str | None:
        """Pin (episode id or "extra"), re-solve, move the cursor to the least confident file.
        Returns an error to show, or None."""
        st = self.seasons[idx]
        if self.blocked(idx, "accept"):
            return self.busy_reason(idx) or "Not now."
        sess = st.session
        prim = self._primary(sess, path)
        had = prim in sess.pins
        prev = sess.pins.get(prim)
        before = next((r.report.confidence.level for r in st.rows
                       if r.path == prim and r.report is not None), None)
        try:
            sess.pin(prim, episode)
        except (ValueError, KeyError) as e:
            return str(e)
        if not had and before is not None:
            st.pre_pin[prim] = before
        self.last_pin = (idx, prim, had, prev)
        self._after_change(idx)
        return None

    def unpin(self, idx: int, path: Path) -> str | None:
        st = self.seasons[idx]
        if self.blocked(idx, "unpin"):
            return self.busy_reason(idx) or "Not now."
        sess = st.session
        prim = self._primary(sess, path)
        if prim not in sess.pins:
            return f"{path.name} is not pinned."
        prev = sess.pins[prim]
        sess.unpin(prim)
        st.pre_pin.pop(prim, None)
        self.last_pin = (idx, prim, True, prev)
        self._after_change(idx)
        return None

    def undo_last(self) -> str | None:
        """Esc: the last pin or unpin is reverted (one level)."""
        if self.last_pin is None:
            return "Nothing to undo."
        idx, path, had, prev = self.last_pin
        if self.blocked(idx, "undo_pin"):
            return self.busy_reason(idx) or "Not now."
        sess = self.seasons[idx].session
        try:
            if had:
                sess.pin(path, "extra" if prev is None else prev)
            else:
                sess.unpin(path)
                self.seasons[idx].pre_pin.pop(path, None)
        except (ValueError, KeyError) as e:
            return str(e)
        self.last_pin = None
        if idx != self.active:  # the undone pin is on another tab: show it
            self.active = idx
            if self._view() is not None:
                self.main.query_one("#tabs", Tabs).active = f"season-{idx}"  # type: ignore[union-attr]
        self._after_change(idx)
        self.message = f"Undid the last change to {path.name}."
        if self._view() is not None:
            self.main.update_status()  # type: ignore[union-attr]
        return None

    # ---- LLM (`l`) ------------------------------------------------------------------------

    def ask_llm(self, idx: int, path: Path) -> str | None:
        st = self.seasons[idx]
        if self.blocked(idx, "ask_llm"):
            return self.busy_reason(idx) or "Not now."
        if self.opts.llm_factory is None:
            return "The LLM scorer is not available here."
        sess = st.session
        prim = self._primary(sess, path)
        rows = [f.path for f in sess.files]
        if prim not in rows:
            return f"{path.name} has no evidence to send."
        scorer = self.opts.llm_factory()
        scorer.send = {rows.index(prim)}
        est = scorer.estimate_cost(sess.files, sess.context())
        if est > self.opts.budget - self.llm_spent + 1e-12:
            return over_budget(est, self.llm_spent, self.opts.budget)
        self.worker = "llm"
        self.message = f"Asking the LLM about {path.name} (estimated ${est:.4f})."
        if self._view() is not None:
            self.main.refresh_view()  # type: ignore[union-attr]
        self._llm_worker(idx, scorer)
        return None

    @work(thread=True, exclusive=True, group="session", exit_on_error=False)
    def _llm_worker(self, idx: int, scorer) -> None:
        try:
            self.seasons[idx].session.apply_llm(scorer)
        except Exception as e:  # noqa: BLE001
            self._spend(scorer.total_cost_usd)
            self.post_message(LlmDone(idx, scorer.total_cost_usd, f"{type(e).__name__}: {e}"))
            return
        self._spend(scorer.total_cost_usd)
        self.post_message(LlmDone(idx, scorer.total_cost_usd, None))

    def on_llm_done(self, m: LlmDone) -> None:
        self.worker = "analysis" if self.analyzing is not None else None
        self._after_change(m.idx)
        self.message = (f"LLM failed: {m.error}" if m.error else
                        f"LLM answered (spent ${m.spent:.4f}, ${self.llm_spent:.4f} so far).")
        if self._view() is not None:
            self.main.update_status()  # type: ignore[union-attr]

    # ---- plan and apply -------------------------------------------------------------------

    def finished(self) -> list[int]:
        return [i for i, s in enumerate(self.seasons) if s.finished]

    def build_plans(self, exclude: set[Path]) -> Plans:
        from showvenist.rename.plan import build_plan

        out: Plans = []
        opts = replace(self.opts.plan_options, exclude=set(exclude))
        for i in self.finished():
            st = self.seasons[i]
            if st.report is None:
                continue
            try:
                plan = build_plan(st.report, self.opts.bundle, opts, history=self.opts.history,
                                  **self.opts.plan_kwargs)
            except ValueError as e:
                plan = Plan(show_slug=st.report.show_slug, naming_order="?", errors=[str(e)])
            out.append((st.inp.title, plan))
        return out

    def apply_blocked(self) -> str | None:
        if self.applied:
            return "Already applied in this session: run identify again."
        if self.worker is not None:
            return self.busy_reason(self.active) or "Wait for the running task to finish."
        return None

    def open_plan(self) -> None:
        if self.worker in ("llm", "apply"):
            self.main.status(self.busy_reason(self.active))  # type: ignore[union-attr]
            return
        if self.applied:
            self.main.status("Already applied: run identify again.")  # type: ignore[union-attr]
            return
        screen = PlanScreen(self.build_plans, self.exclude, self.apply_blocked,
                            self.start_apply, applying=lambda: self.worker == "apply")

        def closed(_=None) -> None:
            if not self.applied:
                self.plan_pending = screen.moves > 0

        self.push_screen(screen, closed)

    def start_apply(self, plans: Plans, cross_ok: bool) -> None:
        why = self.apply_blocked()
        if why:
            if isinstance(self.screen, PlanScreen):
                self.screen.status(why)
            return
        self.worker = "apply"
        if isinstance(self.screen, PlanScreen):
            self.screen.status("Applying…")
        self._apply_worker(list(plans), cross_ok)

    @work(thread=True, exclusive=True, group="session", exit_on_error=False)
    def _apply_worker(self, plans: Plans, cross_ok: bool) -> None:
        """One run for the invocation (§12.4): recovery, the run row after consent, the
        overrides as `rejected` matches, then every plan through apply_plan."""
        from showvenist.rename.movelog import PlanRefused, apply_plan
        from showvenist.session import record_overrides

        results: list[tuple[str, ApplyResult]] = []
        try:
            db = self.opts.open_state_db()
        except Exception as e:  # noqa: BLE001
            self.post_message(ApplyDone([], f"state.db could not be opened: {e}", None))
            return
        try:
            hist = History(db)
            if self.run_id is None:
                self.run_id = hist.begin_run("identify", self.opts.command_args).id
            # overrides only for files in the plan being applied (not unticked or left out),
            # each once even when the apply is retried after a refusal
            in_plan = {m.old for _, p in plans for m in p.moves if m.kind != "sidecar"}
            for i in self.finished():
                sess = self.seasons[i].session
                only = {p for p in sess.pins if p in in_plan} - self.overrides_recorded
                if only:
                    record_overrides(hist, sess, self.run_id, sess.show_slug, only=only)
                    self.overrides_recorded |= only
            for title, plan in plans:
                if not plan.moves:
                    continue
                res = apply_plan(plan, db, command="identify", args=self.opts.command_args,
                                 confirm=lambda p: True,
                                 confirm_cross_fs=lambda ms: cross_ok, run_id=self.run_id)
                results.append((title, res))
        except PlanRefused as e:
            self.post_message(ApplyDone(results, str(e), self.run_id))
            return
        except Exception as e:  # noqa: BLE001
            self.post_message(ApplyDone(results, f"{type(e).__name__}: {e}", self.run_id))
            return
        finally:
            db.close()
        self.post_message(ApplyDone(results, None, self.run_id))

    def on_apply_done(self, m: ApplyDone) -> None:
        self.worker = None
        moved = any(r.done for _, r in m.results)
        if moved or m.results:
            self.applied = True
            self.plan_pending = False
            for st in self.seasons:
                if st.finished:
                    st.status = "applied"
        self.summary = summary_text(m.results, self.recovery, m.error)
        if isinstance(self.screen, PlanScreen):
            self.pop_screen()
        if self._view() is not None:
            self.main.refresh_view()  # type: ignore[union-attr]
        self.push_screen(SummaryScreen(self.summary))

    # ---- quit -----------------------------------------------------------------------------

    def pending(self) -> str | None:
        pins = sum(len(s.pinned) for s in self.seasons if s.finished)
        what = []
        if pins:
            what.append(f"{pins} pin{'s' if pins != 1 else ''}")
        if self.plan_pending:
            what.append("an unapplied plan")
        return " and ".join(what) or None

    async def action_quit(self) -> None:
        """ctrl+q (Textual's binding) asks like q does."""
        self.request_quit()

    def _say(self, text: str) -> None:
        if isinstance(self.screen, PlanScreen):
            self.screen.status(text)
        elif self._view() is not None:
            self.main.status(text)  # type: ignore[union-attr]

    def request_quit(self) -> None:
        """Every way out: asks about pending pins or plan, never quits during an apply, and
        cancels the analysis when it quits."""
        if self.worker == "apply":
            self._say("Files are being moved: wait for the summary.")
            return
        if isinstance(self.screen, QuitDialog):
            return
        what = None if self.applied else self.pending()
        if what is None:
            self.quit_now()
            return
        on_plan = isinstance(self.screen, PlanScreen)

        def answer(choice: str | None) -> None:
            if choice == "discard":
                self.quit_now()
            elif choice == "apply" and not on_plan:
                self.open_plan()

        self.push_screen(QuitDialog(what), answer)

    def quit_now(self) -> None:
        self._cancel.set()  # the analysis stops after the files in flight (the cache keeps them)
        self.workers.cancel_group(self, "session")
        self.exit(0)


def run_review(seasons: list[SeasonInput], opts: ReviewOptions, **run_kwargs: Any) -> int:
    """Run the TUI; returns the exit code: 1 when the app crashed (§4.5)."""
    app = ReviewApp(seasons, opts)
    code = app.run(**run_kwargs)
    rc = app.return_code
    if rc is None or rc != 0:
        return 1 if rc is None else int(rc)
    return 0 if code is None else int(code)


__all__ = ["ReviewApp", "ReviewOptions", "SeasonInput", "run_review"]
