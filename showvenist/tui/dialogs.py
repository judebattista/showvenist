"""Small modal screens: the episode entry (`e`), y/n confirms, the quit question and help."""

from __future__ import annotations

import re
from collections.abc import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from showvenist.bundle.schema import Bundle

_NUM = re.compile(r"\d+")

DIALOG_CSS = """
Vertical.dialog { width: 72; max-width: 100%; height: auto; padding: 1 2;
                  border: thick $accent; background: $panel; }
.error { color: $error; }
"""


def two_numbers(text: str) -> tuple[int, int] | None:
    """Season and episode from any text with exactly two separate numbers (`S05E03`, `5x03`,
    `5-3`, `5 3`); None otherwise."""
    nums = _NUM.findall(text)
    if len(nums) != 2:
        return None
    return int(nums[0]), int(nums[1])


def episode_in_ordering(bundle: Bundle, ordering: str, season: int, episode: int) -> str | None:
    """The episode id at (season, episode) in `ordering`, or None."""
    for e in bundle.orderings[ordering].entries:
        if (e.season, e.episode) == (season, episode):
            return e.episode_id
    return None


def ordering_range(bundle: Bundle, ordering: str, specials: bool) -> str:
    """`S01 1-22, S02 1-20` for the ordering (season 0 only when specials are in scope)."""
    by: dict[int, list[int]] = {}
    for e in bundle.orderings[ordering].entries:
        if e.season == 0 and not specials:
            continue
        by.setdefault(e.season, []).append(e.episode)
    return ", ".join(f"S{s:02d} {min(v)}–{max(v)}" for s, v in sorted(by.items()))


class EpisodeEntry(ModalScreen[str | None]):
    """`e`: type a season and an episode. `submit(text)` returns an error to show (the entry
    stays open) or None (done)."""

    DEFAULT_CSS = DIALOG_CSS
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, file_name: str, range_text: str, submit: Callable[[str], str | None]):
        super().__init__()
        self.file_name = file_name
        self.range_text = range_text
        self.submit = submit

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(f"Episode for {self.file_name} (season then episode, e.g. S01E05, "
                         "1x5, 1-5 or 1 5)")
            yield Static(f"Valid: {self.range_text}", id="entry-range")
            yield Input(id="entry-input", placeholder="S01E05")
            yield Static("", id="entry-error", classes="error")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        err = self.submit(event.value)
        if err is None:
            self.dismiss(event.value)
        else:
            self.query_one("#entry-error", Static).update(err)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmDialog(ModalScreen[bool]):
    """y/n question."""

    DEFAULT_CSS = DIALOG_CSS
    BINDINGS = [
        Binding("y", "answer(True)", "Yes"),
        Binding("n", "answer(False)", "No"),
        Binding("escape", "answer(False)", "No", show=False),
    ]

    def __init__(self, question: str):
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.question, id="confirm-question")
            yield Static("y yes · n no")

    def action_answer(self, yes: bool) -> None:
        self.dismiss(yes)


class QuitDialog(ModalScreen[str]):
    """Quitting with pins or an unapplied plan: apply, discard or stay."""

    DEFAULT_CSS = DIALOG_CSS
    BINDINGS = [
        Binding("a", "answer('apply')", "Apply"),
        Binding("d", "answer('discard')", "Discard"),
        Binding("s", "answer('stay')", "Stay"),
        Binding("escape", "answer('stay')", "Stay", show=False),
    ]

    def __init__(self, what: str):
        super().__init__()
        self.what = what

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(f"You have {self.what}, kept only in memory until you apply.",
                         id="quit-question")
            yield Static("a apply (opens the plan) · d discard and quit · s stay")

    def action_answer(self, choice: str) -> None:
        self.dismiss(choice)


HELP = """\
j/k or ↓/↑   move                      ←/→   previous / next season tab
n / N        next / previous file to review
a / b / c    accept candidate a, b or c (pins it, re-solves the season)
e            type an episode: any text with two numbers, season then episode
x            mark as extra             u     unpin
Esc          undo the last pin (one level)
l            ask the LLM about this file (within --budget)
f            show all files / only files that need review
p            plan: every finished season's moves, tick/untick, Enter to apply
?            this help                 q     quit

Glyphs: ✓ high · ? needs review · ✗ extra · • pinned by you · = duplicate · … not ready/reading
"""


class HelpScreen(ModalScreen[None]):
    DEFAULT_CSS = DIALOG_CSS
    BINDINGS = [Binding("escape,q,question_mark", "close", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(HELP, id="help-text")

    def action_close(self) -> None:
        self.dismiss(None)
