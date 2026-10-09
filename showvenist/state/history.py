"""Match history (spec §7.2, §11.4, decision 59). Contract; the body is task sv-o4f.2.

State records only what is decided or executed: `identify` in report mode and `--plan` never
call anything here. Statuses are confirmed / rejected / undone (no 'proposed').
"""

from __future__ import annotations

from collections.abc import Iterable

from showvenist.state.db import Match, MatchSource, MediaFile, Run, StateDb
from showvenist.state.migrations import MATCH_STATUSES  # noqa: F401  (documented vocabulary)


class History:
    def __init__(self, db: StateDb):
        self.db = db

    # --- runs and files -------------------------------------------------------------------
    def begin_run(self, command: str, args: str) -> Run:
        """One `run` row per apply / assign / undo invocation."""
        raise NotImplementedError("sv-o4f.2")

    def touch_file(
        self, fingerprint: str, *, size: int, path: str, content_key: str | None = None,
        show_slug: str | None = None, season_label: str | None = None,
        disc_label: str | None = None,
    ) -> MediaFile:
        """Insert or update `media_file` (first_seen kept, last_seen and last_path refreshed)."""
        raise NotImplementedError("sv-o4f.2")

    # --- decisions ------------------------------------------------------------------------
    def confirm(
        self, fingerprint: str, show_slug: str, *, episode_id: str | None, season: int | None,
        episode: int | None, title: str | None, source: MatchSource, run_id: int | None,
        confidence: str | None = None, margin: float | None = None,
        bundle_revision: str | None = None,
    ) -> Match:
        """Record a confirmed match (episode_id None = confirmed extra). Any other confirmed
        match of the same file becomes `rejected` (assign over a confirmed match, §4.4)."""
        raise NotImplementedError("sv-o4f.2")

    def reject(
        self, fingerprint: str, show_slug: str, *, episode_id: str | None, run_id: int | None,
        season: int | None = None, episode: int | None = None, title: str | None = None,
    ) -> Match:
        """Record the tool's top choice as rejected when a pin overrode it (negative label)."""
        raise NotImplementedError("sv-o4f.2")

    def mark_undone(self, run_id: int) -> int:
        """`undo`: the run's confirmed matches become `undone`; returns how many."""
        raise NotImplementedError("sv-o4f.2")

    def forget(self, fingerprint: str) -> int:
        """Remove every record of a file (media_file, match; rename_op rows keep their text
        but lose the file link); returns matches removed."""
        raise NotImplementedError("sv-o4f.2")

    # --- reads ----------------------------------------------------------------------------
    def confirmed(self, show_slug: str, season: int | None = None) -> list[Match]:
        """Confirmed matches (episodes and extras) of a show, optionally one season, as
        numbered when recorded. `undone` and `rejected` rows are not included."""
        raise NotImplementedError("sv-o4f.2")

    def rejected(self, show_slug: str) -> list[Match]:
        """Negative labels for eval."""
        raise NotImplementedError("sv-o4f.2")

    def is_confirmed_extra(self, fingerprint: str) -> bool:
        raise NotImplementedError("sv-o4f.2")

    def confirmed_for_episode(self, show_slug: str, episode_id: str) -> list[Match]:
        """Files already confirmed for an episode (duplicate warnings, the prior)."""
        raise NotImplementedError("sv-o4f.2")

    def history_penalty(
        self, show_slug: str, fingerprints: Iterable[str], episode_ids: list[str],
        penalty: float,
    ) -> list[list[float]]:
        """(files x episodes) penalty matrix for fusion (§11.4): `penalty` where the episode is
        confirmed for a DIFFERENT fingerprint, 0 elsewhere. Soft, never an exclusion."""
        raise NotImplementedError("sv-o4f.2")

    def highest_dupe_index(self, show_slug: str, episode_id: str) -> int:
        """Highest `dupe_X` already used for an episode in `rename_op` (0 if none); the folder is
        checked separately by the planner (decision 56)."""
        raise NotImplementedError("sv-o4f.2")
