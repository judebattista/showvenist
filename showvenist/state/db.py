"""SQLite state database (spec §7.2): connection, migration and row types. The M5 contract.

Standard-library `sqlite3`, no ORM, WAL mode. `$XDG_DATA_HOME/showvenist/state.db`. The
analysis cache is a separate, disposable file (§7.3) and never touches this one.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from showvenist.config import data_dir
from showvenist.state.migrations import migrate

MatchStatus = Literal["confirmed", "rejected", "undone"]
MatchSource = Literal["auto", "manual", "name"]
OpKind = Literal["episode", "extra", "duplicate", "sidecar"]
OpStatus = Literal["planned", "done", "undone"]


def state_db_path() -> Path:
    return data_dir() / "state.db"


@dataclass(frozen=True)
class MediaFile:
    fingerprint: str
    content_key: str | None
    size: int
    last_path: str
    show_slug: str | None
    season_label: str | None
    disc_label: str | None
    first_seen: str
    last_seen: str


@dataclass(frozen=True)
class Run:
    id: int
    started_at: str
    command: str
    args: str


@dataclass(frozen=True)
class Match:
    id: int
    fingerprint: str
    show_slug: str
    episode_id: str | None  # None = extra
    season: int | None
    episode: int | None
    title: str | None
    status: MatchStatus
    source: MatchSource
    confidence: Literal["high", "medium", "low"] | None
    margin: float | None
    bundle_revision: str | None
    run_id: int | None
    created_at: str


@dataclass(frozen=True)
class RenameOp:
    id: int
    run_id: int
    fingerprint: str
    kind: OpKind
    old_path: str
    new_path: str
    status: OpStatus
    match_id: int | None
    show_slug: str | None
    episode_id: str | None
    dupe_index: int | None
    applied_at: str | None
    undone_at: str | None


class StateDb:
    """One connection, serialized by a lock (the TUI and the CLI share it across threads)."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.migrated_from = migrate(self.conn)

    @classmethod
    def open_default(cls) -> StateDb:
        return cls(state_db_path())

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Serialized `with conn:` block: commits on success, rolls back on error."""
        with self._lock, self.conn:
            yield self.conn

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> StateDb:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
