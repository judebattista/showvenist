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
from showvenist.state.migrations import SCHEMA_VERSION, SchemaTooNew, migrate

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
    # decision 64: the last episode's number of a two-episode file, numbered like `episode`;
    # None = one episode
    episode_end: int | None = None


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
    # what the move decided, as numbered in the naming order (schema v2; None before it and
    # for sidecars); `episode_end` (schema v3) is the last episode of a two-episode file
    season: int | None = None
    episode: int | None = None
    episode_end: int | None = None


class StateDbMissing(Exception):
    """A read-only open found no state.db."""


class SchemaOutdated(Exception):
    """A read-only open found a state.db older than this showvenist's schema."""


class StateDb:
    """One connection, serialized by a lock (the TUI and the CLI share it across threads)."""

    def __init__(self, path: Path | str, *, readonly: bool = False):
        self.path = str(path)
        self.readonly = readonly
        self._lock = threading.RLock()
        if readonly:
            self._open_readonly()
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.migrated_from = migrate(self.conn)

    def _open_readonly(self) -> None:
        """No migration, no journal-mode change, no file creation; `mode=ro`."""
        if self.path == ":memory:" or not Path(self.path).exists():
            raise StateDbMissing(f"no state.db at {self.path}; run a command that records "
                                 "history first (e.g. identify --apply or assign)")
        uri = Path(self.path).resolve().as_uri() + "?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA foreign_keys=ON")
            version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
            self.migrated_from = version
            if version > SCHEMA_VERSION:
                self.conn.close()
                raise SchemaTooNew(
                    f"state.db is schema {version}; this showvenist knows {SCHEMA_VERSION}")
            if version < SCHEMA_VERSION:
                self.conn.close()
                raise SchemaOutdated(
                    f"state.db is schema {version}, this showvenist needs {SCHEMA_VERSION}; "
                    "run a command that writes history (identify --apply, assign, undo) to "
                    "upgrade it")

    @classmethod
    def open_readonly(cls, path: Path | str | None = None) -> StateDb:
        """Open an existing state.db for reading only (report, --plan, history, eval
        --from-history). Raises StateDbMissing / SchemaOutdated rather than writing."""
        return cls(path if path is not None else state_db_path(), readonly=True)

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
