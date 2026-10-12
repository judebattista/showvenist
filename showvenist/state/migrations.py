"""state.db schema and migrator (spec §7.2, decision 59). The M5 contract.

`MIGRATIONS[i]` takes the schema from version i to i + 1; `PRAGMA user_version` records the
version. Opening a database newer than this code refuses (never downgrades or guesses).
"""

from __future__ import annotations

import sqlite3

MATCH_STATUSES = ("confirmed", "rejected", "undone")  # no 'proposed': state records decisions only
MATCH_SOURCES = ("auto", "manual", "name")
OP_KINDS = ("episode", "extra", "duplicate", "sidecar")
OP_STATUSES = ("planned", "done", "undone")


def _in(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


_V1 = f"""
CREATE TABLE media_file (
    fingerprint  TEXT PRIMARY KEY,
    content_key  TEXT,
    size         INTEGER NOT NULL,
    last_path    TEXT NOT NULL,
    show_slug    TEXT,
    season_label TEXT,
    disc_label   TEXT,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL
);
CREATE INDEX media_file_content_key ON media_file (content_key);

CREATE TABLE run (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    command    TEXT NOT NULL,
    args       TEXT NOT NULL
);

CREATE TABLE match (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint    TEXT NOT NULL REFERENCES media_file (fingerprint),
    show_slug      TEXT NOT NULL,
    episode_id     TEXT,            -- NULL = extra
    season         INTEGER,         -- as numbered when recorded; NULL for extras
    episode        INTEGER,
    title          TEXT,
    status         TEXT NOT NULL CHECK (status IN ({_in(MATCH_STATUSES)})),
    source         TEXT NOT NULL CHECK (source IN ({_in(MATCH_SOURCES)})),
    confidence     TEXT CHECK (confidence IN ('high', 'medium', 'low')),
    margin         REAL,
    bundle_revision TEXT,
    run_id         INTEGER REFERENCES run (id),
    created_at     TEXT NOT NULL,
    CHECK (episode_id IS NOT NULL OR (season IS NULL AND episode IS NULL))
);
CREATE INDEX match_file ON match (fingerprint, status);
CREATE INDEX match_episode ON match (show_slug, episode_id, status);
CREATE INDEX match_run ON match (run_id);

CREATE TABLE rename_op (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     INTEGER NOT NULL REFERENCES run (id),
    fingerprint TEXT NOT NULL,      -- the video's; a sidecar carries its video's
    kind       TEXT NOT NULL CHECK (kind IN ({_in(OP_KINDS)})),
    old_path   TEXT NOT NULL,
    new_path   TEXT NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ({_in(OP_STATUSES)})),
    match_id   INTEGER REFERENCES match (id),
    show_slug  TEXT,                -- with episode_id and dupe_index: numbers dupe_X (decision 56)
    episode_id TEXT,
    dupe_index INTEGER,
    applied_at TEXT,
    undone_at  TEXT
);
CREATE INDEX rename_op_run ON rename_op (run_id, status);
CREATE INDEX rename_op_status ON rename_op (status);
CREATE INDEX rename_op_dupe ON rename_op (show_slug, episode_id, dupe_index);
"""

# v2: what a journaled move decided, written with the `planned` row, so that recovery can
# confirm the match of a move it completes (§13.4 step 3) after the plan is gone; and whether
# the move crosses filesystems (recovery may finish a verified copy by removing the original).
# All nullable: rows journaled under v1 have none of it (recovery then writes no match).
_V2 = f"""
ALTER TABLE rename_op ADD COLUMN season INTEGER;
ALTER TABLE rename_op ADD COLUMN episode INTEGER;
ALTER TABLE rename_op ADD COLUMN title TEXT;
ALTER TABLE rename_op ADD COLUMN match_source TEXT CHECK (match_source IN ({_in(MATCH_SOURCES)}));
ALTER TABLE rename_op ADD COLUMN confidence TEXT CHECK (confidence IN ('high', 'medium', 'low'));
ALTER TABLE rename_op ADD COLUMN bundle_revision TEXT;
ALTER TABLE rename_op ADD COLUMN disc_label TEXT;
ALTER TABLE rename_op ADD COLUMN prior_match INTEGER;  -- 1: links an earlier confirmed match
ALTER TABLE rename_op ADD COLUMN cross_fs INTEGER;     -- 1: copy, verify, remove the original
"""

# v3: two-episode files (decision 64). `episode_end` is the last episode's number of a file that
# holds two consecutive episodes, in the same numbering as `episode`; NULL = one episode (every
# row written before v3).
_V3 = """
ALTER TABLE match ADD COLUMN episode_end INTEGER;
ALTER TABLE rename_op ADD COLUMN episode_end INTEGER;
"""

# v4: `created_dir` is the outermost folder a move had to create for its target (the plan's
# `extras/`), NULL when the target folder existed (every row written before v4). Undo removes
# such a folder again when it is empty afterwards (§13.5), and no other folder.
_V4 = """
ALTER TABLE rename_op ADD COLUMN created_dir TEXT;
"""

MIGRATIONS: tuple[str, ...] = (_V1, _V2, _V3, _V4)
SCHEMA_VERSION = len(MIGRATIONS)


class SchemaTooNew(Exception):
    """state.db was written by a newer showvenist."""


def migrate(conn: sqlite3.Connection) -> int:
    """Bring `conn` to SCHEMA_VERSION; returns the version it started from."""
    start = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if start > SCHEMA_VERSION:
        raise SchemaTooNew(f"state.db is schema {start}; this showvenist knows {SCHEMA_VERSION}")
    for v in range(start, SCHEMA_VERSION):
        with conn:  # one transaction per step
            conn.executescript("BEGIN;" + MIGRATIONS[v])
            conn.execute(f"PRAGMA user_version = {v + 1}")
    return start
