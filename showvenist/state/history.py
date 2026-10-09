"""Match history (spec §7.2, §11.4, decision 59).

State records only what is decided or executed: `identify` in report mode and `--plan` never
call anything here. Statuses are confirmed / rejected / undone (no 'proposed').
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from showvenist.state.db import Match, MatchSource, MediaFile, Run, StateDb
from showvenist.state.migrations import MATCH_STATUSES  # noqa: F401  (documented vocabulary)

_MATCH_COLS = (
    "id, fingerprint, show_slug, episode_id, season, episode, title, status, source,"
    " confidence, margin, bundle_revision, run_id, created_at"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _match(r) -> Match:
    return Match(**{k: r[k] for k in r.keys()})


def _file(r) -> MediaFile:
    return MediaFile(**{k: r[k] for k in r.keys()})


class History:
    def __init__(self, db: StateDb):
        self.db = db

    # --- runs and files -------------------------------------------------------------------
    def begin_run(self, command: str, args: str) -> Run:
        """One `run` row per apply / assign / undo invocation."""
        now = _now()
        with self.db.transaction() as c:
            cur = c.execute(
                "insert into run (started_at, command, args) values (?,?,?)", (now, command, args)
            )
            return Run(int(cur.lastrowid), now, command, args)

    def touch_file(
        self, fingerprint: str, *, size: int, path: str, content_key: str | None = None,
        show_slug: str | None = None, season_label: str | None = None,
        disc_label: str | None = None,
    ) -> MediaFile:
        """Insert or update `media_file` (first_seen kept, last_seen and last_path refreshed).
        Optional fields keep their stored value when passed as None."""
        now = _now()
        with self.db.transaction() as c:
            c.execute(
                "insert into media_file (fingerprint, content_key, size, last_path, show_slug,"
                " season_label, disc_label, first_seen, last_seen) values (?,?,?,?,?,?,?,?,?)"
                " on conflict (fingerprint) do update set size=excluded.size,"
                " last_path=excluded.last_path, last_seen=excluded.last_seen,"
                " content_key=coalesce(excluded.content_key, content_key),"
                " show_slug=coalesce(excluded.show_slug, show_slug),"
                " season_label=coalesce(excluded.season_label, season_label),"
                " disc_label=coalesce(excluded.disc_label, disc_label)",
                (fingerprint, content_key, size, path, show_slug, season_label, disc_label,
                 now, now),
            )
            row = c.execute(
                "select * from media_file where fingerprint=?", (fingerprint,)
            ).fetchone()
        return _file(row)

    def get_file(self, fingerprint: str) -> MediaFile | None:
        with self.db.transaction() as c:
            row = c.execute(
                "select * from media_file where fingerprint=?", (fingerprint,)
            ).fetchone()
        return _file(row) if row else None

    # --- decisions ------------------------------------------------------------------------
    def _insert(
        self, c, fingerprint, show_slug, *, episode_id, season, episode, title, status, source,
        run_id, confidence=None, margin=None, bundle_revision=None,
    ) -> Match:
        if episode_id is None and (season is not None or episode is not None):
            raise ValueError("an extra carries no season or episode")
        cur = c.execute(
            "insert into match (fingerprint, show_slug, episode_id, season, episode, title,"
            " status, source, confidence, margin, bundle_revision, run_id, created_at)"
            " values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (fingerprint, show_slug, episode_id, season, episode, title, status, source,
             confidence, margin, bundle_revision, run_id, _now()),
        )
        row = c.execute(
            f"select {_MATCH_COLS} from match where id=?", (cur.lastrowid,)
        ).fetchone()
        return _match(row)

    def confirm(
        self, fingerprint: str, show_slug: str, *, episode_id: str | None, season: int | None,
        episode: int | None, title: str | None, source: MatchSource, run_id: int | None,
        confidence: str | None = None, margin: float | None = None,
        bundle_revision: str | None = None,
    ) -> Match:
        """Record a confirmed match (episode_id None = confirmed extra). Any other confirmed
        match of the same file and show becomes `rejected` (assign over a confirmed match,
        §4.4); re-confirming the same target just replaces the earlier row."""
        if episode_id is None and (season is not None or episode is not None):
            raise ValueError("an extra carries no season or episode")
        with self.db.transaction() as c:
            same = c.execute(
                "select id from match where fingerprint=? and show_slug=? and"
                " status='confirmed' and episode_id is ?",
                (fingerprint, show_slug, episode_id),
            ).fetchone()
            if same:  # re-confirming the same target refreshes the row, rejects nothing
                c.execute(
                    "update match set season=?, episode=?, title=?, source=?, confidence=?,"
                    " margin=?, bundle_revision=?, run_id=?, created_at=? where id=?",
                    (season, episode, title, source, confidence, margin, bundle_revision,
                     run_id, _now(), same[0]),
                )
                c.execute(
                    "update match set status='rejected' where fingerprint=? and show_slug=?"
                    " and status='confirmed' and id<>?", (fingerprint, show_slug, same[0]))
                return _match(c.execute(
                    f"select {_MATCH_COLS} from match where id=?", (same[0],)).fetchone())
            c.execute(
                "update match set status='rejected' where fingerprint=? and show_slug=?"
                " and status='confirmed'",
                (fingerprint, show_slug),
            )
            return self._insert(
                c, fingerprint, show_slug, episode_id=episode_id, season=season,
                episode=episode, title=title, status="confirmed", source=source, run_id=run_id,
                confidence=confidence, margin=margin, bundle_revision=bundle_revision,
            )

    def reject(
        self, fingerprint: str, show_slug: str, *, episode_id: str | None, run_id: int | None,
        season: int | None = None, episode: int | None = None, title: str | None = None,
    ) -> Match:
        """Record the tool's top choice as rejected when a pin overrode it (negative label)."""
        with self.db.transaction() as c:
            return self._insert(
                c, fingerprint, show_slug, episode_id=episode_id, season=season,
                episode=episode, title=title, status="rejected", source="auto", run_id=run_id,
            )

    def mark_undone(self, run_id: int) -> int:
        """`undo`: the run's confirmed matches become `undone`; returns how many."""
        with self.db.transaction() as c:
            return c.execute(
                "update match set status='undone' where run_id=? and status='confirmed'",
                (run_id,),
            ).rowcount

    def forget(self, fingerprint: str) -> int:
        """Remove every record of a file (media_file, match; rename_op rows keep their text
        but lose the file link); returns matches removed."""
        with self.db.transaction() as c:
            ids = [r[0] for r in c.execute(
                "select id from match where fingerprint=?", (fingerprint,))]
            if ids:  # rename_op.match_id is a FK: detach before deleting
                c.execute(
                    f"update rename_op set match_id=null where match_id in"
                    f" ({','.join('?' * len(ids))})", ids)
            c.execute("delete from match where fingerprint=?", (fingerprint,))
            c.execute("delete from media_file where fingerprint=?", (fingerprint,))
        return len(ids)

    # --- reads ----------------------------------------------------------------------------
    def _select(self, where: str, args: tuple) -> list[Match]:
        with self.db.transaction() as c:
            rows = c.execute(
                f"select {_MATCH_COLS} from match where {where} order by id", args
            ).fetchall()
        return [_match(r) for r in rows]

    def confirmed(self, show_slug: str, season: int | None = None) -> list[Match]:
        """Confirmed matches (episodes and extras) of a show, optionally one season, as
        numbered when recorded. `undone` and `rejected` rows are not included."""
        if season is None:
            return self._select("show_slug=? and status='confirmed'", (show_slug,))
        return self._select(
            "show_slug=? and status='confirmed' and season=?", (show_slug, season))

    def matches_for_file(self, fingerprint: str) -> list[Match]:
        """Every match row of a file, whatever its status."""
        return self._select("fingerprint=?", (fingerprint,))

    def undone(self, show_slug: str) -> list[Match]:
        """Matches whose run was undone: kept for the record, ignored by everything else."""
        return self._select("show_slug=? and status='undone'", (show_slug,))

    def shows(self) -> list[str]:
        """Slugs with at least one confirmed match."""
        with self.db.transaction() as c:
            return [r[0] for r in c.execute(
                "select distinct show_slug from match where status='confirmed' order by 1")]

    def confirmed_for_file(self, fingerprint: str) -> list[Match]:
        return self._select("fingerprint=? and status='confirmed'", (fingerprint,))

    def rejected(self, show_slug: str) -> list[Match]:
        """Negative labels for eval."""
        return self._select("show_slug=? and status='rejected'", (show_slug,))

    def is_confirmed_extra(self, fingerprint: str) -> bool:
        return bool(self._select(
            "fingerprint=? and status='confirmed' and episode_id is null", (fingerprint,)))

    def confirmed_for_episode(self, show_slug: str, episode_id: str) -> list[Match]:
        """Files already confirmed for an episode (duplicate warnings, the prior)."""
        return self._select(
            "show_slug=? and episode_id=? and status='confirmed'", (show_slug, episode_id))

    def history_penalty(
        self, show_slug: str, fingerprints: Iterable[str], episode_ids: list[str],
        penalty: float,
    ) -> list[list[float]]:
        """(files x episodes) penalty matrix for fusion (§11.4): `penalty` where the episode is
        confirmed for a DIFFERENT fingerprint, 0 elsewhere. Soft, never an exclusion."""
        fps = list(fingerprints)
        holders: dict[str, set[str]] = {}
        with self.db.transaction() as c:
            for r in c.execute(
                "select episode_id, fingerprint from match where show_slug=? and"
                " status='confirmed' and episode_id is not null", (show_slug,)
            ):
                holders.setdefault(r[0], set()).add(r[1])
        return [
            [penalty if (holders.get(e, set()) - {fp}) else 0.0 for e in episode_ids]
            for fp in fps
        ]

    def highest_dupe_index(self, show_slug: str, episode_id: str) -> int:
        """Highest `dupe_X` already used for an episode in `rename_op` (0 if none); the folder is
        checked separately by the planner (decision 56)."""
        with self.db.transaction() as c:
            r = c.execute(
                "select max(dupe_index) from rename_op where show_slug=? and episode_id=?",
                (show_slug, episode_id),
            ).fetchone()
        return int(r[0] or 0)
