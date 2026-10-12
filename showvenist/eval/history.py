"""`eval --from-history` (spec §16.1): manifest rows and negative labels from state.db.

Confirmed episode matches become positive labels written as `@<episode id>` (matches keep the
numbering they were recorded with but not the ordering's name), confirmed extras `extra`. A
two-episode file (decision 64) becomes the set label `@<first id>+@<second id>` (decision 50)
when the show's bundle is at hand to find the second episode, else `@<first id>`.
Rejected matches are negative labels: they only add one line to the eval output."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from showvenist.bundle.schema import Bundle
from showvenist.eval.types import FileDecision, ManifestRow, Problem
from showvenist.state.history import History, held_episode_ids


@dataclass
class HistoryLabels:
    rows: list[ManifestRow] = field(default_factory=list)
    # path -> episode ids the file was rejected for (None = rejected as an extra)
    negatives: dict[str, set[str | None]] = field(default_factory=dict)
    problems: list[Problem] = field(default_factory=list)  # files that no longer exist


def history_labels(history: History, shows: Iterable[str] | None = None,
                   bundle_of: Callable[[str], Bundle | None] | None = None) -> HistoryLabels:
    out = HistoryLabels()
    for slug in sorted(set(shows) if shows is not None else history.shows()):
        bundle = bundle_of(slug) if bundle_of is not None else None
        for m in history.confirmed(slug):
            f = history.get_file(m.fingerprint)
            if f is None:
                continue
            label = ("extra" if m.episode_id is None
                     else "+".join(f"@{e}" for e in held_episode_ids(bundle, m)))
            row = ManifestRow(f.last_path, slug, label)
            if not Path(f.last_path).is_file():
                out.problems.append(Problem(row, "file no longer exists at its last known path"))
            else:
                out.rows.append(row)
        for m in history.rejected(slug):
            f = history.get_file(m.fingerprint)
            if f is not None:
                out.negatives.setdefault(f.last_path, set()).add(m.episode_id)
    return out


def negative_line(negatives: dict[str, set[str | None]], decisions: Iterable[FileDecision]) -> str:
    """"decisions equal to a rejected match: n of m", over the files that have a rejected match
    and a decision in this run."""
    seen = hit = 0
    for d in decisions:
        neg = negatives.get(d.path)
        if not neg:
            continue
        seen += 1
        hit += (d.episode_id if d.kind == "episode" else None) in neg
    return f"decisions equal to a rejected match: {hit} of {seen}"
