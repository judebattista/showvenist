"""Refresh diff and history impact (spec §6.8). Synthetic bundles, no network."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from showvenist.bundle.diff import (
    diff_bundles,
    history_impact,
    load_history_impact,
    render_diff,
    staleness_notice,
)
from showvenist.bundle.schema import (
    CastEntry,
    Ordering,
    OrderingEntry,
    Provenance,
    Synopsis,
)
from showvenist.state.db import StateDb
from showvenist.state.history import History
from tests.test_merge import p_ep, primary


def eps(*ks):
    return [(1, i + 1, p_ep(k)) for i, k in enumerate(ks)]


def with_orderings(b, **orderings):
    return b.model_copy(update={"orderings": {**b.orderings, **orderings}})


def ordering(typ, ids, rank=None, name=None):
    return Ordering(source="tmdb", type=typ, type_rank=rank, name=name, entries=[
        OrderingEntry(season=1, episode=i + 1, episode_id=e) for i, e in enumerate(ids)])


def test_identical_is_empty():
    b = primary(eps(0, 1, 2))
    d = diff_bundles(b, b.model_copy(deep=True))
    assert d.empty and render_diff(d).startswith("No changes")


def test_added_removed_and_renumbered():
    old = primary(eps(0, 1, 2))
    new = primary([(1, 1, p_ep(1)), (1, 2, p_ep(0)), (1, 3, p_ep(3))])
    d = diff_bundles(old, new)
    assert [i for i, _ in d.episodes_added] == ["tmdb:103"]
    assert [i for i, _ in d.episodes_removed] == ["tmdb:102"]
    assert d.renumbered["aired"] == [("tmdb:100", "S01E01", "S01E02"),
                                     ("tmdb:101", "S01E02", "S01E01")]
    text = render_diff(d)
    assert "Episodes added (1):" in text and "tmdb:102" in text
    assert "tmdb:100  S01E01 -> S01E02" in text


def test_orderings_added_removed_and_first_ranked_change():
    ids = ["tmdb:100", "tmdb:101"]
    old = with_orderings(primary(eps(0, 1)), dvd_a=ordering("dvd", ids, 1),
                         dvd_b=ordering("dvd", ids, 2), old_only=ordering("absolute", ids))
    new = with_orderings(primary(eps(0, 1)), dvd_a=ordering("dvd", ids, 2),
                         dvd_b=ordering("dvd", ids, 1), fresh=ordering("production", ids))
    d = diff_bundles(old, new)
    assert d.orderings_added == ["fresh"] and d.orderings_removed == ["old_only"]
    assert d.first_ranked == {"dvd": ("dvd_a", "dvd_b")}
    assert "dvd: dvd_a -> dvd_b" in render_diff(d)


def test_field_changes_counts_and_examples():
    old = primary([(1, i + 1, p_ep(i)) for i in range(6)])
    new = old.model_copy(deep=True)
    for i, e in enumerate(new.episodes):
        e.title = f"New {i}"
        e.runtime_min = 50
    new.episodes[0].alt_titles = ["Alt"]
    new.episodes[1].synopses.append(Synopsis(source="tvmaze", text="more"))
    new.episodes[2].synopses[0] = Synopsis(source="tmdb", text="rewritten")
    new.episodes[3].guest_cast = [CastEntry(actor="A", character="B")]
    new.episodes[4].air_date = "2011-01-01"
    new.episodes[5].subtitles = []
    d = diff_bundles(old, new)
    assert d.changes["title"].count == 6 and len(d.changes["title"].examples) == 3
    assert d.changes["runtime_min"].count == 6
    assert d.changes["alt_titles"].count == 1 and d.changes["credits"].count == 1
    assert d.changes["synopses"].count == 2
    assert d.changes["air_date"].count == 1
    text = render_diff(d)
    assert "Title changed (6):" in text and "… and 3 more" in text


def test_alt_titles_added_and_removed():
    old = primary(eps(0))
    old.episodes[0].alt_titles = ["Gone", "Kept"]
    new = old.model_copy(deep=True)
    new.episodes[0].alt_titles = ["Kept", "Fresh"]
    d = diff_bundles(old, new)
    assert d.changes["alt_titles"].count == 1
    assert "+ 'Fresh'; - 'Gone'" in d.changes["alt_titles"].examples[0]
    new.episodes[0].alt_titles = []
    assert "- 'Gone', 'Kept'" in diff_bundles(old, new).changes["alt_titles"].examples[0]


def test_subtitles_added():
    from showvenist.bundle.schema import SubtitleRef

    old = primary(eps(0, 1))
    new = old.model_copy(deep=True)
    new.episodes[0].subtitles = [SubtitleRef(lang="en", path="subs/a.srt", source="localsubs")]
    assert diff_bundles(old, new).subtitles_added == 1


def test_conflicts_new_resolved_and_show_level():
    c1 = {"kind": "split", "season": "1", "primary": "tmdb:100", "secondary": "tvmaze:5"}
    c2 = {"kind": "air_date", "season": "1", "primary": "tmdb:101", "secondary": "tvmaze:6"}
    old = primary(eps(0, 1))
    old = old.model_copy(update={"provenance": [Provenance(adapter="tvmaze", conflict=c1)]})
    new = old.model_copy(deep=True)
    new.provenance = [Provenance(adapter="tvmaze", conflict=c2)]
    new.show.in_production = True
    new.show.ids.tvmaze = 77
    d = diff_bundles(old, new)
    assert len(d.conflicts_new) == 1 and "air_date" in d.conflicts_new[0]
    assert len(d.conflicts_resolved) == 1 and "split" in d.conflicts_resolved[0]
    assert "in_production: False -> True" in d.show and "ids.tvmaze: None -> 77" in d.show


def test_long_lists_are_capped():
    old = primary(eps(0))
    new = primary([(1, 1, p_ep(0))])
    new.episodes += [p_ep(i % 8).model_copy(update={"id": f"tmdb:9{i:02d}"}) for i in range(20)]
    text = render_diff(diff_bundles(old, new))
    assert "Episodes added (20):" in text and "… and 12 more" in text


# --- history impact -------------------------------------------------------------------------


@pytest.fixture
def hist():
    h = History(StateDb(":memory:"))

    def confirm(fp, eid, s, e):
        h.touch_file(fp, size=1, path=f"/lib/{fp}.mkv")
        h.confirm(fp, "testshow-2010", episode_id=eid, season=s, episode=e, title=None,
                  source="auto", run_id=None)

    confirm("f1", "tmdb:100", 1, 1)   # renumbered
    confirm("f2", "tmdb:101", 1, 2)   # removed
    confirm("f3", "tmdb:102", 1, 3)   # unchanged
    h.touch_file("f4", size=1, path="/lib/extra.mkv")
    h.confirm("f4", "testshow-2010", episode_id=None, season=None, episode=None, title=None,
              source="manual", run_id=None)
    return h


def test_history_impact_removed_and_renumbered_and_read_only(hist):
    old = primary(eps(0, 1, 2))
    new = primary([(1, 1, p_ep(5)), (1, 2, p_ep(0)), (1, 3, p_ep(2))])
    rows = history_impact(old, new, hist, "testshow-2010")
    got = {(r.path, r.episode_id, r.old, r.new) for r in rows}
    assert got == {("/lib/f1.mkv", "tmdb:100", "S01E01", "S01E02"),
                   ("/lib/f2.mkv", "tmdb:101", "S01E02", "removed")}
    assert len(hist.confirmed("testshow-2010")) == 4  # nothing changed
    text = render_diff(diff_bundles(old, new), rows)
    assert "History impact (2 confirmed matches):" in text and "/lib/f1.mkv" in text


def test_history_impact_unchanged_when_another_ordering_keeps_the_number(hist):
    ids = ["tmdb:100", "tmdb:101", "tmdb:102"]
    old = with_orderings(primary(eps(0, 1, 2)), dvd=ordering("dvd", ids))
    new = with_orderings(primary([(1, 1, p_ep(1)), (1, 2, p_ep(0)), (1, 3, p_ep(2))]),
                         dvd=ordering("dvd", ids))
    rows = history_impact(old, new, hist, "testshow-2010")
    # f1 and f2 were recorded as S01E01/S01E02, which aired AND dvd gave; dvd still does
    assert rows == []


def test_load_history_impact_without_db_or_outdated(tmp_path):
    b = primary(eps(0))
    rows, note = load_history_impact(b, b, "x", tmp_path / "state.db")
    assert rows == [] and "no history" in note
    assert "History impact: no history" in render_diff(diff_bundles(b, b), rows, note)
    import sqlite3

    p = tmp_path / "old.db"
    con = sqlite3.connect(p)
    con.execute("PRAGMA user_version=1")
    con.commit()
    con.close()
    rows, note = load_history_impact(b, b, "x", p)
    assert rows == [] and "not checked" in note


def test_load_history_impact_newer_schema_is_a_note(tmp_path):
    import sqlite3

    b = primary(eps(0))
    p = tmp_path / "new.db"
    con = sqlite3.connect(p)
    con.execute("PRAGMA user_version=999")
    con.commit()
    con.close()
    rows, note = load_history_impact(b, b, "x", p)
    assert rows == [] and "not checked" in note and "999" in note


OVERRIDES = """\
new_episodes:
  "custom:x": { title: "Extra half" }
orderings:
  disk:
    - { season: 1, episode: 1, episode_id: "tmdb:100" }
    - { season: 1, episode: 2, episode_id: "custom:x" }
"""


def test_load_history_impact_applies_overrides_to_both_bundles(tmp_path):
    db = StateDb(tmp_path / "state.db")
    h = History(db)
    h.touch_file("f1", size=1, path="/lib/a.mkv")
    h.confirm("f1", "testshow-2010", episode_id="custom:x", season=1, episode=2, title=None,
              source="manual", run_id=None)
    db.close()
    ov = tmp_path / "overrides.yaml"
    ov.write_text(OVERRIDES)
    old, new = primary(eps(0, 1)), primary(eps(0, 1))
    rows, note = load_history_impact(old, new, "testshow-2010", tmp_path / "state.db")
    assert [r.new for r in rows] == ["removed"]  # without overrides custom:x looks removed
    rows, note = load_history_impact(old, new, "testshow-2010", tmp_path / "state.db",
                                     overrides=ov)
    assert rows == [] and note is None
    # the refreshed bundle drops tmdb:100, so the override ordering renumbers nothing but
    # disk keeps custom:x at S01E02: still unaffected
    rows, _ = load_history_impact(old, primary(eps(1)), "testshow-2010",
                                  tmp_path / "state.db", overrides=ov)
    assert rows == []
    # overrides that no longer load over the refreshed bundle: compared without, with a note
    clash = primary(eps(0, 1))
    clash.episodes.append(p_ep(7).model_copy(update={"id": "custom:x"}))
    rows, note = load_history_impact(old, clash, "testshow-2010", tmp_path / "state.db",
                                     overrides=ov)
    assert "does not apply to the refreshed bundle" in note
    assert "History impact: overrides.yaml does not apply" in render_diff(
        diff_bundles(old, clash), rows, note)


def test_load_history_impact_reads_real_db(tmp_path):
    db = StateDb(tmp_path / "state.db")
    h = History(db)
    h.touch_file("f1", size=1, path="/lib/a.mkv")
    h.confirm("f1", "testshow-2010", episode_id="tmdb:100", season=1, episode=1, title=None,
              source="auto", run_id=None)
    db.close()
    old, new = primary(eps(0, 1)), primary([(1, 1, p_ep(1))])
    rows, note = load_history_impact(old, new, "testshow-2010", tmp_path / "state.db")
    assert note is None and [r.new for r in rows] == ["removed"]


def test_staleness_notice():
    b = primary(eps(0))
    b.show.in_production = True
    b.provenance = [Provenance(adapter="tmdb", fetched_at="2026-01-01T00:00:00Z")]
    now = datetime(2026, 3, 1, tzinfo=UTC)
    n = staleness_notice(b, "testshow-2010", 30, now)
    assert n and "ingest testshow-2010 --refresh --diff" in n
    assert staleness_notice(b, "x", 90, now) is None
    b.show.in_production = False
    assert staleness_notice(b, "x", 30, now) is None
