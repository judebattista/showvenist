"""Title-order tie-breaker (spec §11.5, decision 63): title index parsing, the strict verdict,
the history query and the block bonus. Synthetic paths and matrices only."""

import itertools

import numpy as np
import pytest

from showvenist.assign.blocks import block_confidence, solve_blocks
from showvenist.assign.titleorder import TitleOrderVerdict, history_groups, judge
from showvenist.assign.types import BlockParams, DiscSpec
from showvenist.hierarchy.parse import parse_title_index
from showvenist.scoring.types import level_of
from showvenist.state.db import StateDb
from showvenist.state.history import History

# ---- parsing ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "want"),
    [
        ("B1_t03.mkv", 3),
        ("title_t00.mkv", 0),
        ("Burn Notice_t12.mkv", 12),
        ("A1_T07.MKV", 7),
        ("disc_t123.mkv", 123),
        ("show_s01e02.mkv", None),
        ("t03.mkv", None),
        ("B1_t03_extra.mkv", None),
        ("B1_t3.mkv", None),
    ],
)
def test_parse_title_index(name, want):
    assert parse_title_index(name) == want


# ---- verdict ---------------------------------------------------------------------------------


def test_all_concordant_is_on_and_counts_discs():
    v = judge({"Disc 1": [(0, 0), (1, 1), (2, 2)], "Disc 2": [(3, 5), (5, 6)]})
    assert v == TitleOrderVerdict(True, agree=2)
    assert v.line() == "title order used as a tie-breaker; 2 discs agree"


def test_one_discordant_pair_turns_it_off():
    v = judge({"Disc 1": [(0, 0), (1, 1)], "Disc 2": [(2, 4), (3, 3)], "Disc 3": [(0, 7)]})
    assert not v.use and v.against == "Disc 2" and v.reverse
    assert v.line() == "title order not used; Disc 2 runs in reverse"


def test_mixed_disc_is_out_of_order_not_reverse():
    v = judge({"Disc 1": [(0, 0), (1, 2), (2, 1)]})
    assert not v.use and not v.reverse
    assert v.line() == "title order not used; Disc 1 is out of title order"


def test_no_pairs_is_on():
    v = judge({"Disc 1": [(4, 2)], "Disc 2": []})
    assert v == TitleOrderVerdict(True, agree=0)
    assert v.line() == "title order used as a tie-breaker; no evidence yet"
    assert judge({}).use


def test_same_episode_or_title_makes_no_pair():
    assert judge({"Disc 1": [(0, 3), (1, 3), (2, 3)]}) == TitleOrderVerdict(True, 0)


def test_pairs_never_span_groups():
    # each disc alone is concordant; across discs title order restarts
    assert judge({"Disc 1": [(5, 0), (6, 1)], "Disc 2": [(0, 2), (1, 3)]}).use


def test_history_groups_by_folder_and_skip_files_without_index():
    moves = [
        ("/lib/Show/Season 1/Disc 2/B1_t01.mkv", 1, 5),
        ("/lib/Show/Season 1/Disc 2/B1_t00.mkv", 1, 4),
        ("/lib/Show/Season 1/Disc 1/title_t00.mkv", 1, 1),
        ("/lib/Show/Season 1/Disc 1/renamed.mkv", 1, 2),  # no index: ignored
    ]
    g = history_groups(moves)
    assert g == {
        "Season 1/Disc 1": [(0, (1, 1))],
        "Season 1/Disc 2": [(1, (1, 5)), (0, (1, 4))],
    }
    assert judge(g) == TitleOrderVerdict(True, 1)


def test_history_only_discordant():
    g = history_groups([("/l/S/Season 2/D1/x_t00.mkv", 2, 3), ("/l/S/Season 2/D1/x_t01.mkv", 2, 2)])
    v = judge(g)
    assert not v.use and v.line() == "title order not used; Season 2/D1 runs in reverse"


# ---- history query ---------------------------------------------------------------------------


def _move(h, db, fp, old, ep, *, show="s", status="done", undone=False, kind="episode",
          extra=False, run=None):
    h.touch_file(fp, size=1, path=old)
    m = h.confirm(fp, show, episode_id=None if extra else f"e{ep}", season=None if extra else 1,
                  episode=None if extra else ep, title=None, source="auto", run_id=None)
    with db.transaction() as c:
        c.execute(
            "insert into rename_op (run_id, fingerprint, kind, old_path, new_path, status,"
            " match_id, undone_at) values (?,?,?,?,?,?,?,?)",
            (run, fp, kind, old, old + ".new", status, m.id, "x" if undone else None))
    return m


def test_confirmed_moves_query():
    db = StateDb(":memory:")
    h = History(db)
    run = h.begin_run("apply", "").id
    _move(h, db, "1:a", "/d/a_t00.mkv", 1, run=run)
    _move(h, db, "1:b", "/d/a_t01.mkv", 2, run=run, undone=True)
    _move(h, db, "1:c", "/d/a_t02.mkv", 3, run=run, status="planned")
    _move(h, db, "1:d", "/d/a_t03.mkv", 4, run=run, kind="sidecar")
    _move(h, db, "1:e", "/d/a_t04.mkv", 0, run=run, kind="extra", extra=True)
    _move(h, db, "1:i", "/d/a_t08.mkv", 7, run=run, kind="duplicate")  # a copy: no position
    _move(h, db, "1:f", "/d/a_t05.mkv", 6, run=run, show="other")
    _move(h, db, "1:g", "/d/a_t06.mkv", 7, run=run)
    # a match that is no longer confirmed (re-assigned) drops out
    _move(h, db, "1:h", "/d/a_t07.mkv", 8, run=run)
    h.confirm("1:h", "s", episode_id="e9", season=1, episode=9, title=None, source="manual",
              run_id=None)
    assert h.confirmed_moves("s") == [("/d/a_t00.mkv", 1, 1), ("/d/a_t06.mkv", 1, 7)]


# ---- block bonus -----------------------------------------------------------------------------

W = 0.3
ON = BlockParams(title_order_weight=W)
OFF = BlockParams(title_order_weight=0.0)


def _disc(n, titles, key="D1", number=1, start=0):
    return DiscSpec(key, tuple(range(start, start + n)), number, titles=tuple(titles))


def test_tie_breaks_in_title_order():
    # files 0 and 1 tie on columns 1 and 2; file 0 is t01, file 1 is t02
    S = np.full((3, 3), -5.0)
    S[2, 0] = 8.0  # t00 -> E1, clear
    S[0, 1] = S[0, 2] = S[1, 1] = S[1, 2] = 6.0
    for order in ([0, 1, 2], [1, 0, 2]):  # whatever the row order, title order decides
        titles = [1, 2, 0] if order == [0, 1, 2] else [2, 1, 0]
        sol = solve_blocks(S, [_disc(3, titles)], params=ON)
        first = order[0]
        assert sol.column[first] == 1 and sol.column[order[1]] == 2 and sol.column[2] == 0
        confs = block_confidence(S, [_disc(3, titles)], sol, params=ON)
        assert confs[0].delta == pytest.approx(2 * W) and confs[0].level == "low"


def test_ranks_are_among_matched_files_so_extras_do_not_shift_them():
    # t00, t01 are play-all extras (unmatched); t02..t04 are episodes with a tie on E2/E3
    S = np.full((5, 3), -9.0)
    S[2, 0] = 8.0
    S[3, 1] = S[3, 2] = S[4, 1] = S[4, 2] = 6.0
    titles = [0, 1, 2, 3, 4]
    for flip in (False, True):
        T = list(titles)
        if flip:
            T[3], T[4] = T[4], T[3]
        sol = solve_blocks(S, [_disc(5, T)], params=ON)
        lo, hi = (3, 4) if T[3] < T[4] else (4, 3)
        assert sol.column[:3] == [-1, -1, 0]
        assert sol.column[lo] == 1 and sol.column[hi] == 2


def test_out_of_order_member_still_breaks_ties():
    # a confident file with a high title index sits at E1 (an alternate cut ripped late);
    # the E2/E3 tie still goes in title order
    S = np.full((3, 3), -9.0)
    S[2, 0] = 9.0
    S[0, 1] = S[0, 2] = S[1, 1] = S[1, 2] = 6.0
    sol = solve_blocks(S, [_disc(3, [1, 2, 3])], params=ON)
    assert sol.column == [1, 2, 0]
    sol = solve_blocks(S, [_disc(3, [2, 1, 3])], params=ON)
    assert sol.column == [2, 1, 0]


def test_content_beats_title_order():
    S = np.full((2, 2), -5.0)
    S[0, 1] = S[1, 0] = 7.0  # content: crossed
    S[0, 0] = S[1, 1] = 6.0  # title order would cost 2 nats of content for 0.6 of bonus
    sol = solve_blocks(S, [_disc(2, [0, 1])], params=ON)
    assert sol.column == [1, 0]


def test_weight_zero_disables_and_titles_absent_is_off():
    S = np.full((2, 2), 6.0)
    for params, titles in ((OFF, (1, 0)), (ON, None)):
        d = DiscSpec("D1", (0, 1), 1, titles=titles)
        sol = solve_blocks(S, [d], params=params)
        confs = block_confidence(S, [d], sol, params=params)
        assert [c.delta for c in confs] == [0.0, 0.0]


def test_files_without_index_do_not_move():
    S = np.full((3, 3), 6.0)
    d = DiscSpec("D1", (0, 1, 2), 1, titles=(None, 5, 4))
    base = solve_blocks(S, [DiscSpec("D1", (0, 1, 2), 1)], params=ON)
    sol = solve_blocks(S, [d], params=ON)
    assert sol.column[0] == base.column[0]
    a, b = sol.column[1], sol.column[2]
    assert b < a  # t04 before t05


def test_bonus_never_changes_block_membership_or_lifts_a_tie_to_high():
    """With evidence tied everywhere, the bonus alone gives Δ ≤ 2w (an adjacent swap), far
    below high (4.6). Blocks and the set of claimed episodes match the solve without it."""
    rng = np.random.default_rng(7)
    for n_files, extra_cols in itertools.product((2, 3, 5, 7), (0, 2)):
        E = n_files + extra_cols
        S = np.full((n_files, E), 6.0)
        titles = list(rng.permutation(n_files) + 10)
        d = _disc(n_files, titles)
        sol = solve_blocks(S, [d], params=ON)
        off = solve_blocks(S, [DiscSpec("D1", d.files, 1)], params=OFF)
        assert sorted(sol.column) == sorted(off.column) and sol.blocks == off.blocks
        confs = block_confidence(S, [d], sol, params=ON)
        assert max(c.delta for c in confs) <= 2 * W + 1e-9
        assert all(level_of(c.delta) != "high" for c in confs)
        # matched files run in title order
        matched = sorted((c, titles[i]) for i, c in enumerate(sol.column) if c >= 0)
        assert [t for _, t in matched] == sorted(t for _, t in matched)


def test_cannot_reach_high_with_two_discs_and_noise():
    """Near-ties (noise ≪ 1 nat) on two discs: no file reaches high under the bonus unless it
    is high without it."""
    rng = np.random.default_rng(3)
    for _ in range(30):
        S = np.full((8, 10), -6.0)
        S[:4, :5] = 5.0 + rng.normal(0, 0.05, (4, 5))
        S[4:, 4:] = 5.0 + rng.normal(0, 0.05, (4, 6))
        t1, t2 = list(rng.permutation(4)), list(rng.permutation(4))
        on = [_disc(4, t1), _disc(4, t2, key="D2", number=2, start=4)]
        off = [DiscSpec(x.key, x.files, x.number) for x in on]
        s_on = solve_blocks(S, on, params=ON)
        s_off = solve_blocks(S, off, params=ON)
        c_on = block_confidence(S, on, s_on, params=ON)
        c_off = block_confidence(S, off, s_off, params=ON)
        for a, b in zip(c_on, c_off, strict=True):
            assert a.level != "high" or b.level == "high"
            assert a.delta <= b.delta + 2 * W * 4 + 1e-9


# ---- session end to end ----------------------------------------------------------------------


def _session_case(tmp_path, disc2_reversed=False, history=None):
    from tests.test_session_blocks import make_season, plain_bundle, session_for

    season = make_season(tmp_path, {"Disc 1": ["a_t00.mkv", "a_t01.mkv", "a_t02.mkv"],
                                    "Disc 2": ["b_t00.mkv", "b_t01.mkv", "b_t02.mkv"]})
    d2 = ["e6", "e5", "e4"] if disc2_reversed else ["e4", "e5", "e6"]
    table = {
        "a_t00.mkv": {"e1": 12.0},
        # a near tie that content alone resolves the wrong way (by 0.2 nats)
        "a_t01.mkv": {"e2": 6.0, "e3": 6.1},
        "a_t02.mkv": {"e2": 6.1, "e3": 6.0},
        **{f"b_t0{k}.mkv": {d2[k]: 12.0} for k in range(3)},
    }
    return session_for(season, plain_bundle(6), table, tmp_path, history=history)


def _pick(sess, sol, name):
    from tests.test_session_blocks import assigned

    return assigned(sol, name)


def test_session_title_order_breaks_the_tie(tmp_path):
    sess = _session_case(tmp_path)
    sol = sess.solve()
    assert sol.title_order == TitleOrderVerdict(True, agree=1)
    assert _pick(sess, sol, "a_t01.mkv") == "e2" and _pick(sess, sol, "a_t02.mkv") == "e3"
    lvl = {f.path.name: c.level for f, c in zip(sol.files, sol.confidences, strict=True)}
    assert lvl["a_t01.mkv"] == lvl["a_t02.mkv"] == "low"
    assert lvl["b_t01.mkv"] == "high"
    line = "title order used as a tie-breaker; 1 disc agrees"
    assert sess.explain(next(f.path for f in sol.files if f.path.name == "a_t01.mkv")) \
        .title_order == line
    notes = [n for n in sess.report().notices if "title order" in n]
    assert sorted(notes) == [f"a_t01.mkv: {line}", f"a_t02.mkv: {line}"]


def test_session_reversed_disc_turns_it_off(tmp_path):
    sess = _session_case(tmp_path, disc2_reversed=True)
    sol = sess.solve()
    assert not sol.title_order.use and sol.title_order.reverse
    # content decides: the 0.2-nat preference stands
    assert _pick(sess, sol, "a_t01.mkv") == "e3" and _pick(sess, sol, "a_t02.mkv") == "e2"
    detail = sess.explain(next(f.path for f in sol.files if f.path.name == "a_t01.mkv"))
    assert detail.title_order == "title order not used; Disc 2 runs in reverse"
    assert not [n for n in sess.report().notices if "title order" in n]


def test_session_history_discordance_turns_it_off(tmp_path):
    db = StateDb(":memory:")
    h = History(db)
    run = h.begin_run("apply", "").id
    slug = "synth-show-2000"
    _move(h, db, "9:a", "/old/Synth Show/Season 2/Disc 1/x_t00.mkv", 5, show=slug, run=run)
    _move(h, db, "9:b", "/old/Synth Show/Season 2/Disc 1/x_t01.mkv", 4, show=slug, run=run)
    sess = _session_case(tmp_path, history=h)
    assert sess.show_slug == slug
    sol = sess.solve()
    assert not sol.title_order.use
    assert sol.title_order.against == "Season 2/Disc 1 (history)"
    assert _pick(sess, sol, "a_t01.mkv") == "e3"


def test_majority_of_groups_keeps_it_on():
    ok = [(0, 0), (1, 1)]
    bad = [(0, 1), (1, 0)]
    g = {f"D{i}": ok for i in range(5)} | {"Pilot disc": bad, "Disc 4": bad}
    v = judge(g)
    assert v.use and (v.agree, v.disagree, v.against) == (5, 2, "Pilot disc")
    assert v.line() == ("title order used as a tie-breaker; 5 of 7 discs agree "
                        "(Pilot disc does not)")


def test_two_thirds_is_the_threshold():
    ok, bad = [(0, 0), (1, 1)], [(0, 1), (1, 0)]
    assert judge({"A": ok, "B": ok, "C": bad}).use  # 2 of 3
    assert not judge({"A": ok, "B": bad}).use  # 1 of 2
    assert not judge({"A": ok, "B": bad, "C": bad}).use  # 1 of 3
