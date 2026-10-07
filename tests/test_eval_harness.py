import pytest
from scipy.stats import beta

from showvenist.bundle.schema import Ordering, OrderingEntry
from showvenist.eval.harness import (
    clopper_pearson_upper,
    compute_metrics,
    evaluate,
    find_ordering,
    resolve_labels,
)
from showvenist.eval.types import FileDecision, ManifestRow, ResolvedLabel
from tests.synth import make_bundle

SHOW = "synth"


def firefly_like():
    """Aired order 100..105 as S01E01..06; an "intended-order" custom ordering puts 105 first."""
    b = make_bundle(6)
    ids = ["tmdb:105", "tmdb:100", "tmdb:101", "tmdb:102", "tmdb:103", "tmdb:104"]
    b.orderings["intended-order"] = Ordering(
        source="overrides", type="custom",
        entries=[OrderingEntry(season=1, episode=i + 1, episode_id=e) for i, e in enumerate(ids)],
    )
    b.orderings["dvd-release"] = Ordering(
        source="tmdb", type="dvd",
        entries=[OrderingEntry(season=1, episode=1, episode_id="tmdb:100"),
                 OrderingEntry(season=1, episode=1, episode_id="tmdb:101"),  # ambiguous
                 OrderingEntry(season=1, episode=3, episode_id="tmdb:999")],  # unknown episode
    )
    return b


def row(path, label, order="aired", show=SHOW):
    return ManifestRow(path, show, label, order)


def ep(path, eid, level="high", duplicate=False):
    return FileDecision(path, "episode", eid, level, duplicate=duplicate)


def ex(path, level="high"):
    return FileDecision(path, "extra", None, level)


def lab(path, kind, eid=None, dup=None, label=None):
    return ResolvedLabel(row(path, label or ("S01E01" if kind == "episode" else kind)),
                         kind, eid, dup)


# ---- Clopper-Pearson ------------------------------------------------------------------------

def test_clopper_pearson_known_values():
    assert clopper_pearson_upper(0, 299) == pytest.approx(0.00997, abs=5e-5)
    assert clopper_pearson_upper(0, 299) <= 0.01 < clopper_pearson_upper(0, 298)
    assert clopper_pearson_upper(1, 474) == pytest.approx(0.00997, abs=5e-5)
    # the spec rounds this threshold to 474
    assert clopper_pearson_upper(1, 473) <= 0.01 < clopper_pearson_upper(1, 472)
    assert clopper_pearson_upper(0, 60) == pytest.approx(0.0487, abs=5e-4)
    for n in (1, 10, 299):
        assert clopper_pearson_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n))
    assert clopper_pearson_upper(3, 50) == pytest.approx(beta.ppf(0.95, 4, 47))
    assert clopper_pearson_upper(0, 0) == 1.0
    assert clopper_pearson_upper(5, 5) == 1.0 and clopper_pearson_upper(6, 5) == 1.0


# ---- label resolution -----------------------------------------------------------------------

def test_resolution_through_ordering():
    b = firefly_like()
    rows = [row("/a.mkv", "S01E01"), row("/b.mkv", "s01e01", "intended-order"),
            row("/c.mkv", "EXTRA"), row("/d.mkv", "Episode?"), row("/e.mkv", "S01E02", "Aired")]
    labels, problems = resolve_labels(rows, {SHOW: b})
    assert problems == []
    assert [(r.kind, r.episode_id) for r in labels] == [
        ("episode", "tmdb:100"), ("episode", "tmdb:105"), ("extra", None),
        ("episode?", None), ("episode", "tmdb:101")]
    assert all(r.duplicate_set is None for r in labels)


def test_find_ordering_by_unique_type():
    b = firefly_like()
    assert find_ordering(b, "dvd") is b.orderings["dvd-release"]
    assert find_ordering(b, "nope") is None


def test_unresolved_labels_become_problems():
    b = firefly_like()
    rows = [row("/1", "S01E07"), row("/2", "S01E01", "no-such-order"), row("/3", "S01E01", "dvd"),
            row("/4", "S01E03", "dvd"), row("/5", "pilot"), row("/6", "S01E01", show="other")]
    labels, problems = resolve_labels(rows, {SHOW: b})
    assert labels == []
    assert [p.row.path for p in problems] == ["/1", "/2", "/3", "/4", "/5", "/6"]
    reasons = [p.reason for p in problems]
    assert "no entry" in reasons[0] and "unknown ordering" in reasons[1]
    assert "2 episodes" in reasons[2] and "unknown episode" in reasons[3]
    assert "unparseable" in reasons[4] and "no bundle" in reasons[5]


def test_duplicate_sets_across_orderings():
    b = firefly_like()
    rows = [row("/a.mkv", "S01E01", "intended-order"), row("/a2.mkv", "S01E06"),
            row("/b.mkv", "S01E02")]
    labels, _ = resolve_labels(rows, {SHOW: b})
    assert [r.duplicate_set for r in labels] == ["tmdb:105", "tmdb:105", None]


# ---- metrics --------------------------------------------------------------------------------

def test_empty_inputs():
    r = compute_metrics([], [])
    assert r.overall.files == 0 and r.overall.top1 is None
    assert r.overall.precision_high is None and r.overall.coverage_high is None
    assert r.overall.unmatched_precision is None and r.overall.reliability == {}
    assert r.per_show == {} and r.per_season == {} and r.problems == []
    assert evaluate([], {}, []).overall.files == 0


def test_basic_metrics_and_extras_errors():
    labels = [lab("/1", "episode", "tmdb:100"), lab("/2", "episode", "tmdb:101", label="S01E02"),
              lab("/3", "episode", "tmdb:102", label="S01E03"), lab("/4", "extra"),
              lab("/5", "extra"), lab("/6", "episode", "tmdb:103", label="S01E04")]
    decisions = [ep("/1", "tmdb:100"),            # right, high
                 ex("/2"),                         # episode sent to extras at high: error
                 ep("/3", "tmdb:104", "medium"),  # wrong, medium
                 ex("/4"),                         # identified extra: right, high
                 ep("/5", "tmdb:105", "low"),      # extra matched as an episode, low
                 ep("/6", "tmdb:103", "medium"),  # right, medium
                 ep("/unlabeled", "tmdb:100")]    # ignored
    m = compute_metrics(labels, decisions).overall
    assert m.files == 6
    assert m.top1 == pytest.approx(3 / 6)
    assert m.coverage_high == pytest.approx(3 / 6)
    assert (m.precision_high.errors, m.precision_high.n) == (1, 3)
    assert m.precision_high.upper_95 == pytest.approx(beta.ppf(0.95, 2, 2))
    assert m.reliability == {"high": pytest.approx(2 / 3), "medium": 0.5, "low": 0.0}
    assert m.unmatched_precision == 0.5  # /2 and /4 decided extra, only /4 is one
    assert m.unmatched_recall == 0.5      # /4 found, /5 missed


def test_extra_matched_at_high_is_precision_error():
    m = compute_metrics([lab("/x", "extra")], [ep("/x", "tmdb:100")]).overall
    assert (m.precision_high.errors, m.precision_high.n) == (1, 1)
    assert m.unmatched_precision is None and m.unmatched_recall == 0.0


def test_no_decision_is_a_problem():
    r = compute_metrics([lab("/1", "episode", "tmdb:100"), lab("/2", "extra")], [ex("/2")])
    assert [(p.row.path, p.reason) for p in r.problems] == [("/1", "no decision")]
    assert r.overall.files == 1 and r.overall.top1 == 1.0


def test_duplicate_sets_out_of_precision():
    labels = [lab("/a", "episode", "tmdb:100", "tmdb:100"),
              lab("/a2", "episode", "tmdb:100", "tmdb:100"),
              lab("/b", "episode", "tmdb:101", "tmdb:101"),
              lab("/b2", "episode", "tmdb:101", "tmdb:101")]
    decisions = [ep("/a", "tmdb:100"), ep("/a2", "tmdb:100", duplicate=True),
                 ep("/b", "tmdb:101"), ep("/b2", "tmdb:104")]  # /b2 wrong at high
    m = compute_metrics(labels, decisions).overall
    assert m.precision_high is None  # all four kept out of precision
    assert m.top1 == 0.75 and m.coverage_high == 1.0


def test_episode_unknown_rules():
    labels = [lab("/held", "episode", "tmdb:100"),
              lab("/held2", "episode", "tmdb:101", label="S01E02"),
              lab("/q1", "episode?"), lab("/q2", "episode?"), lab("/q3", "episode?"),
              lab("/q4", "episode?"), lab("/q5", "episode?"), lab("/q6", "episode?"),
              lab("/q7", "episode?", dup="tmdb:101")]
    decisions = [ep("/held", "tmdb:100"), ep("/held2", "tmdb:101"),
                 ex("/q1"),                                # extra at high: provable error
                 ep("/q2", "tmdb:100"),                    # slot held by /held: provable error
                 ep("/q3", "tmdb:100", duplicate=True),   # flagged duplicate of it: exception
                 ep("/q4", "tmdb:103"),                    # free slot: unverifiable
                 ex("/q5", "medium"),                      # below high: unverifiable
                 ep("/q6", "tmdb:100", "low"),             # held slot but not high: unverifiable
                 ep("/q7", "tmdb:101")]                    # marked as the same set: exception
    m = compute_metrics(labels, decisions).overall
    assert m.files == 9
    # top-1, coverage and reliability see only the two labeled files
    assert m.top1 == 1.0 and m.coverage_high == 1.0 and m.reliability == {"high": 1.0}
    # the two labeled files plus the two provable errors, which are verifiable decisions
    assert (m.precision_high.errors, m.precision_high.n) == (2, 4)
    assert m.precision_high.upper_95 == pytest.approx(0.9024, abs=1e-3)
    assert m.episode_unknown_files == 7
    assert m.episode_unknown_high_as_episode == 4  # q2, q3, q4, q7
    assert m.episode_unknown_flagged == 2           # q5, q6
    assert m.episode_unknown_provable_errors == 2
    # `episode?` files are not-extras: q1 and q5 are false positives, no extras to recall
    assert m.unmatched_precision == 0.0 and m.unmatched_recall is None


def test_episode_unknown_held_slot_is_per_show():
    labels = [lab("/held", "episode", "tmdb:100"),
              ResolvedLabel(row("/q", "episode?", show="other"), "episode?")]
    m = compute_metrics(labels, [ep("/held", "tmdb:100"), ep("/q", "tmdb:100")]).overall
    assert m.episode_unknown_provable_errors == 0 and m.precision_high.errors == 0


def test_only_episode_unknown_files():
    m = compute_metrics([lab("/q", "episode?")], [ex("/q")]).overall
    assert m.top1 is None and m.coverage_high is None
    assert (m.precision_high.errors, m.precision_high.n, m.precision_high.upper_95) == (1, 1, 1.0)


def test_per_show_and_per_season_keys():
    b = make_bundle(3)
    b2 = make_bundle(2, season=2)
    b2.orderings["chrono"] = Ordering(source="overrides", type="custom", entries=[
        OrderingEntry(season=3, episode=1, episode_id="tmdb:100")])
    bundles = {SHOW: b, "two": b2}
    rows = [row("/Show/Season 1/a.mkv", "S01E01"),
            row("/Show/Season 1/t04.mkv", "extra"),
            row("/Show/S1/t05.mkv", "episode?"),  # no Season folder: per show only
            row("/Two/Season 2/x.mkv", "S03E01", "chrono", show="two"),  # season from ordering
            row("/Two/Season 02/y.mkv", "S02E02", show="two")]
    r = evaluate(rows, bundles, [ep("/Show/Season 1/a.mkv", "tmdb:100"),
                                 ex("/Show/Season 1/t04.mkv"), ex("/Show/S1/t05.mkv", "low"),
                                 ep("/Two/Season 2/x.mkv", "tmdb:100"),
                                 ep("/Two/Season 02/y.mkv", "tmdb:100", "medium")])
    assert r.problems == []
    assert set(r.per_show) == {SHOW, "two"}
    assert set(r.per_season) == {"synth-S01", "two-S02", "two-S03"}
    assert r.per_season["synth-S01"].files == 2 and r.per_show[SHOW].files == 3
    assert r.per_show[SHOW].episode_unknown_flagged == 1
    assert r.per_season["two-S03"].top1 == 1.0 and r.per_season["two-S02"].top1 == 0.0
    assert r.overall.files == 5 and r.overall.top1 == pytest.approx(3 / 4)


def test_evaluate_merges_problems():
    b = firefly_like()
    r = evaluate([row("/1", "S01E09"), row("/2", "extra")], {SHOW: b}, [])
    assert [p.reason.split()[0] for p in r.problems] == ["S01E09", "no"]


def test_extra_multi_label_parses_as_an_extra_with_the_multi_flag():
    from showvenist.eval.harness import resolve_labels
    from showvenist.eval.types import ManifestRow
    from tests.synth import make_bundle

    rows = [ManifestRow("/a", "synth-show-2000", "extra-multi"),
            ManifestRow("/b", "synth-show-2000", "Extra-Multi"),
            ManifestRow("/c", "synth-show-2000", "extra")]
    labels, problems = resolve_labels(rows, {"synth-show-2000": make_bundle()})
    assert not problems
    assert [(x.kind, x.multi) for x in labels] == [
        ("extra", True), ("extra", True), ("extra", False)]


def test_extra_multi_is_scored_as_an_extra_and_reported_separately():
    from showvenist.eval.harness import compute_metrics

    def multi(path):
        return ResolvedLabel(row(path, "extra-multi"), "extra", multi=True)

    labels = [multi("/m1"), multi("/m2"), lab("/e1", "extra")]
    m = compute_metrics(labels, [ex("/m1"), ep("/m2", "tmdb:100"), ex("/e1")]).overall
    assert m.extra_multi_files == 2 and m.extra_multi_as_extra == 1
    assert m.extra_multi_as_episode_high == 1  # a play-all taken for an episode at high
    assert m.precision_high.errors == 1  # and that is an error at high like any extra missed
