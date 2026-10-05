"""Session end to end on tiny synthetic MKVs (no real media, no network)."""

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path

import pytest

from showvenist.analysis.types import AnalysisCache
from showvenist.bundle.schema import (
    Bundle,
    Episode,
    Ordering,
    OrderingEntry,
    ShowInfo,
    SubtitleRef,
    Synopsis,
)
from showvenist.config import Config
from showvenist.report import exit_code
from showvenist.session import FLAG_DUPLICATE, FLAG_NO_DIALOGUE, Session, disc_of
from tests.fixtures.textsubs import make_mkv_with_text_subs, make_srt

# distinctive words per episode: the synopsis states them, the dialogue paraphrases them
TOPICS = [
    "dragon volcano treasure lighthouse smuggler",
    "submarine glacier penguin telescope blizzard",
    "carnival magician tiger balloon acrobat",
    "orchard beekeeper honey tractor scarecrow",
]
RUNTIMES = [8, 11, 14, 17]  # minutes


def synth_bundle(seasons: int = 1) -> Bundle:
    eps, entries = [], []
    for s in range(1, seasons + 1):
        for k, topic in enumerate(TOPICS):
            eid = f"tmdb:{s}{k:02d}"
            words = topic.split()
            if s > 1:  # keep seasons apart in vocabulary as well
                words = [w + "ish" for w in words]
            eps.append(
                Episode(
                    id=eid,
                    title=f"Story {s}.{k + 1}",
                    runtime_min=RUNTIMES[k],
                    synopses=[
                        Synopsis(
                            source="tmdb",
                            text=f"The crew meets a {words[0]} near the {words[1]}, hunts a "
                            f"{words[2]} by the {words[3]} and escapes a {words[4]}.",
                        )
                    ],
                )
            )
            entries.append(OrderingEntry(season=s, episode=k + 1, episode_id=eid))
    return Bundle(
        show=ShowInfo(title="Synth Show", year=2000),
        episodes=eps,
        orderings={"aired": Ordering(source="tmdb", type="aired", entries=entries)},
    ).with_revision()


FILLER = (
    "we you they never always maybe tonight tomorrow listen look run stop wait follow north "
    "south river road house door window money ship captain doctor engine fire water light "
    "dark quick slow broken heavy empty secret promise trouble plan deal job friend enemy "
    "brother sister mother father truth lie gold silver stone glass paper letter map key"
).split()


def dialogue_for(k: int, season: int = 1) -> list[str]:
    """The episode's script: lines paraphrasing its synopsis among varied filler lines."""
    words = TOPICS[k].split()
    if season > 1:
        words = [w + "ish" for w in words]
    topical = [
        f"Look out, that {words[0]} is coming!",
        f"We can't go back to the {words[1]} tonight.",
        f"Nobody believes the {words[2]} is real.",
        f"Meet me at the {words[3]} at dawn.",
        f"That {words[4]} nearly had us.",
    ]
    rng = random.Random(season * 100 + k)
    lines = []
    for n in range(240):
        lines.append(topical[n % 5] if n % 4 == 0 else " ".join(rng.choices(FILLER, k=8)))
    return lines


def write_refs(bundle: Bundle, bundle_dir: Path) -> Bundle:
    """Reference subtitles (the episodes' full scripts) under bundle_dir/subs."""
    (bundle_dir / "subs").mkdir(parents=True, exist_ok=True)
    eps = []
    for e in bundle.episodes:
        s, k = int(e.id[5]), int(e.id[6:])
        rel = f"subs/{e.id.replace(':', '-')}.en.srt"
        make_srt(bundle_dir / rel, [(5.0 * n, 5.0 * n + 2, t)
                                    for n, t in enumerate(dialogue_for(k, s))])
        eps.append(e.model_copy(update={
            "subtitles": [SubtitleRef(lang="en", path=rel, source="localsubs")]}))
    return bundle.model_copy(update={"episodes": eps}).with_revision(bundle_dir)


def make_episode_mkv(
    path: Path, minutes: float, lines: list[str] | None, lang: str = "eng"
) -> Path:
    """Cues every 5 s over the whole file (so every sampled window catches some)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dur = minutes * 60
    cues = []
    if lines:
        t, n = 2.0, 0
        while t + 2 < dur:
            cues.append((t, t + 2.0, lines[n % len(lines)]))
            t += 5.0
            n += 1
    make_mkv_with_text_subs(path, cues, lang=lang, duration_s=dur)
    return path


@pytest.fixture(scope="module")
def rips(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("rips") / "Synth Show" / "Season 1" / "Disc 1"
    out = {
        "e3": make_episode_mkv(root / "t00.mkv", 14, dialogue_for(2)),
        "e1": make_episode_mkv(root / "t01.mkv", 8, dialogue_for(0)),
        "nosubs": make_episode_mkv(root / "t02.mkv", 17, dialogue_for(3), lang="ger"),
    }
    out["dup"] = root / "t03.mkv"
    shutil.copyfile(out["e3"], out["dup"])
    return out


@pytest.fixture
def session(rips, tmp_path) -> Session:
    bdir = tmp_path / "synth-show-2000"
    s = Session(write_refs(synth_bundle(), bdir), bdir, Config(), season=1,
                cache=AnalysisCache(tmp_path / "analysis.db"))
    s.analyze([rips["e3"], rips["e1"], rips["nosubs"], rips["dup"]])
    return s


def _assigned(sol, path: Path) -> str:
    i = next(k for k, f in enumerate(sol.files) if f.path == path)
    col = sol.assignment.column[i]
    return "extra" if col < 0 else sol.candidates[col].label


def test_assignment_duplicate_and_no_dialogue(session, rips):
    assert [f.path for f in session.files] == [rips["e3"], rips["e1"], rips["nosubs"]]
    assert session.duplicates == {rips["dup"]: rips["e3"]}
    assert session.files[0].disc == "Disc 1"
    sol = session.solve()
    assert _assigned(sol, rips["e3"]) == "S01E03"
    assert _assigned(sol, rips["e1"]) == "S01E01"
    assert sol.confidences[0].level == "high"
    # no usable (English) track: duration only
    i = 2
    assert session.files[i].dialogue is not None and session.files[i].dialogue.track is None
    for name in ("synopsis", "guests", "refsubs"):
        assert not sol.results[name].available[i]
    assert sol.results["duration"].available[i]
    assert _assigned(sol, rips["nosubs"]) == "S01E04"
    assert sol.confidences[i].level != "high"

    rep = session.report()
    assert len(rep.files) == 4
    by_path = {r.path: r for r in rep.files}
    dup = by_path[str(rips["dup"])]
    assert FLAG_DUPLICATE in dup.flags
    assert dup.decision.episode is not None and dup.decision.episode.episode == 3
    assert FLAG_NO_DIALOGUE in by_path[str(rips["nosubs"])].flags
    assert exit_code(rep) == 2


def test_pin_and_unpin(session, rips):
    session.solve()
    sol = session.pin(rips["e1"], "S01E02")
    assert _assigned(sol, rips["e1"]) == "S01E02"
    pinned = [r.path for r in session.report().files if "pinned" in r.flags]
    assert pinned == [str(rips["e1"])]
    sol = session.pin(rips["e3"], "extra")
    assert _assigned(sol, rips["e3"]) == "extra"
    with pytest.raises(ValueError):
        session.pin(rips["nosubs"], "S01E02")  # already pinned to another file
    sol = session.unpin(rips["e1"])
    sol = session.unpin(rips["e3"])
    assert _assigned(sol, rips["e1"]) == "S01E01"
    assert _assigned(sol, rips["e3"]) == "S01E03"
    # pinning via a duplicate's path pins its primary
    sol = session.pin(rips["dup"], "tmdb:101")
    assert _assigned(sol, rips["e3"]) == "S01E02"


def test_unreadable_file_is_recorded(tmp_path, rips):
    bad = tmp_path / "bad.mkv"
    bad.write_bytes(b"not a matroska file" * 100)
    seen = []
    s = Session(synth_bundle(), None, Config(), season=1)
    s.analyze([bad, rips["e1"]], progress=lambda n, t, p, e: seen.append((n, t, p, e)))
    assert bad in s.errors
    assert [f.path for f in s.files] == [rips["e1"]]
    assert [x[0] for x in seen] == [1, 2] and seen[0][3] is not None and seen[1][3] is None
    assert _assigned(s.solve(), rips["e1"]) == "S01E01"


def test_disc_of():
    assert disc_of(Path("/r/Show/Season 1/FIREFLY_D2/t.mkv")) == "FIREFLY_D2"
    assert disc_of(Path("/r/Show/Season 1/DISC_3/t.mkv")) == "DISC_3"
    assert disc_of(Path("/r/Show/Season 1/t.mkv")) is None


def test_no_candidates_raises():
    with pytest.raises(ValueError):
        Session(synth_bundle(), None, Config(), season=9)


# ---- LLM pass (§10.2) with a fake client: never the network ------------------------------------


class FakeLlmClient:
    """Stands in for anthropic.Anthropic(): `messages.parse(**request)`. Rates the candidate
    whose topic words the excerpt contains, quoting the excerpt verbatim."""

    def __init__(self):
        self.calls: list[dict] = []
        self.messages = self

    def parse(self, **kw):
        from types import SimpleNamespace as NS

        from showvenist.scoring.llm import CandidateScore, LLMResult

        self.calls.append(kw)
        block, excerpt = (c["text"] for c in kw["messages"][0]["content"])
        labels = re.findall(r"\[(S\d+E\d+)\]", block)
        top = None
        for k, topic in enumerate(TOPICS):
            word = topic.split()[0]
            if word in excerpt:
                top = next((lb for lb in labels if int(lb.split("E")[1]) == k + 1), None)
                at = excerpt.index(word)
                quote = excerpt[max(0, at - 10): at + 20]
                break
        scores = [
            CandidateScore(episode=lb, probability=0.9 if lb == top else 0.01,
                           quotes=[quote] if lb == top else [])
            for lb in labels
        ]
        usage = NS(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0,
                   cache_read_input_tokens=0)
        return NS(parsed_output=LLMResult(scores=scores, none_probability=0.02),
                  stop_reason="end_turn", usage=usage)


def test_llm_pass_targets_and_apply(session, rips):
    from showvenist.scoring.llm import LlmScorer

    sol = session.solve()
    assert session.llm_targets("off") == []
    assert session.llm_targets("all") == [0, 1, 2]
    assert session.llm_targets("auto") == [
        i for i, c in enumerate(sol.confidences) if c.level != "high"
    ]
    assert 2 in session.llm_targets("auto")  # the no-dialogue file is below high
    with pytest.raises(ValueError):
        session.llm_targets("sometimes")
    client = FakeLlmClient()
    llm = LlmScorer(client)
    llm.send = {0, 1}
    assert llm.estimate_cost(session.files, session.context()) > 0
    sol = session.apply_llm(llm)
    assert len(client.calls) == 4  # two files, both candidate orders
    assert sol.regimes == ["llm", "llm", "base"]
    assert {"duration", "refsubs", "llm"} <= set(sol.results)
    assert _assigned(sol, rips["e3"]) == "S01E03" and _assigned(sol, rips["e1"]) == "S01E01"
    assert llm.total_cost_usd > 0
