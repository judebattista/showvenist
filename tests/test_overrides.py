import pytest

from showvenist.bundle.overrides import BundleError, apply_overrides
from showvenist.bundle.schema import Bundle, CastEntry, Episode, ShowInfo, Synopsis


def make() -> Bundle:
    return Bundle(
        show=ShowInfo(title="S"),
        episodes=[
            Episode(
                id="tmdb:1",
                title="A",
                alt_titles=["a1"],
                directors=["D"],
                synopses=[Synopsis(source="tmdb", text="t")],
            ),
            Episode(id="tmdb:2", title="B"),
        ],
    ).with_revision()


def run(tmp_path, text, bundle=None):
    p = tmp_path / "overrides.yaml"
    p.write_text(text)
    return apply_overrides(bundle or make(), p)


def test_fields(tmp_path):
    b = run(
        tmp_path,
        """
# comment
episodes:
  tmdb:1:
    title: New
    air_date: "2020-01-02"
    runtime_min: 86
    alt_titles: [a1, a2]
    directors: [X, Y]
    writers: [W]
    synopses:
      - {source: manual, text: hello}
    guest_cast:
      - {actor: Z, character: Q}
  tmdb:2:
    excluded: true
""",
    )
    e1, e2 = b.episodes
    assert (e1.title, e1.air_date, e1.runtime_min) == ("New", "2020-01-02", 86)
    assert e1.alt_titles == ["a1", "a2"]
    assert e1.directors == ["X", "Y"] and e1.writers == ["W"]
    assert [s.source for s in e1.synopses] == ["tmdb", "manual"]
    assert e1.guest_cast == [CastEntry(actor="Z", character="Q")]
    assert e2.excluded
    assert b.bundle_revision == b.compute_revision() != make().bundle_revision


def test_orderings(tmp_path):
    b = run(
        tmp_path,
        """
orderings:
  rel:
    - { season: 1, episode: 1, episode_id: "tmdb:2" }
""",
    )
    o = b.orderings["rel"]
    assert o.entries[0].episode_id == "tmdb:2" and o.source == "overrides"


def test_empty_file(tmp_path):
    assert run(tmp_path, "").episodes == make().episodes


@pytest.mark.parametrize(
    "text,line,frag",
    [
        ("episodes:\n  tmdb:1:\n    title: x\n    bogus: 1\n", 4, "unknown field"),
        ("episodes:\n  tmdb:1:\n    excluded: [1]\n", 3, "excluded"),
        ("episodes:\n  tmdb:1:\n    synopses:\n      - {source: m}\n", 4, "text"),
        ("orderings:\n  r:\n    - {season: 1}\n", 3, "episode"),
        ("junk: 1\n", 1, "unknown top-level"),
        ("episodes: [\n", None, "invalid YAML"),
    ],
)
def test_errors(tmp_path, text, line, frag):
    with pytest.raises(BundleError) as ei:
        run(tmp_path, text)
    assert frag in str(ei.value)
    assert str(ei.value).startswith(str(tmp_path / "overrides.yaml"))
    if line is not None:
        assert ei.value.line == line
        assert f"overrides.yaml:{line}:" in str(ei.value)


def test_vanished_episodes_warn_and_are_skipped(tmp_path):
    text = (
        "episodes:\n  tmdb:1:\n    title: x\n  tmdb:99:\n    title: y\n"
        "orderings:\n  r:\n    - {season: 1, episode: 1, episode_id: tmdb:1}\n"
        "    - {season: 1, episode: 2, episode_id: nope}\n"
    )
    with pytest.warns(UserWarning, match="unknown episode"):
        b = run(tmp_path, text)
    assert b.episodes[0].title == "x"
    assert [e.episode_id for e in b.orderings["r"].entries] == ["tmdb:1"]


def _write(tmp_path, text):
    p = tmp_path / "overrides.yaml"
    p.write_text(text)
    return p


def test_new_episodes_and_an_inherited_ordering(tmp_path):
    """Samurai Jack: TMDB lists one 'Birth of Evil'; the disks hold two parts before the finale."""
    from showvenist.bundle.schema import Ordering, OrderingEntry

    b = make()  # tmdb:1, tmdb:2
    b = b.model_copy(update={"orderings": {"aired": Ordering(
        source="tmdb", type="aired",
        entries=[OrderingEntry(season=1, episode=1, episode_id="tmdb:1"),
                 OrderingEntry(season=2, episode=1, episode_id="tmdb:2")])}})
    p = _write(tmp_path, """
new_episodes:
  "custom:boe-1": {title: "The Birth of Evil, Part 1", runtime_min: 22, synopses: ["Part one."]}
  "custom:boe-2": {title: "The Birth of Evil, Part 2", runtime_min: 22}
orderings:
  disk:
    base: aired
    entries:
      - {season: 2, episode: 1, episode_id: "custom:boe-1"}
      - {season: 2, episode: 2, episode_id: "custom:boe-2"}
      - {season: 2, episode: 3, episode_id: "tmdb:2"}
""")
    out = apply_overrides(b, p)
    ids = [e.id for e in out.episodes]
    assert ids == ["tmdb:1", "tmdb:2", "custom:boe-1", "custom:boe-2"]
    boe = out.episode_by_id()["custom:boe-1"]
    assert boe.runtime_min == 22 and boe.synopses[0].text == "Part one."
    disk = out.orderings["disk"]
    assert [(e.season, e.episode, e.episode_id) for e in disk.entries] == [
        (1, 1, "tmdb:1"), (2, 1, "custom:boe-1"), (2, 2, "custom:boe-2"), (2, 3, "tmdb:2")]
    assert out.orderings["aired"] == b.orderings["aired"]  # the base is untouched


def test_new_episode_errors(tmp_path):
    with pytest.raises(BundleError, match="already exists"):
        apply_overrides(make(), _write(tmp_path, 'new_episodes:\n  "tmdb:1": {title: x}\n'))
    with pytest.raises(BundleError, match="needs a mapping with a 'title'"):
        bad = 'new_episodes:\n  "custom:a": {runtime_min: 5}\n'
        apply_overrides(make(), _write(tmp_path, bad))
    with pytest.raises(BundleError, match="unknown base"):
        bad = 'orderings:\n  o:\n    base: nope\n    entries: []\n'
        apply_overrides(make(), _write(tmp_path, bad))
