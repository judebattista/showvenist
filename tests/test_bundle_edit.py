"""bundle edit / --apply-fix / bundle check numbering (spec §6.6). Editors are scripts."""

from __future__ import annotations

import os
import stat
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from showvenist import cli
from showvenist.bundle.edit import FixError, apply_fix, edit_overrides
from showvenist.bundle.merge import merge_sources
from showvenist.bundle.overrides import apply_overrides
from showvenist.bundle.store import bundle_path, load_bundle, save_bundle
from tests.test_merge import day, p_ep, primary, s_ep, secondary

runner = CliRunner()
SLUG = "testshow-2010"


def two_season_split_bundle():
    P = primary([(1, 1, p_ep(0)), (1, 2, p_ep(1, title="XXXVII", runtime_min=44)),
                 (2, 1, p_ep(2)), (2, 2, p_ep(3, title="XLII", runtime_min=44)),
                 (2, 3, p_ep(4))])
    S = secondary([(1, 1, s_ep(0)),
                   (1, 2, s_ep(1, title="The Glass Orchard (1)", runtime_min=22)),
                   (1, 3, s_ep(8, title="The Glass Orchard (2)", runtime_min=22,
                               air_date=day(1))),
                   (2, 1, s_ep(2)),
                   (2, 2, s_ep(3, title="Copper and Salt (1)", runtime_min=22)),
                   (2, 3, s_ep(9, title="Copper and Salt (2)", runtime_min=22,
                               air_date=day(3))),
                   (2, 4, s_ep(4))])
    return merge_sources(P, [S])


def split_fixes(b):
    return [p.conflict["fix"] for p in b.provenance
            if p.conflict and p.conflict["kind"] == "split"]


@pytest.fixture
def root(tmp_path):
    save_bundle(two_season_split_bundle(), tmp_path)
    return tmp_path


def make_editor(tmp_path: Path, body: str) -> str:
    """An editor: a Python script run with the file path as argv[1]."""
    script = tmp_path / "editor.py"
    script.write_text("#!" + os.sys.executable + "\nimport sys, pathlib, os\n"
                      "p = pathlib.Path(sys.argv[1])\n" + textwrap.dedent(body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def edit(root, *args, editor: str | None = None, input=None, monkeypatch=None):
    env = {"EDITOR": editor, "VISUAL": None} if editor else {}
    return runner.invoke(cli.app, ["bundle", "edit", SLUG, "-o", str(root), *args],
                         env=env, input=input)


def test_check_numbers_conflicts(root):
    r = runner.invoke(cli.app, ["bundle", "check", SLUG, "-o", str(root)])
    assert r.exit_code == 0, r.output
    assert "Unresolved conflicts (3):" in r.output  # numbering, split S1, split S2
    assert "Season 1:" in r.output and "Season 2:" in r.output
    assert "[1] numbering" in r.output and "[2] split" in r.output and "[3] split" in r.output
    assert f"bundle edit {SLUG} --apply-fix 3" in r.output
    assert "--apply-fix 1" not in r.output  # numbering has no paste-ready fix


def test_creates_template_when_absent_and_saves(root, tmp_path):
    ed = make_editor(tmp_path, "pass\n")
    r = edit(root, editor=ed)
    assert r.exit_code == 0 and "saved" in r.output, r.output
    ov = bundle_path(SLUG, root) / "overrides.yaml"
    assert ov.read_text().startswith("# Overrides for this show")
    assert yaml.safe_load(ov.read_text()) is None
    assert not (bundle_path(SLUG, root) / ".overrides.yaml.edit").exists()


def test_unchanged_existing_file_is_not_rewritten(root, tmp_path):
    ov = bundle_path(SLUG, root) / "overrides.yaml"
    ov.write_text("# mine\n")
    mtime = ov.stat().st_mtime_ns
    r = edit(root, editor=make_editor(tmp_path, "pass\n"))
    assert r.exit_code == 0 and "unchanged" in r.output
    assert ov.stat().st_mtime_ns == mtime


def test_apply_fix_then_editor_sees_it_and_result_loads(root, tmp_path):
    ed = make_editor(tmp_path, "pathlib.Path(os.environ['SEEN']).write_text(p.read_text())\n")
    seen = tmp_path / "seen.txt"
    r = runner.invoke(cli.app, ["bundle", "edit", SLUG, "--apply-fix", "2", "-o", str(root)],
                      env={"EDITOR": ed, "VISUAL": None, "SEEN": str(seen)})
    assert r.exit_code == 0, r.output
    assert "new_episodes:" in seen.read_text() and "# Overrides for this show" in seen.read_text()
    b = load_bundle(SLUG, root)  # overrides.yaml loads
    assert "custom:tvmaze-501" in b.episode_by_id()
    s1 = [e.episode_id for e in b.orderings["aired-split"].entries if e.season == 1]
    assert s1 == ["tmdb:100", "custom:tvmaze-501", "custom:tvmaze-508"]


def test_apply_both_seasons_merges_into_one_aired_split(root, tmp_path):
    ed = make_editor(tmp_path, "pass\n")
    assert edit(root, "--apply-fix", "2", editor=ed).exit_code == 0
    r = edit(root, "--apply-fix", "3", editor=ed)
    assert r.exit_code == 0, r.output
    ov = bundle_path(SLUG, root) / "overrides.yaml"
    text = ov.read_text()
    assert text.count("\n  aired-split:") == 1 and text.count("\nnew_episodes:") == 1
    assert text.count("\norderings:") == 1
    b = load_bundle(SLUG, root)
    order = b.orderings["aired-split"]
    ids = {s: [e.episode_id for e in order.entries if e.season == s] for s in (1, 2)}
    assert ids[1] == ["tmdb:100", "custom:tvmaze-501", "custom:tvmaze-508"]
    assert ids[2] == ["tmdb:102", "custom:tvmaze-503", "custom:tvmaze-509", "tmdb:104"]
    # applying season 2's fix again changes nothing
    r = edit(root, "--apply-fix", "3", editor=ed)
    assert "already has season 2" in r.output and "unchanged" in r.output


def test_apply_fix_unit_keeps_comments_and_inserts_in_place():
    b = two_season_split_bundle()
    f1, f2 = split_fixes(b)[0], split_fixes(b)[-1]
    existing = textwrap.dedent("""\
        # why: the disc set
        episodes:
          "tmdb:100":
            runtime_min: 40   # short

        # my own ordering
        orderings:
          mine:
            - { season: 1, episode: 1, episode_id: "tmdb:100" }

        # trailing note
        """)
    existing, _ = apply_fix(existing, f1)
    assert "# trailing note" in existing
    out, notes = apply_fix(existing, f2)
    doc = yaml.safe_load(out)
    assert "# why: the disc set" in out and "# my own ordering" in out and "# short" in out
    assert out.count("aired-split:") == 1 and out.count("\nnew_episodes:") == 1
    assert set(doc["orderings"]) == {"mine", "aired-split"}
    assert {e["season"] for e in doc["orderings"]["aired-split"]["entries"]} == {1, 2}
    assert len(doc["new_episodes"]) == 4 and notes == []
    apply_overrides(b, _write(out))


def _write(text: str) -> Path:
    import tempfile

    d = Path(tempfile.mkdtemp())
    p = d / "overrides.yaml"
    p.write_text(text)
    return p


def test_apply_fix_into_new_sections_when_file_has_only_episodes():
    b = two_season_split_bundle()
    out, _ = apply_fix('episodes:\n  "tmdb:100":\n    runtime_min: 40\n', split_fixes(b)[0])
    doc = yaml.safe_load(out)
    assert set(doc) == {"episodes", "new_episodes", "orderings"}
    apply_overrides(b, _write(out))


def test_apply_fix_refuses_different_base():
    b = two_season_split_bundle()
    f = split_fixes(b)[0]
    existing = yaml.safe_dump({"orderings": {"aired-split": {
        "base": "dvd", "entries": [{"season": 9, "episode": 1, "episode_id": "tmdb:100"}]}}})
    with pytest.raises(FixError, match="base"):
        apply_fix(existing, f)


def test_apply_fix_bad_number_and_non_split(root):
    r = edit(root, "--apply-fix", "4", editor="true")
    assert r.exit_code == 1 and "3 conflicts" in r.output
    r = edit(root, "--apply-fix", "1", editor="true")
    assert r.exit_code == 1 and "no paste-ready fix" in r.output
    b = load_bundle(SLUG, root, apply_overrides=False)
    from showvenist.bundle.schema import Provenance

    b = b.model_copy(update={"provenance": [Provenance(adapter="tvmaze", conflict={
        "kind": "air_date", "season": "1", "detail": "d"})]})
    save_bundle(b, root)
    r = edit(root, "--apply-fix", "1", editor="true")
    assert r.exit_code == 1 and "no paste-ready fix" in r.output


def test_invalid_yaml_reopens_then_fixed(root, tmp_path):
    counter = tmp_path / "n"
    ed = make_editor(tmp_path, f"""\
        n = pathlib.Path({str(counter)!r})
        k = int(n.read_text()) if n.exists() else 0
        n.write_text(str(k + 1))
        p.write_text('episodes:\\n  "tmdb:100":\\n    bogus_field: 1\\n' if k == 0
                     else 'episodes:\\n  "tmdb:100":\\n    runtime_min: 77\\n')
        """)
    r = edit(root, editor=ed, input="y\n")
    assert r.exit_code == 0, r.output
    assert "overrides.yaml:3: unknown field 'bogus_field'" in r.output
    assert counter.read_text() == "2"
    assert load_bundle(SLUG, root).episode_by_id()["tmdb:100"].runtime_min == 77


def test_invalid_then_abort_never_writes(root, tmp_path):
    ov = bundle_path(SLUG, root) / "overrides.yaml"
    ov.write_text("# keep me\n")
    ed = make_editor(tmp_path, "p.write_text('episodes: [oops\\n')\n")
    r = edit(root, editor=ed, input="n\n")
    assert r.exit_code == 1 and "invalid YAML" in r.output and "Aborted" in r.output
    assert ov.read_text() == "# keep me\n"
    assert not (bundle_path(SLUG, root) / ".overrides.yaml.edit").exists()


def test_abort_with_apply_fix_leaves_no_file(root, tmp_path):
    ed = make_editor(tmp_path, "sys.exit(1)\n")
    r = edit(root, "--apply-fix", "1", editor=ed)
    assert r.exit_code == 1
    assert not (bundle_path(SLUG, root) / "overrides.yaml").exists()


def test_editor_falls_back_to_visual(root, tmp_path, monkeypatch):
    ed = make_editor(tmp_path, "p.write_text('# via visual\\n')\n")
    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setenv("VISUAL", ed)
    r = runner.invoke(cli.app, ["bundle", "edit", SLUG, "-o", str(root)])
    assert r.exit_code == 0
    assert (bundle_path(SLUG, root) / "overrides.yaml").read_text() == "# via visual\n"


def test_visual_wins_over_editor_and_takes_arguments(root, tmp_path, monkeypatch):
    d = tmp_path / "my editors"
    d.mkdir()
    script = d / "ed.py"
    script.write_text("import sys, pathlib\n"
                      "pathlib.Path(sys.argv[-1]).write_text('# ' + sys.argv[1] + '\\n')\n")
    monkeypatch.setenv("VISUAL", f"{os.sys.executable} '{script}' --flag")
    monkeypatch.setenv("EDITOR", "false")
    r = runner.invoke(cli.app, ["bundle", "edit", SLUG, "-o", str(root)])
    assert r.exit_code == 0, r.output
    assert (bundle_path(SLUG, root) / "overrides.yaml").read_text() == "# --flag\n"


def test_apply_fix_without_final_newline():
    b = two_season_split_bundle()
    existing = 'new_episodes:\n  "custom:x":\n    title: Mine\n    runtime_min: 5'
    out, _ = apply_fix(existing, split_fixes(b)[0])
    assert out.startswith(existing + "\n")
    doc = yaml.safe_load(out)
    assert doc["new_episodes"]["custom:x"] == {"title": "Mine", "runtime_min": 5}
    assert set(doc["new_episodes"]) == {"custom:x", "custom:tvmaze-501", "custom:tvmaze-508"}
    assert out.endswith("\n")
    apply_overrides(b, _write(out))


@pytest.mark.parametrize("text", ["episodes: [oops\n", "episodes: {}\n---\norderings: {}\n"],
                         ids=["malformed", "two-documents"])
def test_apply_fix_invalid_yaml_is_a_clean_error(root, text):
    with pytest.raises(FixError, match="not valid YAML"):
        apply_fix(text, split_fixes(two_season_split_bundle())[0])
    ov = bundle_path(SLUG, root) / "overrides.yaml"
    ov.write_text(text)
    r = edit(root, "--apply-fix", "2", editor="true")
    assert r.exit_code == 1 and "not valid YAML" in r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)
    assert ov.read_text() == text


def test_apply_fix_keeps_a_section_trailing_comment_before_the_new_lines():
    b = two_season_split_bundle()
    existing = textwrap.dedent("""\
        new_episodes:
          "custom:x": { title: "Mine" }
          # about custom:x

        # top-level note
        episodes: {}
        """)
    out, _ = apply_fix(existing, split_fixes(b)[0])
    lines = out.splitlines()
    assert lines.index("  # about custom:x") == 2
    assert lines.index("  custom:tvmaze-501:") > 2
    assert lines.index("# top-level note") > lines.index("  custom:tvmaze-508:")


def test_apply_fix_keeps_crlf(root, tmp_path):
    ov = bundle_path(SLUG, root) / "overrides.yaml"
    ov.write_bytes(b'# mine\r\nepisodes:\r\n  "tmdb:100":\r\n    runtime_min: 40\r\n')
    r = edit(root, "--apply-fix", "2", editor="true")
    assert r.exit_code == 0, r.output
    data = ov.read_bytes()
    assert b"new_episodes:\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")
    assert "custom:tvmaze-501" in load_bundle(SLUG, root).episode_by_id()


def test_edit_overrides_seam_with_callables(root):
    b = load_bundle(SLUG, root, apply_overrides=False)
    seen = []
    res = edit_overrides(b, bundle_path(SLUG, root), fix=None,
                         run_editor=lambda p: (seen.append(p.read_text()), 0)[1],
                         confirm=lambda m: False, echo=lambda m: None)
    assert res == "saved" and seen[0].startswith("# Overrides")
