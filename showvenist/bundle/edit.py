"""`bundle edit` (spec §6.6): create overrides.yaml from a template, append a conflict's
paste-ready fix, open the editor, validate on save the way load_bundle does."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import yaml

from showvenist.bundle.overrides import BundleError, override_warnings
from showvenist.bundle.schema import Bundle

TEMPLATE = """\
# Overrides for this show. Hand-written; the tool never changes it except through `bundle edit`.
# Keys are episode ids from show.json (quote them). Say why you overrode something in a comment.
#
# episodes:
#   "tmdb:62091":
#     synopses:
#       - source: manual
#         text: "TMDB's overview was one line, so here is a better one."
#     runtime_min: 86
#   "tmdb:62104":
#     excluded: true          # unaired; not on my disc set
#
# new_episodes:               # an episode the source lacks, e.g. half of a two-parter
#   "custom:boe-1": { title: "The Birth of Evil, Part 1", runtime_min: 22 }
#
# orderings:                  # a custom numbering, e.g. how my discs are laid out
#   disk:
#     base: aired             # inherit every season from this ordering...
#     entries:                # ...and replace only the seasons listed here
#       - { season: 4, episode: 12, episode_id: "custom:boe-1" }
#
# title_card:                 # an on-screen title card that names the episode (see the spec)
#   windows: [ { from: end, skip_seconds: 0, length_seconds: 180 } ]
#   pattern: 'EPISODE\\s+(?P<ref>[IVXLC]+)'
#   kind: roman               # roman | number | title
"""


class FixError(Exception):
    """The fix can't be merged into the existing file automatically."""


def _map(node: yaml.Node | None) -> dict[str, tuple[yaml.Node, yaml.Node]]:
    if not isinstance(node, yaml.MappingNode):
        return {}
    return {str(k.value): (k, v) for k, v in node.value if isinstance(k, yaml.ScalarNode)}


def _dump(doc: object, indent: int = 0) -> list[str]:
    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)
    return [" " * indent + ln for ln in text.splitlines()]


def _insert_at(node: yaml.Node, lines: list[str]) -> int:
    """Index of the line after the last content line of `node` (trailing blanks and comments
    that precede the next key are not part of it)."""
    if isinstance(node, yaml.CollectionNode) and not node.flow_style:
        # block collection (an indentless sequence reports None): ends at the next token
        i = min(node.end_mark.line, len(lines))
        while i > 0 and (not lines[i - 1].strip() or lines[i - 1].lstrip().startswith("#")):
            i -= 1
        return i
    return node.end_mark.line + 1


def _block(node: yaml.Node, what: str) -> yaml.Node:
    if not isinstance(node, yaml.CollectionNode) or node.flow_style:
        raise FixError(f"{what} is written in flow style or empty; paste the fix by hand")
    return node


def apply_fix(existing: str, fix: str) -> tuple[str, list[str]]:
    """Append a conflict's `fix` block to the text of overrides.yaml. Top-level keys the file
    doesn't have are appended; `new_episodes` and `orderings` the file already has are merged
    entry by entry, so no YAML key is duplicated. In particular every season's fix defines
    `orderings.aired-split` (base + entries): the entries of seasons the file doesn't cover yet
    are added to the one mapping. Comments in `existing` are kept. Returns (text, notes);
    raises FixError when it can't merge (flow-style sections, a different base)."""
    notes: list[str] = []
    fix_doc = yaml.safe_load(fix) or {}
    cur = yaml.safe_load(existing) if existing.strip() else {}
    cur = cur or {}
    if not isinstance(fix_doc, dict) or not isinstance(cur, dict):
        raise FixError("overrides.yaml and the fix must both be mappings")
    root = yaml.compose(existing, Loader=yaml.SafeLoader) if existing.strip() else None
    top = _map(root)
    lines = existing.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    inserts: list[tuple[int, list[str]]] = []  # (line index, new lines)
    appended: dict[str, object] = {}

    for key, val in fix_doc.items():
        if key not in cur:
            appended[key] = val
            continue
        if not isinstance(val, dict) or not isinstance(cur[key], dict):
            raise FixError(f"{key!r} is not a mapping in the file; paste the fix by hand")
        section = _block(top[key][1], f"{key!r}")
        indent = section.start_mark.column
        new_children: dict[str, object] = {}
        for child, cval in val.items():
            if child not in cur[key]:
                new_children[child] = cval
            elif key == "orderings":
                _merge_entries(child, cval, cur[key][child], _map(section)[child][1],
                               lines, inserts, notes)
            else:
                notes.append(f"{key}.{child} is already in overrides.yaml; left as it is")
        if new_children:
            inserts.append((_insert_at(section, lines), _dump(new_children, indent)))

    for idx, new in sorted(inserts, key=lambda t: t[0], reverse=True):
        lines[idx:idx] = new
    out = "\n".join(lines)
    if appended:
        head = [ln for ln in fix.splitlines() if ln.startswith("#")]
        block = "\n".join([*head, *_dump(appended)])
        out = (out + "\n\n" if out.strip() else "") + block
    return out + "\n", notes


def _merge_entries(name: str, new: object, old: object, node: yaml.Node, lines: list[str],
                   inserts: list[tuple[int, list[str]]], notes: list[str]) -> None:
    """Merge the {base, entries} ordering `new` into the file's ordering `name`."""
    if not (isinstance(new, dict) and isinstance(old, dict)
            and isinstance(old.get("entries"), list)):
        raise FixError(f"orderings.{name} in the file is not {{base, entries}}; "
                       "paste the fix by hand")
    if old.get("base") != new.get("base"):
        raise FixError(f"orderings.{name} in the file has base {old.get('base')!r}, the fix "
                       f"{new.get('base')!r}; paste the fix by hand")
    have = {e.get("season") for e in old["entries"] if isinstance(e, dict)}
    seasons = sorted({e["season"] for e in new["entries"]} - have)
    for s in sorted({e["season"] for e in new["entries"]} & have):
        notes.append(f"orderings.{name} already has season {s}; left as it is")
    if not seasons:
        return
    seq = _block(_map(node)["entries"][1], f"orderings.{name}.entries")
    indent = seq.start_mark.column
    items = [e for e in new["entries"] if e["season"] in seasons]
    text = [" " * indent + "- " + yaml.safe_dump(
        e, default_flow_style=True, sort_keys=False, width=1000).strip() for e in items]
    inserts.append((_insert_at(seq, lines), text))


def validate(bundle: Bundle, work: Path, target: Path) -> list[str]:
    """Load `work` over `bundle` as load_bundle would; errors name `target`, not the scratch
    file. Returns warnings (overrides for episodes that don't exist); raises BundleError."""
    from showvenist.bundle.overrides import apply_overrides

    try:
        apply_overrides(bundle, work)
    except BundleError as e:
        raise BundleError(target, e.message, e.line) from None
    return override_warnings(bundle, work, shown_as=target)


def edit_overrides(
    bundle: Bundle,
    bundle_dir: Path,
    *,
    fix: str | None,
    run_editor: Callable[[Path], int],
    confirm: Callable[[str], bool],
    echo: Callable[[str], None],
) -> str:
    """Returns "saved", "unchanged" or "aborted". The edit happens in a scratch copy, so
    overrides.yaml is only replaced (atomically) by a file that validated."""
    target = bundle_dir / "overrides.yaml"
    original = target.read_text(encoding="utf-8") if target.exists() else None
    text = original if original is not None else TEMPLATE
    if fix is not None:
        text, notes = apply_fix(text, fix)
        for n in notes:
            echo(f"note: {n}")
    work = bundle_dir / ".overrides.yaml.edit"
    work.write_text(text, encoding="utf-8")
    try:
        while True:
            if run_editor(work) != 0:
                echo("editor exited with an error; nothing saved")
                return "aborted"
            try:
                warns = validate(bundle, work, target)
            except BundleError as e:
                echo(str(e))
                if confirm("Reopen the editor? (no: abort without saving)"):
                    continue
                return "aborted"
            for w in warns:
                echo(f"warning: {w}")
            break
        new = work.read_text(encoding="utf-8")
        if original is not None and new == original:
            return "unchanged"
        os.replace(work, target)
        return "saved"
    finally:
        work.unlink(missing_ok=True)
