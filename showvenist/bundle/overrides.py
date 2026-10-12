"""Overrides handling (spec §6.6). overrides.yaml is hand-written; the tool never writes it."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from showvenist.bundle.schema import (
    Bundle,
    CastEntry,
    Episode,
    Ordering,
    OrderingEntry,
    SubtitleRef,
    Synopsis,
    TitleCardConfig,
)


class BundleError(Exception):
    def __init__(self, path: Path | str, message: str, line: int | None = None):
        self.path = Path(path)
        self.message = message
        self.line = line
        super().__init__(str(self))

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line is not None else f"{self.path}"
        return f"{loc}: {self.message}"


_SCALARS = {"title", "air_date", "excluded", "runtime_min", "production_code"}
_REPLACE_LISTS = {"directors", "writers"}
_ADD_LISTS = {"alt_titles", "synopses", "guest_cast"}
_REPLACE_MODELS = {"subtitles"}  # replace semantics; an empty list removes the references
_FIELDS = _SCALARS | _REPLACE_LISTS | _ADD_LISTS | _REPLACE_MODELS
_ORDERING_SOURCE = "overrides"
_ORDERING_TYPE = "custom"


def _line(node: yaml.Node | None) -> int | None:
    return node.start_mark.line + 1 if node is not None else None


def _map(node: yaml.Node | None) -> dict[Any, tuple[yaml.Node, yaml.Node]]:
    """Mapping node -> {key string: (key node, value node)}."""
    if not isinstance(node, yaml.MappingNode):
        return {}
    return {str(k.value): (k, v) for k, v in node.value if isinstance(k, yaml.ScalarNode)}


def load_overrides_text(path: Path) -> tuple[dict[str, Any], yaml.Node | None]:
    try:
        text = path.read_text(encoding="utf-8")
        root = yaml.compose(text, Loader=yaml.SafeLoader)
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        raise BundleError(path, f"invalid YAML: {getattr(e, 'problem', e)}",
                          mark.line + 1 if mark else None) from e
    if data is None:
        return {}, root
    if not isinstance(data, dict):
        raise BundleError(path, "top level must be a mapping", _line(root))
    return data, root


def _warn_vanished(path: Path, ep_id: str, node) -> None:
    """Spec §6.6: after a refresh, overrides for vanished episodes warn; they are skipped."""
    warnings.warn(
        f"{path}:{_line(node)}: override for unknown episode {ep_id!r} ignored",
        stacklevel=3,
    )


def apply_overrides(bundle: Bundle, path: Path) -> Bundle:
    """Merge overrides.yaml at `path` over `bundle`; returns a new Bundle with fresh revision."""
    data, root = load_overrides_text(path)
    top = _map(root)
    for key, (knode, _) in top.items():
        if key not in ("episodes", "orderings", "new_episodes", "title_card"):
            raise BundleError(path, f"unknown top-level key {key!r}", _line(knode))

    episodes = {e.id: e.model_copy(deep=True) for e in bundle.episodes}
    added: list[str] = []
    _add_new_episodes(data, top, episodes, added, path)
    ep_section = data.get("episodes")
    if ep_section is not None and not isinstance(ep_section, dict):
        raise BundleError(path, "'episodes' must be a mapping", _line(top["episodes"][1]))
    ep_nodes = _map(top["episodes"][1]) if "episodes" in top else {}
    for ep_id, fields in (ep_section or {}).items():
        ep_id = str(ep_id)
        if ep_id not in ep_nodes:
            raise BundleError(path, f"unsupported episode key {ep_id!r} (quote it)", None)
        knode, vnode = ep_nodes[ep_id]
        if ep_id not in episodes:
            _warn_vanished(path, ep_id, knode)
            continue
        if fields is None:
            continue
        if not isinstance(fields, dict):
            raise BundleError(path, f"overrides for {ep_id} must be a mapping", _line(vnode))
        episodes[ep_id] = _apply_episode(episodes[ep_id], fields, _map(vnode), path)

    orderings = dict(bundle.orderings)
    ord_section = data.get("orderings")
    if ord_section is not None and not isinstance(ord_section, dict):
        raise BundleError(path, "'orderings' must be a mapping", _line(top["orderings"][1]))
    ord_nodes = _map(top["orderings"][1]) if "orderings" in top else {}
    for name, entries in (ord_section or {}).items():
        name = str(name)
        if name not in ord_nodes:
            raise BundleError(path, f"unsupported ordering key {name!r} (quote it)", None)
        knode, vnode = ord_nodes[name]
        if name in bundle.orderings and bundle.orderings[name].source != _ORDERING_SOURCE:
            raise BundleError(path, f"ordering {name!r} already exists in the bundle; "
                              "pick another name", _line(knode))
        base_name: str | None = None
        if isinstance(entries, dict):
            # {base: <ordering>, entries: [...]}: inherit `base`, replace the seasons listed here
            extra = set(entries) - {"base", "entries"}
            if extra or "entries" not in entries:
                raise BundleError(path, f"ordering {name!r}: use 'base' and 'entries' only "
                                  "(entries is required)", _line(vnode))
            base_name = str(entries.get("base") or "")
            if base_name and base_name not in orderings:
                raise BundleError(path, f"ordering {name!r}: unknown base {base_name!r}",
                                  _line(vnode))
            sub_nodes = _map(vnode)
            entries, vnode = entries["entries"], sub_nodes["entries"][1]
        if not isinstance(entries, list):
            raise BundleError(path, f"ordering {name!r} must be a list of entries", _line(vnode))
        parsed = []
        for item, inode in zip(entries, vnode.value, strict=True):
            try:
                parsed.append(OrderingEntry.model_validate(item))
            except ValidationError as e:
                raise BundleError(path, _verr(e), _line(inode)) from e
            if parsed[-1].episode_id not in episodes:
                _warn_vanished(path, parsed.pop().episode_id, inode)
        if base_name:
            replaced = {e.season for e in parsed}
            inherited = [e for e in orderings[base_name].entries if e.season not in replaced]
            parsed = sorted(inherited + parsed, key=lambda e: (e.season, e.episode))
        orderings[name] = Ordering(
            source=_ORDERING_SOURCE, type=_ORDERING_TYPE, name=name, entries=parsed
        )

    title_card = None
    if data.get("title_card") is not None:
        try:
            title_card = TitleCardConfig.model_validate(data["title_card"])
        except ValidationError as e:
            raise BundleError(path, f"title_card: {_verr(e)}",
                              _line(top["title_card"][1])) from e

    merged = bundle.model_copy(
        update={
            "episodes": [episodes[e.id] for e in bundle.episodes] + [episodes[i] for i in added],
            "orderings": orderings,
            "title_card": title_card,
        }
    )
    return merged.with_revision(path.parent)


def _add_new_episodes(data: dict, top: dict, episodes: dict, added: list, path: Path) -> None:
    """`new_episodes:` add episodes the source lacks (e.g. one half of a two-part TMDB entry)."""
    section = data.get("new_episodes")
    if section is None:
        return
    if not isinstance(section, dict):
        raise BundleError(path, "'new_episodes' must be a mapping", _line(top["new_episodes"][1]))
    nodes = _map(top["new_episodes"][1])
    for ep_id, fields in section.items():
        ep_id = str(ep_id)
        if ep_id not in nodes:
            raise BundleError(path, f"unsupported episode key {ep_id!r} (quote it)", None)
        knode, vnode = nodes[ep_id]
        if ep_id in episodes:
            raise BundleError(path, f"new episode {ep_id!r} already exists in the bundle; "
                              "use a new id such as 'custom:...'", _line(knode))
        if not isinstance(fields, dict) or not str(fields.get("title") or "").strip():
            raise BundleError(path, f"new episode {ep_id!r} needs a mapping with a 'title'",
                              _line(vnode))
        rest = {k: v for k, v in fields.items() if k != "title"}
        ep = Episode(id=ep_id, title=str(fields["title"]))
        episodes[ep_id] = _apply_episode(ep, rest, _map(vnode), path)
        added.append(ep_id)


def _verr(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" if err["loc"] else err["msg"]
        for err in e.errors()
    )


def _apply_episode(ep: Episode, fields: dict, nodes: dict, path: Path) -> Episode:
    update: dict[str, Any] = {}
    for name, value in fields.items():
        name = str(name)
        if name not in nodes:
            raise BundleError(path, f"unsupported key {name!r} (quote YAML keys)", None)
        knode, vnode = nodes[name]
        if name not in _FIELDS:
            raise BundleError(path, f"unknown field {name!r} for episode {ep.id}", _line(knode))
        try:
            if name in _SCALARS:
                checked = Episode.model_validate({**ep.model_dump(), name: value})
                update[name] = getattr(checked, name)
            elif name in _REPLACE_LISTS or name == "alt_titles":
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise BundleError(path, f"{name} must be a list of strings", _line(vnode))
                if name == "alt_titles":
                    update[name] = ep.alt_titles + [v for v in value if v not in ep.alt_titles]
                else:
                    update[name] = list(value)
            elif name in _REPLACE_MODELS:
                if not isinstance(value, list):
                    raise BundleError(path, f"{name} must be a list", _line(vnode))
                update[name] = [SubtitleRef.model_validate(v) for v in value]
            else:
                if not isinstance(value, list):
                    raise BundleError(path, f"{name} must be a list", _line(vnode))
                model = Synopsis if name == "synopses" else CastEntry
                if name == "synopses":  # a bare string is a synopsis from the overrides file
                    value = [{"source": "overrides", "text": v} if isinstance(v, str) else v
                             for v in value]
                new = [model.model_validate(v) for v in value]
                cur = getattr(ep, name)
                update[name] = cur + [n for n in new if n not in cur]
        except ValidationError as e:
            raise BundleError(path, f"{ep.id}.{name}: {_verr(e)}", _line(vnode)) from e
    return ep.model_copy(update=update)


def override_warnings(bundle: Bundle, path: Path, shown_as: Path | str | None = None) -> list[str]:
    """Problems overrides.yaml at `path` has against `bundle`, as messages: overrides for
    episodes that no longer exist (the warnings apply_overrides raises) and, if the file no
    longer loads at all, that error. Used after a refresh and by `bundle edit`."""
    out: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            apply_overrides(bundle, path)
        except BundleError as e:
            out.append(f"overrides do not apply to this bundle: {e}")
        out += [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    if shown_as is not None:
        out = [m.replace(str(path), str(shown_as)) for m in out]
    return out
