"""Bundle storage and retrieval (spec §6.2)."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

from pydantic import ValidationError

from showvenist.bundle.overrides import BundleError
from showvenist.bundle.schema import Bundle
from showvenist.config import bundles_dir

__all__ = [
    "BundleError",
    "bundle_path",
    "list_bundles",
    "load_bundle",
    "save_bundle",
    "slugify",
]


def slugify(title: str, year: int | None = None) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "show"
    return f"{s}-{year}" if year else s


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else bundles_dir()


def bundle_path(slug: str, root: Path | None = None) -> Path:
    """Directory of the bundle for `slug`."""
    return _root(root) / slug


def save_bundle(bundle: Bundle, root: Path | None = None, slug: str | None = None) -> Path:
    """Write show.json atomically; returns its path. Revision is recomputed. `slug` defaults to
    the one the show's title and year give (a refresh keeps the bundle's existing slug)."""
    d = bundle_path(slug or slugify(bundle.show.title, bundle.show.year), root)
    # title_card comes from overrides.yaml only; it never reaches show.json or its revision
    bundle = bundle.model_copy(update={"title_card": None}).with_revision(d)
    d.mkdir(parents=True, exist_ok=True)
    target = d / "show.json"
    data = bundle.model_dump(mode="json", exclude={"title_card"})
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".show.json.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def load_bundle(slug: str, root: Path | None = None, apply_overrides: bool = True) -> Bundle:
    from showvenist.bundle import overrides as _ov

    d = bundle_path(slug, root)
    path = d / "show.json"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise BundleError(path, "bundle not found") from e
    try:
        bundle = Bundle.model_validate_json(text)
    except json.JSONDecodeError as e:
        raise BundleError(path, f"invalid JSON: {e.msg}", e.lineno) from e
    except ValidationError as e:
        # pydantic wraps JSON syntax errors in ValidationError too
        errs = e.errors()
        if errs and errs[0]["type"] == "json_invalid":
            try:
                json.loads(text)
            except json.JSONDecodeError as je:
                raise BundleError(path, f"invalid JSON: {je.msg}", je.lineno) from e
        raise BundleError(path, _ov._verr(e)) from e
    ov = d / "overrides.yaml"
    if apply_overrides and ov.exists():
        return _ov.apply_overrides(bundle, ov)
    return bundle


def list_bundles(root: Path | None = None) -> list[str]:
    r = _root(root)
    if not r.is_dir():
        return []
    return sorted(p.name for p in r.iterdir() if (p / "show.json").is_file())
