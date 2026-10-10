"""Plan tokens (spec §4.3, decision 60): a short hash of the exact moves of a plan, so that
`identify --apply --confirm TOKEN` consents to the plan that was shown and to nothing else."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Iterable

from showvenist.rename.types import Plan


def plan_token(plans: Iterable[Plan]) -> str:
    """sha256 over every move of every plan, in plan order, as `xxxx-xxxx`."""
    h = hashlib.sha256()
    for plan in plans:
        for m in plan.moves:
            for part in (m.kind, str(m.old), str(m.new), m.fingerprint):
                b = part.encode("utf-8", "surrogateescape")
                h.update(len(b).to_bytes(4, "big"))
                h.update(b)
        h.update(b"\x00plan\x00")
    d = h.hexdigest()[:8]
    return f"{d[:4]}-{d[4:]}"


def normalize_token(s: str) -> str:
    """Lowercase, without dashes or spaces: the 8 hex characters. ValueError on a bad form."""
    t = re.sub(r"[-\s]", "", s).lower()
    if not re.fullmatch(r"[0-9a-f]{8}", t):
        raise ValueError(f"not a plan token: {s!r} (expected 8 hex characters, e.g. 7f3a-9c1e)")
    return t


def _path():
    from showvenist.config import cache_dir

    return cache_dir() / "last-plan"


def save_last_token(token: str) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(token + "\n")
    os.replace(tmp, p)


def read_last_token() -> str | None:
    try:
        return _path().read_text().strip() or None
    except FileNotFoundError:
        return None
