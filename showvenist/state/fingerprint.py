"""File fingerprinting and content keys (spec §7.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK = 64 * 1024


def _read_at(f, offset: int, size: int) -> bytes:
    f.seek(max(0, min(offset, size - CHUNK)))
    return f.read(CHUNK)


def fingerprint(path: Path) -> str:
    """"<size>:<blake2b hex of first and last 64 KB>". Files under 128 KB hash the (overlapping)
    head and tail windows, which still depends on the whole file."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        h.update(_read_at(f, 0, size))
        h.update(_read_at(f, size - CHUNK, size))
    return f"{size}:{h.hexdigest()}"


def content_key(path: Path) -> str:
    """"<size>:<blake2b hex of 64 KB at 25/50/75% of the file>": skips the MakeMKV header."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        for num in (1, 2, 3):
            h.update(_read_at(f, size * num // 4, size))
    return f"{size}:{h.hexdigest()}"
