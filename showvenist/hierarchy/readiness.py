"""File readiness detection (spec §5.2): stable size/mtime and a structurally complete file."""

from __future__ import annotations

import os
import struct
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from showvenist.hierarchy.types import Readiness, default_stat

_EBML = 0x1A45DFA3
_SEGMENT = 0x18538067
_INFO = 0x1549A966
_DURATION = 0x4489
_CUES = 0x1C53BB6B
_SEEKHEAD = 0x114D9B74
_SEEK = 0x4DBB
_SEEK_ID = 0x53AB
_SEEK_POS = 0x53AC

_UNKNOWN = -1
_MAX_INFO = 1 << 20


def _read_id(f) -> tuple[int, int] | None:
    b = f.read(1)
    if not b or b[0] == 0:
        return None
    n = 9 - b[0].bit_length()
    if n > 4:
        return None
    rest = f.read(n - 1)
    if len(rest) != n - 1:
        return None
    return int.from_bytes(b + rest, "big"), n


def _read_size(f) -> tuple[int, int] | None:
    b = f.read(1)
    if not b or b[0] == 0:
        return None
    n = 9 - b[0].bit_length()
    rest = f.read(n - 1)
    if len(rest) != n - 1:
        return None
    val = int.from_bytes(bytes([b[0] & (0xFF >> n)]) + rest, "big")
    if val == (1 << (7 * n)) - 1:
        return _UNKNOWN, n
    return val, n


def _header(f) -> tuple[int, int, int] | None:
    """Return (id, data_size or _UNKNOWN, header_length) at the current position."""
    i = _read_id(f)
    if i is None:
        return None
    s = _read_size(f)
    if s is None:
        return None
    return i[0], s[0], i[1] + s[1]


def _children(data: bytes):
    """Yield (id, payload) for the elements in a small in-memory master element."""
    pos = 0
    while pos < len(data):
        b = data[pos]
        if b == 0:
            return
        n = 9 - b.bit_length()
        eid = int.from_bytes(data[pos : pos + n], "big")
        pos += n
        if pos >= len(data):
            return
        b = data[pos]
        if b == 0:
            return
        m = 9 - b.bit_length()
        size = int.from_bytes(bytes([b & (0xFF >> m)]) + data[pos + 1 : pos + m], "big")
        pos += m
        yield eid, data[pos : pos + size]
        pos += size


def _duration(info: bytes) -> float:
    for eid, payload in _children(info):
        if eid == _DURATION:
            if len(payload) == 4:
                return struct.unpack(">f", payload)[0]
            if len(payload) == 8:
                return struct.unpack(">d", payload)[0]
    return 0.0


def _cues_position(seekhead: bytes) -> int | None:
    for eid, payload in _children(seekhead):
        if eid != _SEEK:
            continue
        sid = pos = None
        for cid, cp in _children(payload):
            if cid == _SEEK_ID:
                sid = int.from_bytes(cp, "big")
            elif cid == _SEEK_POS:
                pos = int.from_bytes(cp, "big")
        if sid == _CUES and pos is not None:
            return pos
    return None


def matroska_status(path: Path) -> str | None:
    """Return None when the file is a complete Matroska, else the reason it is not."""
    try:
        with open(path, "rb") as f:
            fsize = os.fstat(f.fileno()).st_size
            h = _header(f)
            if h is None or h[0] != _EBML:
                return "not matroska"
            f.seek(h[2] + h[1])
            h = _header(f)
            if h is None:
                return "truncated"
            if h[0] != _SEGMENT:
                return "not matroska"
            seg_data = f.tell()
            seg_end = fsize if h[1] == _UNKNOWN else seg_data + h[1]
            if seg_end > fsize:
                seg_end = fsize  # segment size claims more than is on disk
                size_short = True
            else:
                size_short = False
            has_duration = False
            has_cues = False
            cues_hint: int | None = None
            pos = seg_data
            while pos < seg_end:
                f.seek(pos)
                h = _header(f)
                if h is None:
                    return "truncated"
                eid, size, hl = h
                body = pos + hl
                if size == _UNKNOWN:
                    break  # unknown-size cluster: cannot skip it
                end = body + size
                if end > fsize:
                    return "truncated"
                if eid == _INFO and size <= _MAX_INFO:
                    has_duration = _duration(f.read(size)) > 0
                elif eid == _SEEKHEAD and size <= _MAX_INFO:
                    cp = _cues_position(f.read(size))
                    if cp is not None:
                        cues_hint = seg_data + cp
                elif eid == _CUES:
                    has_cues = True
                    break
                pos = end
                if cues_hint is not None and pos < cues_hint <= fsize - 4 and has_duration:
                    # jump straight to where the SeekHead says the Cues are
                    f.seek(cues_hint)
                    ch = _header(f)
                    if ch is not None and ch[0] == _CUES:
                        if cues_hint + ch[2] + ch[1] > fsize:
                            return "truncated"
                        has_cues = True
                        break
            if not has_duration:
                return "no duration"
            if not has_cues:
                return "truncated" if size_short else "no cues"
            return None
    except OSError:
        return "unreadable"


def matroska_complete(path: Path) -> bool:
    return matroska_status(path) is None


def _av_openable(path: Path) -> bool:
    try:
        import av

        with av.open(str(path)):
            return True
    except Exception:
        return False


class ReadinessChecker:
    def __init__(
        self,
        stable_seconds: float = 60,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        settle_seconds: float = 1.0,
        stat: Callable[[Path], os.stat_result] = default_stat,
    ) -> None:
        self.stable_seconds = stable_seconds
        self.clock = clock
        self.sleep = sleep
        self.settle_seconds = settle_seconds
        self.stat = stat

    def check(self, paths: Sequence[Path]) -> dict[Path, Readiness]:
        out: dict[Path, Readiness] = {}
        first: dict[Path, os.stat_result] = {}
        now = self.clock()
        for p in paths:
            try:
                st = self.stat(p)
            except OSError:
                out[p] = Readiness(False, "unreadable")
                continue
            age = now - st.st_mtime
            if age < self.stable_seconds:
                out[p] = Readiness(False, f"modified {max(age, 0):.0f} s ago")
            else:
                first[p] = st
        if first:
            self.sleep(self.settle_seconds)  # one shared settle interval for the batch
        for p, st in first.items():
            try:
                st2 = self.stat(p)
            except OSError:
                out[p] = Readiness(False, "unreadable")
                continue
            if st2.st_size != st.st_size:
                out[p] = Readiness(False, "still growing")
            elif st2.st_mtime != st.st_mtime:
                out[p] = Readiness(False, "mtime changed")
            else:
                out[p] = self._structure(p)
        return {p: out[p] for p in paths}

    @staticmethod
    def _structure(p: Path) -> Readiness:
        if p.suffix.lower() in (".mkv", ".mka", ".webm"):
            reason = matroska_status(p)
            return Readiness(reason is None, reason)
        if _av_openable(p):
            return Readiness(True)
        return Readiness(False, "unreadable")
