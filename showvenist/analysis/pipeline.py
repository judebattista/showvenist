"""Concurrent stage 1 + 2 analysis of many files (spec §8.5).

Two stages, pipelined:

- **I/O stage**, limited per storage root (``io_concurrency`` files reading at once, default
  1): content key, fingerprint, probe and every chunk decode (demux, bitmap decode, render)
  run while holding the root's gate. Parallel reads on one HDD array or network share thrash.
- **Compute stage**: Tesseract OCR of the decoded images in a shared pool of
  ``cfg.ocr_workers`` threads (0 = CPU count), each running one ``tesseract`` process per batch
  of images (``ocr.ocr_batch``).

Each file runs the ordinary ``windows.extract_dialogue`` in its own job thread with a reader
that decodes under the gate and then waits for the pool, so while file N waits for its OCR the
gate passes to file N+1: reading the next file overlaps with OCR of the current one. A few files
per root are in flight; the gate goes to the earliest file waiting, so files finish roughly in
input order and two files of one root never read at the same time. Because the window logic
itself is unchanged (and the batched OCR reads each image exactly as a single-image call does),
results and cache entries are identical to calling ``extract_dialogue`` file by file.

    results = analyze_files(paths, language="en", vocabulary=vocab, cfg=cfg.analysis,
                            cache=cache, progress=cb)
"""

from __future__ import annotations

import heapq
import itertools
import os
import threading
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from showvenist.analysis.ocr import _pool_size, batches, ocr_batch
from showvenist.analysis.probe import probe
from showvenist.analysis.types import AnalysisCache, Dialogue, MediaProbe, SubtitleTrack
from showvenist.analysis.windows import decode_chunk, extract_dialogue, finish_chunk
from showvenist.config import AnalysisConfig
from showvenist.state.fingerprint import content_key, fingerprint

# progress(done, total, path, error): called once per input path, from the calling thread
Progress = Callable[[int, int, Path, str | None], None]

# Files in flight per root beyond its I/O slots: one is OCR'd while the next reads, and a
# spare keeps the reader busy when OCR of two files is outstanding.
LOOKAHEAD = 2


@dataclass
class FileResult:
    """Stage 1 + 2 of one input path. ``error`` is set (and the rest may be None) when the file
    couldn't be read. ``duplicate_of`` is set when an earlier input, or a path in
    ``known_keys``, has the same content key: such a file is not probed or read. It equals
    ``path`` itself when ``known_keys`` maps the key to this very path (already analyzed)."""

    path: Path
    probe: MediaProbe | None = None
    fingerprint: str | None = None
    content_key: str | None = None
    dialogue: Dialogue | None = None
    error: str | None = None
    duplicate_of: Path | None = None


def storage_root(path: Path) -> Hashable:
    """Files on one storage root share its I/O limit: the device id, or, when the path can't be
    stat'ed, its first two components."""
    try:
        return ("dev", os.stat(path).st_dev)
    except OSError:
        return ("path", Path(os.path.abspath(path)).parts[:2])


class _Gate:
    """``slots`` holders at once; waiters are served lowest priority value first."""

    def __init__(self, slots: int):
        self._cond = threading.Condition()
        self._free = slots
        self._waiting: list[tuple[int, int]] = []
        self._seq = itertools.count()

    @contextmanager
    def hold(self, priority: int) -> Iterator[None]:
        me = (priority, next(self._seq))
        with self._cond:
            heapq.heappush(self._waiting, me)
            while self._free == 0 or self._waiting[0] != me:
                self._cond.wait()
            heapq.heappop(self._waiting)
            self._free -= 1
            self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                self._free += 1
                self._cond.notify_all()


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


class _Pipeline:
    def __init__(self, roots: Iterable[Hashable], io_slots: int, ocr: ThreadPoolExecutor, split):
        self.gates = {r: _Gate(io_slots) for r in roots}
        self.ocr_pool = ocr
        self.split = split  # images -> batch count hint (workers argument of ocr.batches)

    def ocr(self, images: Sequence, language: str) -> list[str]:
        parts = batches(len(images), self.split)
        futs = [self.ocr_pool.submit(ocr_batch, [images[i] for i in r], language) for r in parts]
        return [t for f in futs for t in f.result()]

    def reader(self, gate: _Gate, prio: int):
        def read(path: Path, track: SubtitleTrack, a: float, b: float, language: str, cfg):
            with gate.hold(prio):
                d = decode_chunk(path, track, a, b)
            return finish_chunk(d, self.ocr(d.images, language) if d.events else [])

        return read


def analyze_files(
    paths: Sequence[Path | str],
    *,
    language: str,
    vocabulary: set[str],
    cfg: AnalysisConfig,
    cache: AnalysisCache | None,
    progress: Progress | None = None,
    io_concurrency: int | None = None,
    known_keys: Mapping[str, Path] | None = None,
) -> list[FileResult]:
    """Stage 1 + 2 (``probe`` + ``extract_dialogue``) for every path, pipelined (module doc).
    Returns one FileResult per input, in input order. A failing file gets ``error`` and never
    aborts the batch. ``io_concurrency`` overrides ``cfg.io_concurrency`` (files reading at
    once per storage root). ``known_keys`` maps content keys already analyzed (e.g. by an
    earlier call) to their path; inputs with such a key get ``duplicate_of`` and aren't read."""
    paths = [Path(p) for p in paths]
    n = len(paths)
    results = [FileResult(path=p) for p in paths]
    if not n:
        return results
    slots = max(1, io_concurrency if io_concurrency is not None else cfg.io_concurrency)
    roots = [storage_root(p) for p in paths]
    by_root: dict[Hashable, list[int]] = {}
    for i, r in enumerate(roots):
        by_root.setdefault(r, []).append(i)
    workers = _pool_size(cfg.ocr_workers)
    in_flight = len(by_root) * (slots + LOOKAHEAD)
    # A chunk's images are split so the files in flight together fill the pool.
    split = max(1, workers // min(n, in_flight))
    pipe = _Pipeline(by_root, slots, ThreadPoolExecutor(workers, "sv-ocr"), split)
    done = 0

    def report(i: int) -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, n, paths[i], results[i].error)

    def keys(i: int) -> None:
        r = results[i]
        with pipe.gates[roots[i]].hold(i):
            r.content_key = content_key(r.path)
            r.fingerprint = fingerprint(r.path)

    def analyze(i: int) -> None:
        r = results[i]
        gate = pipe.gates[roots[i]]
        with gate.hold(i):
            r.probe = probe(r.path, cache)
        r.dialogue = extract_dialogue(
            r.path,
            language,
            vocabulary,
            cfg,
            cache,
            probe_result=r.probe,
            reader=pipe.reader(gate, i),
        )

    def guarded(fn: Callable[[int], None], i: int) -> None:
        try:
            fn(i)
        except Exception as e:  # unreadable/corrupt media: record, keep going
            results[i].error = _err(e)

    try:
        # Content keys first (small reads), so duplicates resolve in input order as they do
        # when files are analyzed one by one.
        with ThreadPoolExecutor(len(by_root), "sv-keys") as kp:
            list(kp.map(lambda idx: [guarded(keys, i) for i in idx], by_root.values()))
        seen: dict[str, Path] = dict(known_keys or {})
        todo: list[int] = []  # primaries, in input order
        for i, r in enumerate(results):
            if r.error is None and r.content_key is not None:
                if r.content_key in seen:
                    r.duplicate_of = seen[r.content_key]
                else:
                    seen[r.content_key] = r.path
                    todo.append(i)
        for i in sorted(set(range(n)) - set(todo)):
            report(i)
        execs = {
            root: ThreadPoolExecutor(min(len(idx), slots + LOOKAHEAD), "sv-file")
            for root, idx in by_root.items()
        }
        try:
            futs: dict[Future, int] = {execs[roots[i]].submit(guarded, analyze, i): i for i in todo}
            for f in as_completed(futs):
                report(futs[f])
        finally:
            for ex in execs.values():  # queued files are dropped when a caller stops early
                ex.shutdown(wait=True, cancel_futures=True)
    finally:
        pipe.ocr_pool.shutdown(wait=True)
    return results
