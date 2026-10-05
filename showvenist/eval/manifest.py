"""Manifest CSV handling (spec §16.1): path,show,label,order,disc."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from showvenist.eval.types import ManifestRow

COLUMNS = ("path", "show", "label", "order", "disc")
_REQUIRED = ("path", "show", "label")


def read_manifest(path: str | Path) -> list[ManifestRow]:
    """Read a manifest. Rows with an empty label are skipped; `order` defaults to "aired".

    Raises ValueError when a required column is missing or a labeled row has no path or show.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        missing = [c for c in _REQUIRED if c not in header]
        if missing:
            raise ValueError(f"{path}: manifest is missing column(s) {', '.join(missing)}")
        reader.fieldnames = header
        rows: list[ManifestRow] = []
        for line, raw in enumerate(reader, start=2):
            get = {k: (raw.get(k) or "").strip() for k in COLUMNS}.__getitem__
            label = get("label")
            if not label:
                continue
            if not get("path") or not get("show"):
                raise ValueError(f"{path}:{line}: labeled row needs both path and show")
            rows.append(
                ManifestRow(
                    path=get("path"),
                    show=get("show"),
                    label=label,
                    order=get("order") or "aired",
                    disc=get("disc"),
                )
            )
    return rows


def write_manifest(rows: Iterable[ManifestRow], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow([r.path, r.show, r.label, r.order, r.disc])
