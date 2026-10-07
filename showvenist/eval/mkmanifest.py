"""Build manifest CSV from hand-named files and labeling sheet (spec §16.1–16.2).

Scans for files matching s<season>e<episode> pattern (case-insensitive, optionally
with letter suffix like s02e09a), extracts episode labels (normalized to S02E09 form),
show slugs (from a slugs dict or guessed from directory name), disc numbers (from path),
and episode orderings. Also reads a hand-labeling sheet for unnamed rips.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from showvenist.hierarchy.parse import walk_media

# Patterns for named files and disc numbers
_NAMED = re.compile(r"[sS](\d{1,2})[eE](\d{1,3})([a-z])?", re.I)
_RIP = re.compile(r"_t(\d\d)\.mkv$", re.I)
_DISC = re.compile(r"disc[ _-]?(\d+)", re.I)

# Default ordering override for known shows
# Samurai Jack: the Blu-ray disks split TMDB's combined entries, so labels use the `disk`
# ordering defined in the bundle's overrides.yaml
DEFAULT_ORDERS = {"Firefly": "intended-order", "Samurai_Jack": "disk"}


def _slug_from_dirname(dirname: str) -> str:
    """Guess a bundle slug from a directory name: lowercase, non-alphanumerics → '-'."""
    # Remove year patterns like (2002) or -2002
    dirname = re.sub(r"[(\-]?20\d{2}[)]*", "", dirname)
    # Convert to lowercase and replace non-alphanumerics with '-'
    slug = re.sub(r"[^a-z0-9]+", "-", dirname.lower())
    # Strip leading/trailing dashes
    slug = slug.strip("-")
    return slug


def _normalize_label(season: int, episode: int) -> str:
    """Normalize episode label to uppercase S##E## format."""
    if episode >= 100:
        return f"S{season:02d}E{episode:03d}".rstrip("E").replace("E0", "E")
    return f"S{season:02d}E{episode:02d}"


def _get_disc(path: str) -> str:
    """Extract disc number from path, e.g. 'disc_1', 'disc-2', etc."""
    match = _DISC.search(path)
    return match.group(1) if match else ""


def build_rows(
    root: Path,
    slugs: dict[str, str] | None = None,
    orders: dict[str, str] | None = None,
) -> list[tuple[str, str, str, str, str]]:
    """Build manifest rows from hand-named files under root.

    Args:
        root: Root directory to scan for named files
        slugs: Optional dict mapping dir name -> bundle slug (guessed if not provided)
        orders: Optional dict mapping show dir -> ordering; defaults include
            {"Firefly": "intended-order"}

    Returns:
        List of tuples: (path, show, label, order, disc)
        where path is relative to root, label is "S02E09" format (normalized and uppercased)
    """
    if slugs is None:
        slugs = {}
    if orders is None:
        orders = {}
    # Merge with defaults
    orders = {**DEFAULT_ORDERS, **orders}

    rows: list[tuple[str, str, str, str, str]] = []

    # Scan root for MKV files
    for mkv in walk_media(root):
        filename = mkv.name

        # Skip raw rips (files matching *_tNN.mkv)
        if _RIP.search(filename):
            continue

        # Check if file matches named pattern (sXeY format)
        match = _NAMED.search(filename)
        if not match:
            continue

        season = int(match.group(1))
        episode = int(match.group(2))
        # suffix = match.group(3)  # Letter suffix like 'a' in s02e09a - we drop this for the label

        # Get relative path
        rel_path = str(mkv.relative_to(root))

        # Extract show from directory name (first path component)
        show_dir = mkv.relative_to(root).parts[0]
        show = slugs.get(show_dir, _slug_from_dirname(show_dir))

        # Normalize label (uppercase, drop suffix)
        label = _normalize_label(season, episode)

        # Get ordering
        order = orders.get(show_dir, "aired")

        # Get disc number
        disc = _get_disc(rel_path)

        rows.append((rel_path, show, label, order, disc))

    return rows


def sheet_to_rows(
    sheet_csv: Path, orders: dict[str, str] | None = None
) -> list[tuple[str, str, str, str, str]]:
    """Convert hand-labeling sheet to manifest rows.

    Reads data/extras-sheet.csv and produces manifest rows:
    - Rows with non-empty label become manifest rows
    - Rows with same_as set and empty label inherit the label of the referenced row
    - Rows with empty label and no inherited label are skipped

    Args:
        sheet_csv: Path to extras-sheet.csv

    Returns:
        List of tuples: (path, show, label, order, disc)
    """
    # First pass: collect all rows with labels
    rows_by_path: dict[str, dict] = {}
    all_rows: list[dict] = []

    with sheet_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)
            label_val = (row.get("label") or "").strip()
            if label_val:
                rows_by_path[row.get("path", "")] = row

    # Second pass: resolve inheritance and build output
    result: list[tuple[str, str, str, str, str]] = []
    for row in all_rows:
        label = (row.get("label") or "").strip()

        # If no label, check for same_as inheritance
        if not label:
            same_as = (row.get("same_as") or "").strip()
            if same_as and same_as in rows_by_path:
                label = (rows_by_path[same_as].get("label") or "").strip()

        # Skip if no label (either original or inherited)
        if not label:
            continue

        # Normalize label case if it's an episode label (or a two-part 'S01E01+S01E02' label)
        label = "+".join(_normalize_part(part) for part in label.split("+"))

        # Extract show (from first path component)
        path = row.get("path", "") or ""
        parts = path.split("/")
        show = parts[0] if parts else ""

        # Get ordering from row or default
        # the labels are written in the user's disk numbering: per-show ordering, else aired
        default = {**DEFAULT_ORDERS, **(orders or {})}.get(show, "aired")
        order = ((row.get("order") or "") or default).strip() or default

        # Get disc from row
        disc = (row.get("disc") or "").strip()

        result.append((path, show, label, order, disc))

    return result


def _normalize_part(part: str) -> str:
    """'s01e02' -> 'S01E02'; 'e02' -> 'E02' (the season of a two-part label's later parts is
    inherited by the harness); anything else (extra, extra-multi, episode?) is lower-cased."""
    part = part.strip()
    m = re.fullmatch(r"[sS](\d{1,2})[eE](\d{1,3})", part)
    if m:
        return _normalize_label(int(m.group(1)), int(m.group(2)))
    m = re.fullmatch(r"[eE](\d{1,3})", part)
    if m:
        return f"E{int(m.group(1)):02d}"
    # tolerate a show prefix ('the_wire_s03e05'): the show is in its own column
    m = re.search(r"(?<![a-z0-9])[sS](\d{1,2})[eE](\d{1,3})(?![0-9])", part)
    if m and part.lower() not in ("extra", "extra-multi", "episode?"):
        return _normalize_label(int(m.group(1)), int(m.group(2)))
    return part.lower() if part.lower() in ("extra", "extra-multi", "episode?") else part


def merge_sample_labels(sample_csv: Path, full_csv: Path, dest: Path) -> int:
    """Copy the labels typed into a sample sheet onto the full sheet (matched by path), so that
    exact duplicates in the full sheet inherit them. Returns the number of labels copied."""
    with sample_csv.open(encoding="utf-8") as f:
        labels = {r["path"]: (r.get("label") or "").strip() for r in csv.DictReader(f)}
    with full_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    n = 0
    for r in rows:
        lab = labels.get(r["path"], "")
        if lab and not (r.get("label") or "").strip():
            r["label"] = lab
            n += 1
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    return n


def bundle_slugs(bundles_dir: Path, library_root: Path) -> dict[str, str]:
    """Library folder name -> bundle slug, matching on the compact name ('The_West_Wing' ->
    'thewestwing' is a prefix of 'thewestwing1999' for the slug 'the-west-wing-1999')."""

    def compact(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    slugs = [p.name for p in bundles_dir.iterdir() if (p / "show.json").exists()]
    out: dict[str, str] = {}
    for d in library_root.iterdir():
        if not d.is_dir():
            continue
        hits = [s for s in slugs if compact(s).startswith(compact(d.name))]
        if len(hits) == 1:
            out[d.name] = hits[0]
    return out


def write_csv(rows: list[tuple[str, str, str, str, str]], dest: Path) -> None:
    """Write manifest rows to CSV file.

    Args:
        rows: List of tuples (path, show, label, order, disc)
        dest: Destination CSV file path
    """
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "show", "label", "order", "disc"])
        writer.writerows(rows)


def main() -> None:
    """CLI entry point: python -m showvenist.eval.mkmanifest"""
    parser = argparse.ArgumentParser(
        description="Build manifest CSV from hand-named files and labeling sheet"
    )
    parser.add_argument("root", type=Path, help="Root directory to scan")
    parser.add_argument(
        "--sheet",
        type=Path,
        default=None,
        help="Path to extras-sheet.csv (optional)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output manifest CSV file",
    )

    parser.add_argument(
        "--sample", type=Path, default=None,
        help="Labeled sample sheet; its labels are merged into --sheet first (duplicates inherit)",
    )
    parser.add_argument(
        "--bundles", type=Path, default=None,
        help="Bundle root: map library folder names to bundle slugs",
    )

    args = parser.parse_args()
    slugs = bundle_slugs(args.bundles, args.root) if args.bundles else None
    orders: dict[str, str] = {}

    # Build rows from hand-named files
    rows = build_rows(args.root, slugs=slugs, orders=orders)

    # If sheet is provided, also add rows from sheet
    if args.sheet:
        sheet = args.sheet
        if args.sample:
            merged = args.output.with_suffix(".sheet.csv")
            n = merge_sample_labels(args.sample, args.sheet, merged)
            print(f"Copied {n} labels from the sample onto the full sheet")
            sheet = merged
        sheet_rows = sheet_to_rows(sheet, orders)
        if slugs:
            sheet_rows = [(p, slugs.get(s, s), lab, o, d) for p, s, lab, o, d in sheet_rows]
        rows.extend(sheet_rows)

    # Write to output
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
