#!/usr/bin/env python3
"""
Spike: Extract SxxEyy labels from MKV filenames in the four season directories.
PEP 723 script: uv run spike/labels.py
"""

# /// script
# requires-python = ">=3.11"
# ///

import csv
import re
from pathlib import Path

# Mapping: path -> (show_key, season_num)
SEASONS = [
    ("/mnt/storage/tv/Supernatural/Season02", "supernatural-s02", 2),
    ("/mnt/storage/tv/The_Incredible_Hulk/Season_02", "the-incredible-hulk-s02", 2),
    ("/mnt/storage/tv/The_West_Wing/Season_01", "the-west-wing-s01", 1),
    ("/mnt/storage/tv/The_Invisible_Man/Season_01", "the-invisible-man-s01", 1),
]

OUT_DIR = Path(__file__).parent / "out"
OUT_CSV = OUT_DIR / "labels.csv"

# Regex to match SxxEyy pattern (case-insensitive)
# Match s followed by 1-2 digits, e, then 1-2 digits
PATTERN = re.compile(r"s(\d{1,2})e(\d{1,2})", re.IGNORECASE)


def extract_label(filename: str) -> str | None:
    """Extract SxxEyy label from filename. Return None if not found or is a duplicate (has suffix like 'a')."""
    match = PATTERN.search(filename)
    if not match:
        return None

    # Get the full match
    label_match = match.group(0)
    end_pos = match.end()

    # Check if there's a non-digit, non-dot suffix after the label (like 'a' or 'b')
    # We want to skip files like "s02e09a.mkv"
    if end_pos < len(filename):
        next_char = filename[end_pos]
        # If next char is a letter (except in extension), it's a duplicate
        if next_char.isalpha() and next_char not in 'mkv':
            return None

    return label_match.upper()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for dir_path, show_key, season_num in SEASONS:
        dir_obj = Path(dir_path)
        if not dir_obj.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue

        # List all .mkv files
        for file_path in sorted(dir_obj.glob("*.mkv")):
            label = extract_label(file_path.name)
            if label:
                rows.append({
                    "path": str(file_path),
                    "show": show_key,
                    "season": season_num,
                    "label": label,
                })

    # Write CSV
    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "show", "season", "label"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {OUT_CSV} with {len(rows)} entries")
    else:
        print("No labels found")


if __name__ == "__main__":
    main()
