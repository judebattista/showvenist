import pytest

from showvenist.eval.manifest import read_manifest, write_manifest
from showvenist.eval.types import ManifestRow


def test_read_skips_empty_labels_defaults_order_and_trims(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(
        "path,show,label,order,disc\n"
        " /a/s01e01.mkv , firefly-2002 , S01E01 , dvd ,\n"
        "/a/t04.mkv,firefly-2002,extra,,D1\n"
        "/a/t05.mkv,firefly-2002,,,\n"
        "/a/t06.mkv,firefly-2002,  ,,\n"
    )
    rows = read_manifest(p)
    assert rows == [
        ManifestRow("/a/s01e01.mkv", "firefly-2002", "S01E01", "dvd", ""),
        ManifestRow("/a/t04.mkv", "firefly-2002", "extra", "aired", "D1"),
    ]


def test_optional_columns_and_bom(tmp_path):
    p = tmp_path / "m.csv"
    p.write_bytes("﻿path,show,label\n/x.mkv,s,episode?\n".encode())
    assert read_manifest(p) == [ManifestRow("/x.mkv", "s", "episode?")]


def test_missing_column_and_bad_row(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("path,label\n/x.mkv,extra\n")
    with pytest.raises(ValueError, match="show"):
        read_manifest(p)
    p.write_text("path,show,label\n,s,extra\n")
    with pytest.raises(ValueError, match=":2:"):
        read_manifest(p)


def test_round_trip(tmp_path):
    rows = [
        ManifestRow("/a b/c,d.mkv", "show", "S02E10", "intended-order", "2"),
        ManifestRow("/e.mkv", "show", "extra"),
    ]
    p = tmp_path / "m.csv"
    write_manifest(rows, p)
    assert p.read_text().splitlines()[0] == "path,show,label,order,disc"
    assert read_manifest(p) == rows


def test_empty(tmp_path):
    p = tmp_path / "m.csv"
    write_manifest([], p)
    assert read_manifest(p) == []
