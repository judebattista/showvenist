"""Tests for mkmanifest module (spec §16.1–16.2)."""


import pytest

from showvenist.eval.mkmanifest import build_rows, sheet_to_rows, write_csv


@pytest.fixture
def tmp_lib(tmp_path):
    """Create a temporary library with hand-named files."""
    return tmp_path / "lib"


@pytest.fixture
def tmp_sheet(tmp_path):
    """Create a temporary extras-sheet.csv."""
    return tmp_path / "extras-sheet.csv"


def test_build_rows_hand_named_files(tmp_lib):
    """Test scanning for hand-named files with episode labels."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "Show"
    show_dir.mkdir()

    # Create some named files
    (show_dir / "show_s01e01.mkv").touch()
    (show_dir / "show_s01e02.mkv").touch()
    (show_dir / "show_s02e05.mkv").touch()

    rows = build_rows(tmp_lib)

    assert len(rows) == 3
    assert rows[0][0] == "Show/show_s01e01.mkv"  # path
    assert rows[0][1] == "show"  # show (guessed from dir)
    assert rows[0][2] == "S01E01"  # label (normalized)
    assert rows[0][3] == "aired"  # order (default)
    assert rows[0][4] == ""  # disc (empty)


def test_build_rows_ignores_raw_rips(tmp_lib):
    """Test that raw rips (*_tNN.mkv) are ignored."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "Firefly"
    show_dir.mkdir()

    (show_dir / "title_t00.mkv").touch()
    (show_dir / "title_t01.mkv").touch()
    (show_dir / "firefly_s01e01.mkv").touch()

    rows = build_rows(tmp_lib)

    assert len(rows) == 1
    assert rows[0][0] == "Firefly/firefly_s01e01.mkv"


def test_build_rows_letter_suffix(tmp_lib):
    """Test that letter suffixes are handled (duplicate markers)."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "Supernatural" / "Season02"
    show_dir.mkdir(parents=True)

    # Files with same episode but letter suffix (duplicates)
    (show_dir / "supernatural_s02e09.mkv").touch()
    (show_dir / "supernatural_s02e09a.mkv").touch()
    (show_dir / "supernatural_s02e09b.mkv").touch()

    rows = build_rows(tmp_lib)

    assert len(rows) == 3
    # All should have the same label (suffix is dropped)
    assert rows[0][2] == "S02E09"
    assert rows[1][2] == "S02E09"
    assert rows[2][2] == "S02E09"


def test_build_rows_disc_extraction(tmp_lib):
    """Test disc number extraction from path."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "Samurai_Jack" / "Season01"
    show_dir.mkdir(parents=True)

    disc1_dir = show_dir / "disc1"
    disc1_dir.mkdir()
    disc2_dir = show_dir / "disc2"
    disc2_dir.mkdir()

    (disc1_dir / "samurai_jack_s01e01.mkv").touch()
    (disc2_dir / "samurai_jack_s01e05.mkv").touch()

    rows = build_rows(tmp_lib)

    assert len(rows) == 2
    assert rows[0][4] == "1"  # disc number from first file
    assert rows[1][4] == "2"  # disc number from second file


def test_build_rows_custom_slugs(tmp_lib):
    """Test providing custom show slugs."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "The_West_Wing"
    show_dir.mkdir()

    (show_dir / "the_west_wing_s01e01.mkv").touch()

    slugs = {"The_West_Wing": "west-wing-1999"}
    rows = build_rows(tmp_lib, slugs=slugs)

    assert rows[0][1] == "west-wing-1999"


def test_build_rows_custom_orders(tmp_lib):
    """Test providing custom orderings."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "Firefly"
    show_dir.mkdir()

    (show_dir / "firefly_s01e01.mkv").touch()

    # Default should be "intended-order" for Firefly
    rows = build_rows(tmp_lib)
    assert rows[0][3] == "intended-order"

    # Override it
    orders = {"Firefly": "aired"}
    rows = build_rows(tmp_lib, orders=orders)
    assert rows[0][3] == "aired"


def test_build_rows_case_insensitive(tmp_lib):
    """Test that episode pattern matching is case-insensitive."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "Show"
    show_dir.mkdir()

    (show_dir / "Show_S01E01.mkv").touch()
    (show_dir / "show_s02e02.mkv").touch()
    (show_dir / "SHOW_S03E03.mkv").touch()

    rows = build_rows(tmp_lib)

    assert len(rows) == 3
    # Check that all labels are present (order is not guaranteed by rglob)
    labels = {row[2] for row in rows}
    assert labels == {"S01E01", "S02E02", "S03E03"}


def test_sheet_to_rows_basic(tmp_sheet):
    """Test reading hand-labeling sheet with episode labels."""
    with tmp_sheet.open("w") as f:
        f.write("path,show,season,disc,title_no,duration_min,size_gb,ref_min,vs_episode,hint,title_tag,same_as,label\n")
        f.write("Show/Season01/file1.mkv,Show,S01,,0,44.0,9.0,44,1.00,episode-length,,,S01E01\n")
        f.write("Show/Season01/file2.mkv,Show,S01,,1,44.0,9.0,44,1.00,episode-length,,,extra\n")

    rows = sheet_to_rows(tmp_sheet)

    assert len(rows) == 2
    assert rows[0] == ("Show/Season01/file1.mkv", "Show", "S01E01", "aired", "")
    assert rows[1] == ("Show/Season01/file2.mkv", "Show", "extra", "aired", "")


def test_sheet_to_rows_inheritance(tmp_sheet):
    """Test label inheritance from same_as."""
    with tmp_sheet.open("w") as f:
        f.write("path,show,season,disc,title_no,duration_min,size_gb,ref_min,vs_episode,hint,title_tag,same_as,label\n")
        f.write("Show/Season01/original.mkv,Show,S01,,0,44.0,9.0,44,1.00,episode-length,,,S01E02\n")
        f.write("Show/Season01/duplicate.mkv,Show,S01,,1,44.0,9.0,44,1.00,episode-length,,Show/Season01/original.mkv,\n")

    rows = sheet_to_rows(tmp_sheet)

    # Both rows should be present, with the duplicate inheriting the label
    assert len(rows) == 2
    assert rows[0] == ("Show/Season01/original.mkv", "Show", "S01E02", "aired", "")
    assert rows[1] == ("Show/Season01/duplicate.mkv", "Show", "S01E02", "aired", "")


def test_sheet_to_rows_skip_orphan_duplicates(tmp_sheet):
    """Test that duplicates without valid same_as are skipped."""
    with tmp_sheet.open("w") as f:
        f.write("path,show,season,disc,title_no,duration_min,size_gb,ref_min,vs_episode,hint,title_tag,same_as,label\n")
        f.write("Show/Season01/file1.mkv,Show,S01,,0,44.0,9.0,44,1.00,episode-length,,,S01E01\n")
        f.write("Show/Season01/file2.mkv,Show,S01,,1,44.0,9.0,44,1.00,episode-length,,nonexistent.mkv,\n")

    rows = sheet_to_rows(tmp_sheet)

    # Only the labeled file should be in the result
    assert len(rows) == 1
    assert rows[0][0] == "Show/Season01/file1.mkv"


def test_sheet_to_rows_label_normalization(tmp_sheet):
    """Test that episode labels are normalized to uppercase S##E## format."""
    with tmp_sheet.open("w") as f:
        f.write("path,show,season,disc,title_no,duration_min,size_gb,ref_min,vs_episode,hint,title_tag,same_as,label\n")
        f.write("Show/Season01/file1.mkv,Show,S01,,0,44.0,9.0,44,1.00,episode-length,,,s01e01\n")
        f.write("Show/Season01/file2.mkv,Show,S01,,1,44.0,9.0,44,1.00,episode-length,,,S02e03\n")

    rows = sheet_to_rows(tmp_sheet)

    assert rows[0][2] == "S01E01"
    assert rows[1][2] == "S02E03"


def test_sheet_to_rows_extra_and_episode_question(tmp_sheet):
    """Test that extra and episode? labels are preserved as-is."""
    with tmp_sheet.open("w") as f:
        f.write("path,show,season,disc,title_no,duration_min,size_gb,ref_min,vs_episode,hint,title_tag,same_as,label\n")
        f.write("Show/Season01/file1.mkv,Show,S01,,0,44.0,9.0,44,1.00,episode-length,,,extra\n")
        f.write("Show/Season01/file2.mkv,Show,S01,,1,44.0,9.0,44,1.00,episode-length,,,episode?\n")

    rows = sheet_to_rows(tmp_sheet)

    assert rows[0][2] == "extra"
    assert rows[1][2] == "episode?"


def test_write_csv(tmp_path):
    """Test writing manifest CSV."""
    dest = tmp_path / "manifest.csv"
    rows = [
        ("Show/Season01/file1.mkv", "show", "S01E01", "aired", ""),
        ("Show/Season01/file2.mkv", "show", "S01E02", "aired", "1"),
        ("Show/Season02/file3.mkv", "show", "extra", "aired", ""),
    ]

    write_csv(rows, dest)

    # Read back and verify
    with dest.open() as f:
        lines = f.readlines()

    assert lines[0].strip() == "path,show,label,order,disc"
    assert "Show/Season01/file1.mkv,show,S01E01,aired," in lines[1]
    assert "Show/Season01/file2.mkv,show,S01E02,aired,1" in lines[2]
    assert "Show/Season02/file3.mkv,show,extra,aired," in lines[3]


def test_build_rows_slug_guessing(tmp_lib):
    """Test slug guessing from directory names."""
    tmp_lib.mkdir()

    # Various directory name patterns
    (tmp_lib / "The_West_Wing").mkdir()
    (tmp_lib / "Babylon5").mkdir()
    (tmp_lib / "TheWire").mkdir()

    (tmp_lib / "The_West_Wing" / "show_s01e01.mkv").touch()
    (tmp_lib / "Babylon5" / "show_s01e01.mkv").touch()
    (tmp_lib / "TheWire" / "show_s01e01.mkv").touch()

    rows = build_rows(tmp_lib)

    assert len(rows) == 3
    # Check that slugs are guessed correctly (lowercase, underscores → dashes)
    slugs = {row[1] for row in rows}
    assert "the-west-wing" in slugs
    assert "babylon5" in slugs
    assert "thewire" in slugs


def test_build_rows_three_digit_episodes(tmp_lib):
    """Test handling of 3-digit episode numbers."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "Sesame_Street"
    show_dir.mkdir()

    (show_dir / "sesame_street_s01e100.mkv").touch()
    (show_dir / "sesame_street_s01e101.mkv").touch()

    rows = build_rows(tmp_lib)

    assert len(rows) == 2
    # Verify 3-digit episodes are preserved in label
    assert rows[0][2] == "S01E100"
    assert rows[1][2] == "S01E101"


def test_build_rows_relative_paths(tmp_lib):
    """Test that paths are relative to root."""
    tmp_lib.mkdir()
    show_dir = tmp_lib / "Show" / "Season01"
    show_dir.mkdir(parents=True)

    (show_dir / "show_s01e01.mkv").touch()

    rows = build_rows(tmp_lib)

    # Path should be relative and use forward slashes
    assert rows[0][0] == "Show/Season01/show_s01e01.mkv"


def test_bundle_slugs_and_sample_merge(tmp_path):
    from showvenist.eval.mkmanifest import bundle_slugs, merge_sample_labels

    lib, bundles = tmp_path / "lib", tmp_path / "bundles"
    for d in ("The_West_Wing", "Babylon5", "Mystery"):
        (lib / d).mkdir(parents=True)
    for s in ("the-west-wing-1999", "babylon-5-1994"):
        (bundles / s).mkdir(parents=True)
        (bundles / s / "show.json").write_text("{}")
    assert bundle_slugs(bundles, lib) == {
        "The_West_Wing": "the-west-wing-1999", "Babylon5": "babylon-5-1994"}

    cols = "path,show,label,same_as\n"
    sample = tmp_path / "sample.csv"
    sample.write_text(cols + "A/a_t00.mkv,A,extra,\n")
    full = tmp_path / "full.csv"
    full.write_text(cols + "A/a_t00.mkv,A,,\nA/b_t00.mkv,A,,A/a_t00.mkv\n")
    out = tmp_path / "out.csv"
    assert merge_sample_labels(sample, full, out) == 1
    from showvenist.eval.mkmanifest import sheet_to_rows

    assert [r[2] for r in sheet_to_rows(out)] == ["extra", "extra"]  # the duplicate inherits


def test_sheet_labels_are_normalized_including_two_part_and_extra_multi(tmp_path):
    from showvenist.eval.mkmanifest import sheet_to_rows

    sheet = tmp_path / "s.csv"
    sheet.write_text(
        "path,show,label,same_as\n"
        "A/a_t00.mkv,A,s01e01+s01e02,\n"
        "A/a_t01.mkv,A,S02E05+e06,\n"
        "A/a_t02.mkv,A,Extra-Multi,\n"
        "A/a_t03.mkv,A,s03e04,\n"
        "A/a_t04.mkv,A,EXTRA,\n"
    )
    assert [r[2] for r in sheet_to_rows(sheet)] == [
        "S01E01+S01E02", "S02E05+E06", "extra-multi", "S03E04", "extra"]


def test_a_show_prefix_in_an_episode_label_is_tolerated(tmp_path):
    from showvenist.eval.mkmanifest import sheet_to_rows

    sheet = tmp_path / "s.csv"
    sheet.write_text(
        "path,show,label,same_as\n"
        "A/a_t00.mkv,A,the_wire_s03e05,\n"
        "A/a_t01.mkv,A,Firefly_S01E02+e03,\n"
        "A/a_t02.mkv,A,show_s00e00+e01,\n"
    )
    assert [r[2] for r in sheet_to_rows(sheet)] == [
        "S03E05", "S01E02+E03", "S00E00+E01"]
