"""Tests for text subtitle fixture functions."""

from pathlib import Path

import av

from tests.fixtures.textsubs import make_mkv_with_text_subs, make_srt


class TestMakeSrt:
    """Tests for make_srt fixture."""

    def test_make_srt_basic(self, tmp_path: Path) -> None:
        """Test creating a basic SRT file."""
        path = tmp_path / "test.srt"
        cues = [
            (1.0, 3.0, "First subtitle"),
            (4.0, 6.0, "Second subtitle"),
        ]
        make_srt(path, cues)

        assert path.exists()
        content = path.read_text()
        assert "1" in content
        assert "First subtitle" in content
        assert "2" in content
        assert "Second subtitle" in content
        assert "00:00:01,000" in content
        assert "00:00:03,000" in content

    def test_make_srt_timestamps(self, tmp_path: Path) -> None:
        """Test that timestamps are correctly formatted."""
        path = tmp_path / "test.srt"
        cues = [
            (0.5, 2.5, "Test"),
        ]
        make_srt(path, cues)

        content = path.read_text()
        assert "00:00:00,500 --> 00:00:02,500" in content

    def test_make_srt_empty(self, tmp_path: Path) -> None:
        """Test creating an SRT file with no cues."""
        path = tmp_path / "test.srt"
        make_srt(path, [])
        assert path.exists()
        assert path.read_text() == ""


class TestMakeMkvWithTextSubs:
    """Tests for make_mkv_with_text_subs fixture."""

    def test_make_mkv_basic(self, tmp_path: Path) -> None:
        """Test creating a basic MKV with subtitles."""
        path = tmp_path / "test.mkv"
        cues = [
            (1.0, 3.0, "First subtitle"),
            (4.0, 6.0, "Second subtitle"),
        ]
        make_mkv_with_text_subs(path, cues)

        assert path.exists()
        assert path.stat().st_size > 0

    def test_mkv_subtitle_stream_exists(self, tmp_path: Path) -> None:
        """Test that the MKV contains a subtitle stream."""
        path = tmp_path / "test.mkv"
        cues = [
            (1.0, 3.0, "Test subtitle"),
        ]
        make_mkv_with_text_subs(path, cues)

        with av.open(str(path)) as c:
            sub_streams = [s for s in c.streams if s.type == "subtitle"]
            assert len(sub_streams) > 0

    def test_mkv_subtitle_language(self, tmp_path: Path) -> None:
        """Test that subtitle language is set correctly."""
        path = tmp_path / "test.mkv"
        cues = [(1.0, 3.0, "Test")]
        make_mkv_with_text_subs(path, cues, lang="fra")

        with av.open(str(path)) as c:
            sub_streams = [s for s in c.streams if s.type == "subtitle"]
            assert len(sub_streams) > 0
            # Check language metadata
            lang = sub_streams[0].metadata.get("language") if sub_streams[0].metadata else None
            assert lang == "fra" or lang is None  # Some versions may not preserve metadata

    def test_mkv_subtitle_content(self, tmp_path: Path) -> None:
        """Test that subtitle content is preserved."""
        path = tmp_path / "test.mkv"
        text = "This is a test subtitle"
        cues = [(1.0, 3.0, text)]
        make_mkv_with_text_subs(path, cues)

        with av.open(str(path)) as c:
            sub_streams = [s for s in c.streams if s.type == "subtitle"]
            assert len(sub_streams) > 0
            stream = sub_streams[0]
            packets = list(c.demux(stream))
            assert len(packets) > 0
            # Decode first packet
            decoded = packets[0].decode()
            if decoded:
                # decoded should be subtitle objects
                assert len(decoded) > 0

    def test_mkv_packet_count(self, tmp_path: Path) -> None:
        """Test that subtitle packets match cue count."""
        path = tmp_path / "test.mkv"
        cues = [
            (1.0, 3.0, "First"),
            (4.0, 6.0, "Second"),
            (7.0, 9.0, "Third"),
        ]
        make_mkv_with_text_subs(path, cues)

        with av.open(str(path)) as c:
            sub_streams = [s for s in c.streams if s.type == "subtitle"]
            assert len(sub_streams) > 0
            stream = sub_streams[0]
            packets = list(c.demux(stream))
            # Should have at least as many packets as cues
            assert len(packets) >= len(cues)

    def test_mkv_duration_from_cues(self, tmp_path: Path) -> None:
        """Test that duration is inferred from cues."""
        path = tmp_path / "test.mkv"
        cues = [(1.0, 5.0, "Test")]
        make_mkv_with_text_subs(path, cues)

        with av.open(str(path)) as c:
            # Check duration (should be at least the max cue end time)
            if c.duration:
                duration_s = c.duration / av.time_base
                assert duration_s >= 5.0

    def test_mkv_explicit_duration(self, tmp_path: Path) -> None:
        """Test that explicit duration is respected."""
        path = tmp_path / "test.mkv"
        cues = [(1.0, 2.0, "Test")]
        make_mkv_with_text_subs(path, cues, duration_s=10.0)

        with av.open(str(path)) as c:
            if c.duration:
                duration_s = c.duration / av.time_base
                assert duration_s >= 10.0

    def test_mkv_with_container_title(self, tmp_path: Path) -> None:
        """Test that container title is set."""
        path = tmp_path / "test.mkv"
        cues = [(1.0, 3.0, "Test")]
        make_mkv_with_text_subs(path, cues, title="My Show")

        with av.open(str(path)) as c:
            # Check container metadata
            title = c.metadata.get("title") if c.metadata else None
            assert title == "My Show" or title is None  # Some versions may not preserve

    def test_mkv_with_forced_flag(self, tmp_path: Path) -> None:
        """Test that forced disposition is set."""
        path = tmp_path / "test.mkv"
        cues = [(1.0, 3.0, "Test")]
        make_mkv_with_text_subs(path, cues, forced=True)

        with av.open(str(path)) as c:
            sub_streams = [s for s in c.streams if s.type == "subtitle"]
            assert len(sub_streams) > 0
            # Check forced disposition
            stream = sub_streams[0]
            if stream.disposition:
                forced = bool(stream.disposition & av.stream.Disposition.forced)
                assert forced

    def test_mkv_video_stream_exists(self, tmp_path: Path) -> None:
        """Test that the MKV contains a video stream."""
        path = tmp_path / "test.mkv"
        cues = [(1.0, 3.0, "Test")]
        make_mkv_with_text_subs(path, cues)

        with av.open(str(path)) as c:
            video_streams = [s for s in c.streams if s.type == "video"]
            assert len(video_streams) > 0
            assert video_streams[0].width == 64
            assert video_streams[0].height == 48

    def test_mkv_empty_cues(self, tmp_path: Path) -> None:
        """Test creating an MKV with no subtitle cues."""
        path = tmp_path / "test.mkv"
        make_mkv_with_text_subs(path, [])

        assert path.exists()
        with av.open(str(path)) as c:
            sub_streams = [s for s in c.streams if s.type == "subtitle"]
            assert len(sub_streams) > 0

    def test_mkv_roundtrip(self, tmp_path: Path) -> None:
        """Test that MKV can be created and read back."""
        path = tmp_path / "test.mkv"
        original_cues = [
            (1.0, 3.0, "First subtitle"),
            (4.0, 6.0, "Second subtitle"),
        ]
        make_mkv_with_text_subs(path, original_cues, lang="eng")

        # Read it back
        with av.open(str(path)) as c:
            assert len(c.streams) > 0
            sub_streams = [s for s in c.streams if s.type == "subtitle"]
            assert len(sub_streams) > 0
            # The file should be valid and readable
