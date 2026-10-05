"""Fixtures for creating MKV files with text subtitles."""

from pathlib import Path

import av
import av.stream
import numpy as np


def make_srt(path: Path, cues: list[tuple[float, float, str]]) -> None:
    """Write an SRT subtitle file.

    Args:
        path: File path to write to.
        cues: List of (start_s, end_s, text) tuples.
    """
    lines = []
    for i, (start, end, text) in enumerate(cues, 1):
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        start_str = (
            f"{start_ms // 3600000:02d}:{(start_ms % 3600000) // 60000:02d}:"
            f"{(start_ms % 60000) // 1000:02d},{start_ms % 1000:03d}"
        )
        end_str = (
            f"{end_ms // 3600000:02d}:{(end_ms % 3600000) // 60000:02d}:"
            f"{(end_ms % 60000) // 1000:02d},{end_ms % 1000:03d}"
        )
        lines.append(f"{i}\n{start_str} --> {end_str}\n{text}\n")
    path.write_text("\n".join(lines))


def make_mkv_with_text_subs(
    path: Path,
    cues: list[tuple[float, float, str]],
    lang: str = "eng",
    title: str | None = None,
    forced: bool = False,
    duration_s: float | None = None,
    extra_tracks: tuple = (),
) -> None:
    """Create a minimal MKV file with a video stream and text subtitle stream.

    PyAV 19.0.1 cannot encode 'subrip' codec directly (avcodec_open2 fails with
    InvalidDataError), so this uses 'ass' (Advanced SubStation Alpha) format.
    Both are text-based subtitle formats; ASS is a superset with styling.

    Args:
        path: File path to write to.
        cues: List of (start_s, end_s, text) tuples for subtitles.
        lang: Language code for subtitle stream (default: 'eng').
        title: Container title metadata (default: None).
        forced: Whether subtitles should be marked as forced (default: False).
        duration_s: Total video duration in seconds. If None, calculated from cues.
        extra_tracks: Additional subtitle streams (not yet implemented).
    """
    # Calculate duration from cues if not provided
    if duration_s is None:
        if cues:
            duration_s = max(end for _, end, _ in cues) + 1
        else:
            duration_s = 1

    # Create output container
    container = av.open(str(path), "w")

    # Add container metadata
    if title:
        container.metadata["title"] = title

    # Create a minimal video stream (64x48, 1 fps)
    video_stream = container.add_stream("mpeg4", rate=1)
    video_stream.width = 64
    video_stream.height = 48
    video_stream.pix_fmt = "yuv420p"

    # Create subtitle stream with ASS codec (PyAV's working alternative to subrip)
    subtitle_stream = container.add_stream("ass")
    subtitle_stream.metadata["language"] = lang

    # Set forced disposition if requested
    if forced:
        subtitle_stream.disposition |= av.stream.Disposition.forced

    # Generate video frames (enough to cover duration)
    frame_count = max(1, int(duration_s))  # 1 frame per second
    for _ in range(frame_count):
        # Create a black frame (RGB format, converted to yuv420p by encoder)
        black_frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(black_frame, format="rgb24")
        # Encode frame
        packets = video_stream.encode(frame)
        for pkt in packets:
            container.mux(pkt)

    # Flush video stream
    packets = video_stream.encode()
    for pkt in packets:
        container.mux(pkt)

    # Create ASS header packet (required for ASS format)
    ass_fmt = (
        "[Script Info]\nTitle: Subtitles\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
        "MarginV, Effect, Text\n"
    )
    ass_header = ass_fmt.encode()
    header_pkt = av.Packet(ass_header)
    header_pkt.stream = subtitle_stream
    header_pkt.pts = 0
    header_pkt.dts = 0
    header_pkt.duration = 0
    header_pkt.is_keyframe = True
    container.mux(header_pkt)

    # Create subtitle event packets from cues
    time_base = float(subtitle_stream.time_base)
    for start, end, text in cues:
        # Convert time to ASS format (h:mm:ss.cc)
        def format_ass_time(seconds: float) -> str:
            total_cs = int(seconds * 100)  # centiseconds
            h = total_cs // 360000
            m = (total_cs % 360000) // 6000
            s = (total_cs % 6000) // 100
            cs = total_cs % 100
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        start_str = format_ass_time(start)
        end_str = format_ass_time(end)

        # Create ASS Dialogue event line
        dialogue_line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n"
        event_text = dialogue_line.encode()

        event_pkt = av.Packet(event_text)
        event_pkt.stream = subtitle_stream
        event_pkt.pts = int(start / time_base)
        event_pkt.dts = event_pkt.pts
        event_pkt.duration = int((end - start) / time_base)
        container.mux(event_pkt)

    container.close()
