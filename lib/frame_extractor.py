"""
Frame extractor: samples video at TARGET_FPS and resizes frames.
Uses ffmpeg subprocess for efficient frame extraction — skips decoding
of non-output frames at the codec level.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

from PIL import Image

from config import FRAME_WIDTH, TARGET_FPS


@dataclass
class VideoFrame:
    index: int          # Sequential frame index (0-based)
    timestamp: float    # Time in seconds from video start
    image: Image.Image  # PIL Image


def _probe_video(video_path: str) -> tuple[float, float, int, int]:
    """
    Return (source_fps, duration, width, height) for the first video stream.
    Raises RuntimeError if ffprobe is unavailable or the stream cannot be read.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found — ensure ffmpeg is installed and on PATH")

    data = json.loads(result.stdout)
    stream = data["streams"][0]

    width = int(stream["width"])
    height = int(stream["height"])

    # r_frame_rate is a fraction string like "30000/1001"
    num, den = stream["r_frame_rate"].split("/")
    source_fps = float(num) / float(den)

    # duration may live on stream or format level
    duration = float(stream.get("duration") or data["format"]["duration"])

    return source_fps, duration, width, height


def extract_frames(video_path: str, fps: float = TARGET_FPS) -> list[VideoFrame]:
    """
    Extract frames from video at the specified FPS, resize to FRAME_WIDTH.

    Args:
        video_path: Path to the input video file.
        fps: Target sampling rate in frames per second.

    Returns:
        Ordered list of VideoFrame objects.
    """
    source_fps, duration, orig_width, orig_height = _probe_video(video_path)
    effective_fps = min(fps, source_fps)

    # Compute output dimensions; keep height even (ffmpeg requirement for some codecs)
    out_width = FRAME_WIDTH
    out_height = int(orig_height * FRAME_WIDTH / orig_width)
    if out_height % 2 != 0:
        out_height += 1

    print(
        f"[frame_extractor] source={source_fps:.1f}fps  duration={duration:.1f}s  "
        f"sampling at {effective_fps:.1f}fps"
    )

    frame_size = out_width * out_height * 3  # bytes per RGB24 frame

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={effective_fps},scale={out_width}:{out_height}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-v", "quiet",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found — ensure ffmpeg is installed and on PATH")

    frames: list[VideoFrame] = []
    index = 0

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        img = Image.frombytes("RGB", (out_width, out_height), raw)
        frames.append(VideoFrame(index=index, timestamp=index / effective_fps, image=img))
        index += 1

    proc.wait()
    print(f"[frame_extractor] extracted {len(frames)} frames")
    return frames
