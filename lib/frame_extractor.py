"""
Frame extractor: samples video at TARGET_FPS and resizes frames.
Uses only OpenCV for video I/O — no tracking or CV models.
"""

from __future__ import annotations
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from config import TARGET_FPS, FRAME_WIDTH


@dataclass
class VideoFrame:
    index: int          # Sequential frame index (0-based)
    timestamp: float    # Time in seconds from video start
    image: Image.Image  # PIL Image


def extract_frames(video_path: str, fps: float = TARGET_FPS) -> list[VideoFrame]:
    """
    Extract frames from video at the specified FPS, resize to FRAME_WIDTH.

    Args:
        video_path: Path to the input video file.
        fps: Target sampling rate in frames per second.

    Returns:
        Ordered list of VideoFrame objects.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / source_fps if source_fps > 0 else 0

    # Clamp requested FPS to source FPS
    effective_fps = min(fps, source_fps)
    frame_interval = int(round(source_fps / effective_fps))

    print(
        f"[frame_extractor] source={source_fps:.1f}fps  duration={duration:.1f}s  "
        f"sampling every {frame_interval} frames (~{effective_fps:.1f}fps)"
    )

    frames: list[VideoFrame] = []
    frame_number = 0
    sample_index = 0

    while True:
        is_successful, frame_bgr = capture.read()
        if not is_successful:
            break

        if frame_number % frame_interval == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = _resize_width(Image.fromarray(rgb), FRAME_WIDTH)
            timestamp = frame_number / source_fps

            frames.append(
                VideoFrame(
                    index=sample_index,
                    timestamp=timestamp,
                    image=pil_img,
                )
            )
            sample_index += 1

        frame_number += 1

    capture.release()
    print(f"[frame_extractor] extracted {len(frames)} frames")
    return frames


def _resize_width(img: Image.Image, target_width: int) -> Image.Image:
    """Resize image to target_width, maintaining aspect ratio."""
    w, h = img.size
    if w == target_width:
        return img
    scale = target_width / w
    new_h = int(h * scale)
    return img.resize((target_width, new_h), Image.LANCZOS)
