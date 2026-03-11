"""
Rotation Counter — main pipeline entrypoint.

Usage:
    python main.py [VIDEO_PATH]

Output:
    Prints a single integer: the number of full 360° rotations detected.

The pipeline:
    1. Extract frames from video at TARGET_FPS
    2. Compose contact sheets (numbered frame grids)
    3. Classify each cell's orientation via VLM (async, batched)
    4. Smooth the orientation sequence
    5. Count complete rotations algorithmically
    6. Validate and print result
"""

from __future__ import annotations
import asyncio
import sys
import time
from pathlib import Path

from config import SAVE_LOGS, DUMP_SHEETS, DUMP_SHEETS_DIR
from lib.frame_extractor import extract_frames
from lib.contact_sheet import compose_contact_sheets, dump_labeled_sheets
from lib.vlm_classifier import classify_sheets
from lib.rotation_algorithm import count_front_back_rotations
from lib.validator import validate_result
from lib.save_log import save_log


async def run_pipeline(video_path: str) -> int:
    """
    Execute the full rotation-counting pipeline.

    Returns:
        Integer count of complete 360° rotations.
    """
    t0 = time.perf_counter()

    # Step 1: Frame extraction
    frames = extract_frames(video_path)
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    # Step 2: Contact sheet composition
    sheets = compose_contact_sheets(frames)

    # Step 3: Async VLM classification
    frame_orientations, model_used, total_tokens = await classify_sheets(sheets)

    # Optional: dump sheets with orientation labels to disk for inspection
    if DUMP_SHEETS:
        dump_labeled_sheets(sheets, frame_orientations, DUMP_SHEETS_DIR)

    t_classify = time.perf_counter()
    print(f"[pipeline] classification done in {t_classify - t0:.1f}s")

    # Step 4 + 5: Smooth + count
    result = count_front_back_rotations(frame_orientations)

    # Step 6: Validate
    final_count, issues = validate_result(result)

    t_total = time.perf_counter()
    print(f"[pipeline] total time: {t_total - t0:.1f}s")

    if SAVE_LOGS:
        save_log(
            elapsed=t_total - t0,
            final_count=final_count,
            issues=issues,
            model=model_used,
            orientations=frame_orientations,
            result=result,
            total_tokens=total_tokens,
            video_path=video_path,
        )

    return final_count


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} requires a path to the video file", file=sys.stderr)
        sys.exit(1)

    video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"Error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    count = asyncio.run(run_pipeline(video_path))
    print(count)  # Final output — single integer


if __name__ == "__main__":
    main()
