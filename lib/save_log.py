import json
import os
from datetime import datetime
from pathlib import Path

from config import ACCURACY_MOVING_AVG_WINDOW, LOG_DIR, PERFECT_RUN_PATH
from lib.accuracy import (
    compute_classification_accuracy,
    compute_moving_average_accuracy,
    load_reference_orientations,
)

def save_log(
    video_path: str,
    orientations: dict[int, str],
    result,
    final_count: int,
    elapsed: float,
    model: str,
    total_tokens: int,
) -> None:
    """Save a JSON log of this run for audit/debugging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(LOG_DIR) / f"run_{ts}.json"

    reference = load_reference_orientations(PERFECT_RUN_PATH)
    accuracy = compute_classification_accuracy(orientations, reference)
    accuracy_moving_avg = compute_moving_average_accuracy(
        LOG_DIR, accuracy, window=ACCURACY_MOVING_AVG_WINDOW
    )

    log_data = {
        "model": model,
        "total_tokens": total_tokens,
        "timestamp": ts,
        "video": video_path,
        "final_count": final_count,
        "cumulative_angle": result.cumulative_angle,
        "confidence": result.confidence,
        "elapsed_seconds": round(elapsed, 2),
        "classification_accuracy": accuracy,
        "accuracy_moving_avg": accuracy_moving_avg,
        "frame_orientations": {str(k): v for k, v in sorted(orientations.items())},
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"[pipeline] log saved → {log_path}")

    print("=" * 50)
    print(f"  Rotation count          : {final_count}")
    print(f"  Classification accuracy : {accuracy}")
    print(f"  Frames analyzed         : {len(orientations)}")
    print(f"  Model                   : {model}")
    print(f"  Token usage             : {total_tokens}")
    print("=" * 50)