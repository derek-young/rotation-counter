import json
import os
from datetime import datetime
from pathlib import Path

from config import LOG_DIR

def save_log(
    video_path: str,
    orientations: dict[int, str],
    result,
    final_count: int,
    issues: list[str],
    elapsed: float,
    model: str,
    total_tokens: int,
) -> None:
    """Save a JSON log of this run for audit/debugging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(LOG_DIR) / f"run_{ts}.json"

    log_data = {
        "model": model,
        "total_tokens": total_tokens,
        "timestamp": ts,
        "video": video_path,
        "final_count": final_count,
        "direction": result.direction,
        "cumulative_angle": result.cumulative_angle,
        "confidence": result.confidence,
        "elapsed_seconds": round(elapsed, 2),
        "warnings": issues,
        "frame_orientations": {str(k): v for k, v in sorted(orientations.items())},
        "sequence": result.sequence,
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"[pipeline] log saved → {log_path}")