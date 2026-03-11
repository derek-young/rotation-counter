import json
from pathlib import Path


def load_reference_orientations(path: str) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)["frame_orientations"]


def compute_classification_accuracy(
    orientations: dict[int, str],
    reference: dict[str, str],
) -> float:
    """Return fraction of frames matching the reference (0.0-1.0).

    Only frames present in both dicts are compared.
    """
    str_orientations = {str(k): v for k, v in orientations.items()}
    common = set(str_orientations) & set(reference)
    if not common:
        return 0.0
    matches = sum(1 for k in common if str_orientations[k] == reference[k])
    return round(matches / len(common), 4)


def compute_moving_average_accuracy(
    log_dir: str,
    current: float,
    window: int = 10,
) -> float:
    """Average classification_accuracy across the last `window` runs including current.

    Reads run_*.json log files sorted chronologically by name.
    """
    past_logs = sorted(Path(log_dir).glob("run_*.json"))
    values: list[float] = []
    for p in past_logs[-(window - 1):]:
        try:
            data = json.loads(p.read_text())
            if "classification_accuracy" in data:
                values.append(data["classification_accuracy"])
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    values.append(current)
    return round(sum(values) / len(values), 4)
