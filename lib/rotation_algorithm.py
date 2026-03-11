"""
Rotation counting algorithm.

Converts a sequence of discrete orientation labels (FRONT/RIGHT_SIDE/BACK/LEFT_SIDE)
into a count of complete 360° rotations using a FRONT → BACK → FRONT state machine.

This stage is fully deterministic — all LLM non-determinism is absorbed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RotationResult:
    count: int
    confidence: float
    sequence: list[str] = field(default_factory=list)
    cumulative_angle: float = 0.0


def count_front_back_rotations(
    frame_orientations: dict[int, str],
) -> RotationResult:
    """
    Count complete rotations using FRONT → BACK → FRONT

    Args:
        frame_orientations: {frame_idx: orientation_str} — may contain UNKNOWN.

    Returns:
        RotationResult with count and diagnostic info.
    """
    ordered = [
        frame_orientations[k]
        for k in sorted(frame_orientations.keys())
    ]

    filled = _fill_unknown(ordered)

    count = 0
    state = "IDLE"  # IDLE | SAW_FRONT | SAW_BACK

    for orientation in filled:
        if orientation == "UNKNOWN":
            continue

        if state == "IDLE":
            if orientation == "FRONT":
                state = "SAW_FRONT"
        if state == "SAW_FRONT":
            if orientation == "BACK":
                state = "SAW_BACK"
        elif state == "SAW_BACK":
            if orientation == "FRONT":
                count += 1
                state = "SAW_FRONT"

    unique = set(filled) - {"UNKNOWN"}
    orient_coverage = len(unique) / 4.0

    return RotationResult(
        count=count,
        confidence=round(orient_coverage, 3),
        sequence=filled,
    )


def _fill_unknown(sequence: list[str]) -> list[str]:
    """Replace UNKNOWN entries by copying the nearest known neighbor."""
    result = list(sequence)
    n = len(result)

    for i, val in enumerate(result):
        if val == "UNKNOWN":
            # Look left then right for a known value
            left = next(
                (result[j] for j in range(i - 1, -1, -1) if result[j] != "UNKNOWN"),
                None
            )
            right = next(
                (result[j] for j in range(i + 1, n) if result[j] != "UNKNOWN"),
                None
            )
            result[i] = left or right or "UNKNOWN"

    return result


