"""
Rotation counting algorithm.

Converts a sequence of discrete orientation labels (FRONT/RIGHT_SIDE/BACK/LEFT_SIDE)
into a count of complete 360° rotations using:
  1. Direction detection (CW vs CCW)
  2. Angle unwrapping (monotonic cumulative angle)
  3. Cycle counting (integer multiples of 360°)

This stage is fully deterministic — all LLM non-determinism is absorbed here.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from config import MIN_DIRECTION_CONSISTENCY


ORIENT_TO_ANGLE: dict[str, int] = {
    "FRONT": 0,
    "RIGHT_SIDE": 90,
    "BACK": 180,
    "LEFT_SIDE": 270,
}

ANGLE_TO_ORIENT: dict[int, str] = {v: k for k, v in ORIENT_TO_ANGLE.items()}


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


def _detect_direction(angles: list[int]) -> tuple[str, float]:
    """
    Determine rotation direction (CW=increasing angle, CCW=decreasing).

    Returns (direction, consistency_ratio).
    """
    cw_votes = 0
    ccw_votes = 0

    for i in range(1, len(angles)):
        delta = (angles[i] - angles[i - 1]) % 360
        if delta == 0:
            continue
        # delta in (0, 180) → CW step; delta in (180, 360) → CCW step
        if delta <= 180:
            cw_votes += 1
        else:
            ccw_votes += 1

    total = cw_votes + ccw_votes
    if total == 0:
        return "UNKNOWN", 0.0

    if cw_votes >= ccw_votes:
        return "CW", cw_votes / total
    else:
        return "CCW", ccw_votes / total


def _unwrap_angles(angles: list[int], direction: str) -> list[float]:
    """
    Convert a sequence of 0/90/180/270 angles into a monotonic cumulative
    angle sequence, handling wraparound.

    CW:  increasing  (0 → 90 → 180 → 270 → 360 → 450 ...)
    CCW: decreasing  (0 → -90 → -180 → -270 → -360 ...)
    """
    cumulative = [float(angles[0])]

    for i in range(1, len(angles)):
        prev = angles[i - 1]
        curr = angles[i]

        if direction == "CW":
            # Step forward (mod 360); handle 270→0 wraparound
            delta = (curr - prev) % 360
            if delta > 180:
                # This is actually a small CCW step — treat as small CW residual
                delta -= 360
        else:  # CCW
            # Step backward; handle 0→270 wraparound
            delta = (curr - prev) % 360
            if delta > 180:
                delta -= 360  # convert to negative (e.g. 270 → -90)

        cumulative.append(cumulative[-1] + delta)

    return cumulative
