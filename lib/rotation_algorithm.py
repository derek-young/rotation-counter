"""
Rotation counting algorithm.

Converts a sequence of discrete orientation labels (FRONT/RIGHT_SIDE/BACK/LEFT_SIDE)
into a count of complete 360° rotations using:
  1. Smoothing (majority-vote sliding window)
  2. Direction detection (CW vs CCW)
  3. Angle unwrapping (monotonic cumulative angle)
  4. Cycle counting (integer multiples of 360°)

This stage is fully deterministic — all LLM non-determinism is absorbed here.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field

from config import SMOOTHING_WINDOW, MIN_DIRECTION_CONSISTENCY


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
    direction: str          # "CW", "CCW", or "UNKNOWN"
    confidence: float       # 0.0–1.0
    smoothed_sequence: list[str] = field(default_factory=list)
    cumulative_angle: float = 0.0
    warnings: list[str] = field(default_factory=list)


def count_rotations(
    frame_orientations: dict[int, str],
) -> RotationResult:
    """
    Count complete 360° rotations from a frame_index → orientation mapping.

    Args:
        frame_orientations: {frame_idx: orientation_str} — may contain UNKNOWN.

    Returns:
        RotationResult with count and diagnostic info.
    """
    # Build ordered sequence, skipping UNKNOWN gaps
    ordered = [
        frame_orientations[k]
        for k in sorted(frame_orientations.keys())
    ]

    warnings: list[str] = []

    # Replace UNKNOWN by interpolating from neighbors
    filled = _fill_unknown(ordered)

    # Smooth to reduce VLM misclassifications
    smoothed = _majority_vote_smooth(filled, window=SMOOTHING_WINDOW)

    # Convert to angles
    angles = [ORIENT_TO_ANGLE.get(o, -1) for o in smoothed]
    angles = [a for a in angles if a >= 0]  # drop any remaining UNKNOWN

    if len(angles) < 4:
        warnings.append("Too few valid frames to detect rotations")
        return RotationResult(
            count=0, direction="UNKNOWN", confidence=0.0,
            smoothed_sequence=smoothed, warnings=warnings
        )

    # Detect direction
    direction, dir_confidence = _detect_direction(angles)
    if dir_confidence < MIN_DIRECTION_CONSISTENCY:
        warnings.append(
            f"Low direction consistency ({dir_confidence:.2f}). "
            "Rotation may not be smooth."
        )

    # Unwrap to monotonic cumulative angle
    cumulative = _unwrap_angles(angles, direction)

    # Count complete 360° cycles
    total_angle = abs(cumulative[-1] - cumulative[0])
    count = int(total_angle / 360)

    # Confidence: based on direction consistency + unique orientations seen
    unique = set(smoothed) - {"UNKNOWN"}
    orient_coverage = len(unique) / 4.0  # how many of 4 cardinal directions we saw
    confidence = (dir_confidence + orient_coverage) / 2.0

    if len(unique) < 3:
        warnings.append(
            f"Only {len(unique)} unique orientations seen — possible undercounting"
        )

    return RotationResult(
        count=count,
        direction=direction,
        confidence=round(confidence, 3),
        smoothed_sequence=smoothed,
        cumulative_angle=round(total_angle, 1),
        warnings=warnings,
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


def _majority_vote_smooth(sequence: list[str], window: int) -> list[str]:
    """
    Smooth orientation sequence with a sliding majority-vote window.
    Absorbs single-frame VLM misclassifications.
    """
    n = len(sequence)
    smoothed = []
    half = window // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_vals = [sequence[j] for j in range(start, end) if sequence[j] != "UNKNOWN"]
        if not window_vals:
            smoothed.append("UNKNOWN")
        else:
            counts = Counter(window_vals)
            winner, max_count = counts.most_common(1)[0]
            # On a tie, prefer the current frame's value to avoid smoothing away valid transitions
            if sequence[i] != "UNKNOWN" and counts.get(sequence[i], 0) == max_count:
                winner = sequence[i]
            smoothed.append(winner)

    return smoothed


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
