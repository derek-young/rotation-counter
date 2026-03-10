"""
Validator: sanity-checks the rotation result and triggers re-runs if needed.

Applies heuristic checks that catch common failure modes before returning
the final integer count to the user.
"""

from __future__ import annotations

from config import MIN_UNIQUE_ORIENTATIONS, MIN_DIRECTION_CONSISTENCY
from lib.rotation_algorithm import RotationResult


def validate_result(
    result: RotationResult,
    expected_min: int = 1,
    expected_max: int = 20,
) -> tuple[int, list[str]]:
    """
    Validate a RotationResult and return (final_count, issues).

    Issues are descriptive strings explaining any concerns. An empty list
    means the result passed all checks.

    Args:
        result: Output from count_rotations().
        expected_min: Minimum plausible rotation count.
        expected_max: Maximum plausible rotation count.

    Returns:
        (validated_count, list_of_issues)
    """
    issues: list[str] = list(result.warnings)

    # Check 1: Count is within plausible range
    if result.count < expected_min:
        issues.append(
            f"Count {result.count} is below minimum expected {expected_min}. "
            "Possible undercounting due to insufficient frame sampling."
        )
    if result.count > expected_max:
        issues.append(
            f"Count {result.count} exceeds maximum expected {expected_max}. "
            "Possible hallucination or direction detection error."
        )

    # Check 2: Enough unique orientations were observed
    unique = set(result.sequence) - {"UNKNOWN"}
    if len(unique) < MIN_UNIQUE_ORIENTATIONS:
        issues.append(
            f"Only {len(unique)} unique orientations detected (need ≥{MIN_UNIQUE_ORIENTATIONS}). "
            "VLM may be stuck classifying the same orientation."
        )

    # Check 3: Direction was resolved
    if result.direction == "UNKNOWN":
        issues.append(
            "Could not determine rotation direction. "
            "All frames may have the same orientation."
        )

    # Check 4: Cumulative angle is consistent with count
    # Allow ±0.5 rotation tolerance
    expected_min_angle = (result.count - 0.5) * 360
    expected_max_angle = (result.count + 0.5) * 360
    if not (expected_min_angle <= result.cumulative_angle <= expected_max_angle):
        issues.append(
            f"Cumulative angle {result.cumulative_angle}° inconsistent with "
            f"count={result.count} (expected {expected_min_angle:.0f}–{expected_max_angle:.0f}°)."
        )

    # Check 5: Low confidence warning
    if result.confidence < 0.5:
        issues.append(
            f"Overall confidence is low ({result.confidence:.2f}). "
            "Results may be unreliable."
        )

    return result.count, issues


def print_validation_report(result: RotationResult, issues: list[str]) -> None:
    """Print a human-readable validation summary (goes to stderr / logs)."""
    print("=" * 50)
    print(f"  Rotation count    : {result.count}")
    print(f"  Direction         : {result.direction}")
    print(f"  Cumulative angle  : {result.cumulative_angle}°")
    print(f"  Confidence        : {result.confidence:.2f}")
    print(f"  Frames analyzed   : {len(result.sequence)}")
    if issues:
        print(f"  Warnings ({len(issues)}):")
        for w in issues:
            print(f"    ⚠  {w}")
    else:
        print("  Validation        : PASSED (no issues)")
    print("=" * 50)
