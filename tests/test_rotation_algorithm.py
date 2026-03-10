"""
Unit tests for the rotation counting algorithm.
Run with: python3 -m pytest tests/ -v
"""

import glob
import json
import os
import pytest
import sys

from lib.rotation_algorithm import (
    count_angular_rotations as count_rotations,
    count_front_back_rotations,
    _detect_direction,
    _fill_unknown,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
_LOG_FILES = sorted(glob.glob(os.path.join(_LOGS_DIR, "*.json")))

def make_seq(labels: list[str]) -> dict[int, str]:
    """Convert a list of labels into a frame_index → orientation dict."""
    return {i: label for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Smoke tests: perfect sequences
# ---------------------------------------------------------------------------

def test_five_rotations_cw():
    # 5 clean CW rotations: F→R→B→L repeating, then back to F
    seq = (["FRONT", "RIGHT_SIDE", "BACK", "LEFT_SIDE"] * 5) + ["FRONT"]
    result = count_rotations(make_seq(seq))
    assert result.count == 5, f"Expected 5, got {result.count}"
    assert result.direction == "CW"


def test_five_rotations_ccw():
    # 5 clean CCW rotations: F→L→B→R repeating
    seq = (["FRONT", "LEFT_SIDE", "BACK", "RIGHT_SIDE"] * 5) + ["FRONT"]
    result = count_rotations(make_seq(seq))
    assert result.count == 5, f"Expected 5, got {result.count}"
    assert result.direction == "CCW"


def test_two_rotations_cw():
    seq = (["FRONT", "RIGHT_SIDE", "BACK", "LEFT_SIDE"] * 2) + ["FRONT"]
    result = count_rotations(make_seq(seq))
    assert result.count == 2, f"Expected 2, got {result.count}"


def test_zero_rotations():
    # Only front frames — no movement
    seq = ["FRONT"] * 10
    result = count_rotations(make_seq(seq))
    assert result.count == 0, f"Expected 0, got {result.count}"


# ---------------------------------------------------------------------------
# Robustness: noisy sequences (1-2 misclassified frames)
# ---------------------------------------------------------------------------

def test_noisy_five_rotations():
    """With occasional VLM misclassifications, still returns 5."""
    base = ["FRONT", "RIGHT_SIDE", "BACK", "LEFT_SIDE"] * 5 + ["FRONT"]
    # Inject 2 noise frames
    noisy = list(base)
    noisy[3] = "FRONT"   # Misclassified LEFT_SIDE → FRONT
    noisy[11] = "BACK"   # Misclassified LEFT_SIDE → BACK
    result = count_rotations(make_seq(noisy))
    assert result.count == 5, f"Expected 5 with noise, got {result.count}"


def test_unknown_frames_filled():
    """UNKNOWN frames are interpolated and don't break counting."""
    seq = [
        "FRONT", "UNKNOWN", "RIGHT_SIDE", "BACK", "LEFT_SIDE",
        "FRONT", "RIGHT_SIDE", "UNKNOWN", "LEFT_SIDE",
        "FRONT"
    ]
    filled = _fill_unknown(seq)
    assert "UNKNOWN" not in filled or filled.count("UNKNOWN") == 0, \
        "UNKNOWN frames should be filled in by neighbors"


def test_repeated_states_count_correctly():
    """Multiple frames of same orientation (due to higher FPS) still count once."""
    # At 3 FPS, each orientation might appear 2-3 times
    seq = (["FRONT", "FRONT", "RIGHT_SIDE", "RIGHT_SIDE", "BACK", "BACK", "LEFT_SIDE", "LEFT_SIDE"] * 5
           + ["FRONT", "FRONT"])
    result = count_rotations(make_seq(seq))
    assert result.count == 5, f"Expected 5 with repeated states, got {result.count}"


# ---------------------------------------------------------------------------
# Direction detection tests
# ---------------------------------------------------------------------------

def test_detect_direction_cw():
    angles = [0, 90, 180, 270, 0, 90, 180]
    direction, conf = _detect_direction(angles)
    assert direction == "CW", f"Expected CW, got {direction}"
    assert conf > 0.7, f"Expected high confidence, got {conf}"


def test_detect_direction_ccw():
    angles = [0, 270, 180, 90, 0, 270, 180]
    direction, conf = _detect_direction(angles)
    assert direction == "CCW", f"Expected CCW, got {direction}"
    assert conf > 0.7, f"Expected high confidence, got {conf}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_frame_returns_zero():
    result = count_rotations(make_seq(["FRONT"]))
    assert result.count == 0


def test_three_frames_partial_rotation():
    seq = ["FRONT", "RIGHT_SIDE", "BACK"]
    result = count_rotations(make_seq(seq))
    assert result.count == 0, "Partial rotation should not count"


def test_all_unknown_returns_zero():
    seq = ["UNKNOWN"] * 10
    result = count_rotations(make_seq(seq))
    assert result.count == 0

def test_llm_generated_orientation_1():
    expected_count = 5
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "run_20260309_171010.json")
    with open(log_path) as f:
        data = json.load(f)
    sequence = {int(k): v for k, v in data["frame_orientations"].items()}
    result = count_rotations(sequence)
    assert result.count == expected_count, f"Expected {expected_count}, got {result.count}"

def test_llm_generated_orientation_2():
    expected_count = 5
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "run_20260309_180339.json")
    with open(log_path) as f:
        data = json.load(f)
    sequence = {int(k): v for k, v in data["frame_orientations"].items()}
    result = count_rotations(sequence)
    assert result.count == expected_count, f"Expected {expected_count}, got {result.count}"


# ---------------------------------------------------------------------------
# count_front_back_rotations: all log files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("log_path", _LOG_FILES, ids=[os.path.basename(p) for p in _LOG_FILES])
def test_front_back_rotations_all_logs(log_path):
    with open(log_path) as f:
        data = json.load(f)
    sequence = {int(k): v for k, v in data["frame_orientations"].items()}
    result = count_front_back_rotations(sequence)
    assert result.count == 5, (
        f"{os.path.basename(log_path)}: expected 5, got {result.count}"
    )