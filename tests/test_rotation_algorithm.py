"""
Unit tests for the rotation counting algorithm.
Run with: python3 -m pytest tests/ -v
"""

import glob
import json
import os

import pytest

from lib.rotation_algorithm import (
    _fill_unknown,
    count_front_back_rotations,
)

_LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
_LOG_FILES = sorted(glob.glob(os.path.join(_LOGS_DIR, "*.json")))[-10:]

def make_seq(labels: list[str]) -> dict[int, str]:
    """Convert a list of labels into a frame_index → orientation dict."""
    return {i: label for i, label in enumerate(labels)}


# ---------------------------------------------------------------------------
# Smoke tests: perfect sequences
# ---------------------------------------------------------------------------

def test_five_rotations():
    # 5 clean CW rotations: F→R→B→L repeating, then back to F
    seq = (["FRONT", "RIGHT_SIDE", "BACK", "LEFT_SIDE"] * 5) + ["FRONT"]
    result = count_front_back_rotations(make_seq(seq))
    assert result.count == 5, f"Expected 5, got {result.count}"


def test_zero_rotations():
    # Only front frames — no movement
    seq = ["FRONT"] * 10
    result = count_front_back_rotations(make_seq(seq))
    assert result.count == 0, f"Expected 0, got {result.count}"


# ---------------------------------------------------------------------------
# Robustness: noisy sequences (1-2 misclassified frames)
# ---------------------------------------------------------------------------


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
    result = count_front_back_rotations(make_seq(seq))
    assert result.count == 5, f"Expected 5 with repeated states, got {result.count}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_frame_returns_zero():
    result = count_front_back_rotations(make_seq(["FRONT"]))
    assert result.count == 0


def test_three_frames_partial_rotation():
    seq = ["FRONT", "RIGHT_SIDE", "BACK"]
    result = count_front_back_rotations(make_seq(seq))
    assert result.count == 0, "Partial rotation should not count"


def test_all_unknown_returns_zero():
    seq = ["UNKNOWN"] * 10
    result = count_front_back_rotations(make_seq(seq))
    assert result.count == 0


# ---------------------------------------------------------------------------
# count_front_back_rotations: all log files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("log_path", _LOG_FILES, ids=[os.path.basename(p) for p in _LOG_FILES])
def test_front_back_rotations_all_logs(log_path):
    with open(log_path) as f:
        data = json.load(f)
    sequence = {int(k): v for k, v in data["frame_orientations"].items()}
    result = count_front_back_rotations(sequence)
    assert result.count == data["final_count"], (
        f"{os.path.basename(log_path)}: expected 5, got {result.count}"
    )