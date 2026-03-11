"""
Unit tests for lib/accuracy.py.
Run with: python3 -m pytest tests/ -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.accuracy import (
    compute_classification_accuracy,
    compute_moving_average_accuracy,
    load_reference_orientations,
)

_PERFECT_RUN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
    "perfect_run_4x2_3fps.json",
)

REFERENCE = {
    "0": "FRONT",
    "1": "RIGHT_SIDE",
    "2": "BACK",
    "3": "LEFT_SIDE",
}

# ---------------------------------------------------------------------------
# compute_classification_accuracy
# ---------------------------------------------------------------------------

def test_perfect_match():
    orientations = {0: "FRONT", 1: "RIGHT_SIDE", 2: "BACK", 3: "LEFT_SIDE"}
    assert compute_classification_accuracy(orientations, REFERENCE) == 1.0


def test_no_match():
    orientations = {0: "BACK", 1: "LEFT_SIDE", 2: "FRONT", 3: "RIGHT_SIDE"}
    assert compute_classification_accuracy(orientations, REFERENCE) == 0.0


def test_partial_match():
    orientations = {0: "FRONT", 1: "RIGHT_SIDE", 2: "FRONT", 3: "FRONT"}
    result = compute_classification_accuracy(orientations, REFERENCE)
    assert result == 0.5  # 2 of 4 match


def test_subset_of_reference_frames():
    """Only frames present in both dicts are compared."""
    orientations = {0: "FRONT", 1: "RIGHT_SIDE"}  # frames 2 & 3 absent
    result = compute_classification_accuracy(orientations, REFERENCE)
    assert result == 1.0


def test_extra_frames_not_in_reference():
    """Frames beyond reference keys are ignored."""
    orientations = {0: "FRONT", 1: "RIGHT_SIDE", 2: "BACK", 3: "LEFT_SIDE", 99: "UNKNOWN"}
    assert compute_classification_accuracy(orientations, REFERENCE) == 1.0


def test_empty_orientations():
    assert compute_classification_accuracy({}, REFERENCE) == 0.0


def test_empty_reference():
    orientations = {0: "FRONT", 1: "BACK"}
    assert compute_classification_accuracy(orientations, {}) == 0.0


def test_perfect_run_against_itself():
    reference = load_reference_orientations(_PERFECT_RUN)
    orientations = {int(k): v for k, v in reference.items()}
    assert compute_classification_accuracy(orientations, reference) == 1.0


# ---------------------------------------------------------------------------
# compute_moving_average_accuracy
# ---------------------------------------------------------------------------

def _write_log(directory: Path, name: str, accuracy: float) -> None:
    (directory / name).write_text(json.dumps({"classification_accuracy": accuracy}))


def test_moving_average_no_prior_logs():
    with tempfile.TemporaryDirectory() as tmp:
        result = compute_moving_average_accuracy(tmp, 0.8, window=10)
        assert result == 0.8


def test_moving_average_with_prior_logs():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_log(d, "run_20260101_000001.json", 0.6)
        _write_log(d, "run_20260101_000002.json", 0.8)
        result = compute_moving_average_accuracy(tmp, 1.0, window=10)
        assert result == round((0.6 + 0.8 + 1.0) / 3, 4)


def test_moving_average_respects_window():
    """Only the last (window-1) logs are included, plus current."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for i in range(1, 12):
            _write_log(d, f"run_20260101_{i:06d}.json", 0.0)
        # With window=3, only last 2 prior logs (both 0.0) + current (1.0)
        result = compute_moving_average_accuracy(tmp, 1.0, window=3)
        assert result == round((0.0 + 0.0 + 1.0) / 3, 4)


def test_moving_average_skips_logs_without_accuracy():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "run_20260101_000001.json").write_text(json.dumps({"other_key": 42}))
        _write_log(d, "run_20260101_000002.json", 0.5)
        result = compute_moving_average_accuracy(tmp, 1.0, window=10)
        assert result == round((0.5 + 1.0) / 2, 4)


def test_moving_average_skips_malformed_logs():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "run_20260101_000001.json").write_text("not valid json{{{")
        _write_log(d, "run_20260101_000002.json", 0.4)
        result = compute_moving_average_accuracy(tmp, 0.6, window=10)
        assert result == round((0.4 + 0.6) / 2, 4)


def test_moving_average_perfect_run_excluded():
    """perfect_run_*.json does not match run_*.json glob and is not included."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "perfect_run_4x2_3fps.json").write_text(
            json.dumps({"classification_accuracy": 0.0})
        )
        result = compute_moving_average_accuracy(tmp, 1.0, window=10)
        assert result == 1.0
