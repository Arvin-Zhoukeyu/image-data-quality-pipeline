import numpy as np
import pytest

from src.cleanlab_detector import (
    calculate_detection_metrics,
    detect_label_issues,
    validate_cleanlab_inputs,
)


def test_validate_cleanlab_inputs_accepts_valid_data():
    labels = np.array(
        [0, 1, 0],
        dtype=np.int64,
    )

    pred_probs = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.7, 0.3],
        ],
        dtype=np.float64,
    )

    validate_cleanlab_inputs(
        labels=labels,
        pred_probs=pred_probs,
    )


def test_validate_cleanlab_inputs_rejects_wrong_shape():
    labels = np.array(
        [0, 1],
        dtype=np.int64,
    )

    pred_probs = np.array(
        [0.8, 0.2],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        validate_cleanlab_inputs(
            labels=labels,
            pred_probs=pred_probs,
        )


def test_validate_cleanlab_inputs_rejects_invalid_sums():
    labels = np.array(
        [0, 1],
        dtype=np.int64,
    )

    pred_probs = np.array(
        [
            [0.8, 0.8],
            [0.1, 0.2],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        validate_cleanlab_inputs(
            labels=labels,
            pred_probs=pred_probs,
        )


def test_detection_metrics_are_correct():
    true_noise_mask = np.array(
        [
            True,
            True,
            False,
            False,
        ]
    )

    detected_issue_mask = np.array(
        [
            True,
            False,
            True,
            False,
        ]
    )

    metrics = calculate_detection_metrics(
        true_noise_mask=true_noise_mask,
        detected_issue_mask=detected_issue_mask,
    )

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1

    assert metrics["precision"] == pytest.approx(
        0.5
    )

    assert metrics["recall"] == pytest.approx(
        0.5
    )

    assert metrics["f1"] == pytest.approx(
        0.5
    )


def test_detect_label_issues_returns_expected_shapes():
    labels = np.array(
        [
            0,
            1,
            0,
            1,
            0,
            1,
        ],
        dtype=np.int64,
    )

    pred_probs = np.array(
        [
            [0.90, 0.10],
            [0.10, 0.90],
            [0.85, 0.15],
            [0.80, 0.20],
            [0.75, 0.25],
            [0.20, 0.80],
        ],
        dtype=np.float64,
    )

    true_noise_mask = np.array(
        [
            False,
            False,
            False,
            True,
            False,
            False,
        ]
    )

    result = detect_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        true_noise_mask=true_noise_mask,
    )

    assert result.issue_mask.shape == (
        len(labels),
    )

    assert result.quality_scores.shape == (
        len(labels),
    )

    assert result.predicted_labels.shape == (
        len(labels),
    )

    assert len(result.results_table) == len(
        labels
    )

    assert "label_quality_score" in (
        result.results_table.columns
    )