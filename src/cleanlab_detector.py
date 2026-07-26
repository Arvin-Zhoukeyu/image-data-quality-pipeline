from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class LabelIssueDetectionResult:
    """
    Cleanlab 标签问题检测结果。
    """

    issue_indices: np.ndarray
    issue_mask: np.ndarray
    quality_scores: np.ndarray
    predicted_labels: np.ndarray
    predicted_confidence: np.ndarray
    self_confidence: np.ndarray
    results_table: pd.DataFrame
    metrics: dict[str, float | int]


def validate_cleanlab_inputs(
    labels: np.ndarray,
    pred_probs: np.ndarray,
) -> None:
    """
    检查传入 Cleanlab 的标签和概率是否合法。
    """
    labels = np.asarray(labels)
    pred_probs = np.asarray(pred_probs)

    if labels.ndim != 1:
        raise ValueError(
            "labels must be a one-dimensional array."
        )

    if pred_probs.ndim != 2:
        raise ValueError(
            "pred_probs must be a two-dimensional array."
        )

    if len(labels) != len(pred_probs):
        raise ValueError(
            "labels and pred_probs must contain "
            "the same number of samples."
        )

    if len(labels) == 0:
        raise ValueError(
            "labels and pred_probs cannot be empty."
        )

    if not np.all(np.isfinite(pred_probs)):
        raise ValueError(
            "pred_probs contains NaN or infinite values."
        )

    if np.any(pred_probs < 0) or np.any(pred_probs > 1):
        raise ValueError(
            "All predicted probabilities must be "
            "between zero and one."
        )

    if not np.allclose(
        pred_probs.sum(axis=1),
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Each row of pred_probs must sum to one."
        )

    number_of_classes = pred_probs.shape[1]

    if np.any(labels < 0) or np.any(
        labels >= number_of_classes
    ):
        raise ValueError(
            "labels contains an invalid class index."
        )


def load_noise_mask(
    noise_root: str | Path,
    noise_level: str,
    number_of_samples: int,
) -> Optional[np.ndarray]:
    """
    加载 Week 1 生成的真实噪声位置。

    clean 实验没有人工噪声，因此返回全 False。
    如果噪声实验找不到 noise_mask.npy，则返回 None。
    """
    if noise_level == "clean":
        return np.zeros(
            number_of_samples,
            dtype=bool,
        )

    noise_mask_path = (
        Path(noise_root)
        / noise_level
        / "noise_mask.npy"
    )

    if not noise_mask_path.exists():
        return None

    noise_mask = np.load(
        noise_mask_path
    ).astype(bool)

    if noise_mask.ndim != 1:
        raise ValueError(
            "noise_mask must be one-dimensional."
        )

    if len(noise_mask) != number_of_samples:
        raise ValueError(
            "noise_mask and labels must contain "
            "the same number of samples."
        )

    return noise_mask


def calculate_detection_metrics(
    true_noise_mask: Optional[np.ndarray],
    detected_issue_mask: np.ndarray,
) -> dict[str, float | int]:
    """
    将 Cleanlab 检测结果与人工注入的 noise_mask 比较。

    Positive:
        标签存在人工噪声。

    Predicted positive:
        Cleanlab 判断为 label issue。
    """
    detected_issue_mask = np.asarray(
        detected_issue_mask,
        dtype=bool,
    )

    metrics: dict[str, float | int] = {
        "detected_issue_count": int(
            detected_issue_mask.sum()
        ),
        "detected_issue_rate": float(
            detected_issue_mask.mean()
        ),
    }

    if true_noise_mask is None:
        return metrics

    true_noise_mask = np.asarray(
        true_noise_mask,
        dtype=bool,
    )

    if len(true_noise_mask) != len(
        detected_issue_mask
    ):
        raise ValueError(
            "true_noise_mask and detected_issue_mask "
            "must have the same length."
        )

    true_positive = int(
        np.logical_and(
            true_noise_mask,
            detected_issue_mask,
        ).sum()
    )

    false_positive = int(
        np.logical_and(
            ~true_noise_mask,
            detected_issue_mask,
        ).sum()
    )

    false_negative = int(
        np.logical_and(
            true_noise_mask,
            ~detected_issue_mask,
        ).sum()
    )

    true_negative = int(
        np.logical_and(
            ~true_noise_mask,
            ~detected_issue_mask,
        ).sum()
    )

    metrics.update(
        {
            "true_noise_count": int(
                true_noise_mask.sum()
            ),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": float(
                precision_score(
                    true_noise_mask,
                    detected_issue_mask,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    true_noise_mask,
                    detected_issue_mask,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    true_noise_mask,
                    detected_issue_mask,
                    zero_division=0,
                )
            ),
            "detection_accuracy": float(
                accuracy_score(
                    true_noise_mask,
                    detected_issue_mask,
                )
            ),
        }
    )

    return metrics


def detect_label_issues(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    true_noise_mask: Optional[np.ndarray] = None,
    filter_by: str = "prune_by_noise_rate",
    rank_by: str = "self_confidence",
) -> LabelIssueDetectionResult:
    """
    使用 Cleanlab 找出可能存在错误标签的样本。

    quality score 越低，标签越可疑。
    """
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    pred_probs = np.asarray(
        pred_probs,
        dtype=np.float64,
    )

    validate_cleanlab_inputs(
        labels=labels,
        pred_probs=pred_probs,
    )

    issue_indices = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        filter_by=filter_by,
        return_indices_ranked_by=rank_by,
    )

    issue_indices = np.asarray(
        issue_indices,
        dtype=np.int64,
    )

    quality_scores = get_label_quality_scores(
        labels=labels,
        pred_probs=pred_probs,
        method="self_confidence",
    )

    predicted_labels = pred_probs.argmax(
        axis=1
    ).astype(np.int64)

    predicted_confidence = pred_probs.max(
        axis=1
    )

    row_indices = np.arange(
        len(labels)
    )

    self_confidence = pred_probs[
        row_indices,
        labels,
    ]

    issue_mask = np.zeros(
        len(labels),
        dtype=bool,
    )

    issue_mask[issue_indices] = True

    issue_rank = np.full(
        len(labels),
        fill_value=-1,
        dtype=np.int64,
    )

    issue_rank[issue_indices] = np.arange(
        1,
        len(issue_indices) + 1,
    )

    results_table = pd.DataFrame(
        {
            "sample_index": row_indices,
            "given_label": labels,
            "predicted_label": predicted_labels,
            "predicted_confidence": predicted_confidence,
            "self_confidence": self_confidence,
            "label_quality_score": quality_scores,
            "is_label_issue": issue_mask,
            "issue_rank": issue_rank,
        }
    )

    if true_noise_mask is not None:
        true_noise_mask = np.asarray(
            true_noise_mask,
            dtype=bool,
        )

        if len(true_noise_mask) != len(labels):
            raise ValueError(
                "true_noise_mask and labels must "
                "have the same length."
            )

        results_table["is_true_noise"] = (
            true_noise_mask
        )

        results_table["correctly_detected_noise"] = (
            true_noise_mask & issue_mask
        )

    metrics = calculate_detection_metrics(
        true_noise_mask=true_noise_mask,
        detected_issue_mask=issue_mask,
    )

    return LabelIssueDetectionResult(
        issue_indices=issue_indices,
        issue_mask=issue_mask,
        quality_scores=quality_scores,
        predicted_labels=predicted_labels,
        predicted_confidence=predicted_confidence,
        self_confidence=self_confidence,
        results_table=results_table,
        metrics=metrics,
    )