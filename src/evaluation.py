from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_classification_metrics(
    targets: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
) -> dict[str, Any]:
    """
    计算多分类任务指标。

    使用 macro average：
    每个类别权重相同。
    """
    targets_array = np.asarray(targets)
    predictions_array = np.asarray(predictions)

    if targets_array.shape != predictions_array.shape:
        raise ValueError(
            "targets and predictions must have the same shape."
        )

    accuracy = accuracy_score(
        targets_array,
        predictions_array,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            targets_array,
            predictions_array,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        targets_array,
        predictions_array,
    )

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": matrix,
    }