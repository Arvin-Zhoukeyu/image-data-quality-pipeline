from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision import datasets


CIFAR10_CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def ensure_parent_directory(
    output_path: str | Path,
) -> Path:
    """
    创建输出文件的父目录。
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_path


def plot_quality_score_histogram(
    quality_scores: np.ndarray,
    issue_mask: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    比较所有样本与 Cleanlab 检测样本的标签质量分数。
    """
    output_path = ensure_parent_directory(
        output_path
    )

    quality_scores = np.asarray(
        quality_scores
    )

    issue_mask = np.asarray(
        issue_mask,
        dtype=bool,
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.hist(
        quality_scores,
        bins=40,
        alpha=0.65,
        label="All samples",
    )

    if issue_mask.any():
        axis.hist(
            quality_scores[issue_mask],
            bins=40,
            alpha=0.65,
            label="Detected label issues",
        )

    axis.set_title(
        "Distribution of Cleanlab Label Quality Scores"
    )

    axis.set_xlabel(
        "Label quality score"
    )

    axis.set_ylabel(
        "Number of samples"
    )

    axis.legend()
    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_detection_metrics(
    metrics: dict[str, float | int],
    output_path: str | Path,
) -> None:
    """
    绘制 Precision、Recall 和 F1。
    """
    required_metrics = [
        "precision",
        "recall",
        "f1",
    ]

    if not all(
        metric in metrics
        for metric in required_metrics
    ):
        return

    output_path = ensure_parent_directory(
        output_path
    )

    metric_names = [
        "Precision",
        "Recall",
        "F1",
    ]

    metric_values = [
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["f1"]),
    ]

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    bars = axis.bar(
        metric_names,
        metric_values,
    )

    axis.set_title(
        "Cleanlab Label-Issue Detection Performance"
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        metric_values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_confidence_comparison(
    results_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    比较正常样本和 Cleanlab 可疑样本的 self-confidence。
    """
    output_path = ensure_parent_directory(
        output_path
    )

    normal_scores = results_table.loc[
        ~results_table["is_label_issue"],
        "self_confidence",
    ].to_numpy()

    issue_scores = results_table.loc[
        results_table["is_label_issue"],
        "self_confidence",
    ].to_numpy()

    data = [normal_scores]

    labels = ["Not detected"]

    if len(issue_scores) > 0:
        data.append(issue_scores)
        labels.append("Detected issue")

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
    )

    axis.set_title(
        "Self-Confidence by Detection Status"
    )

    axis.set_ylabel(
        "Probability assigned to given label"
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_top_label_issues(
    data_dir: str | Path,
    results_table: pd.DataFrame,
    output_path: str | Path,
    maximum_samples: int = 16,
    class_names: Sequence[str] = CIFAR10_CLASS_NAMES,
) -> None:
    """
    显示 Cleanlab 排名最靠前的可疑标签样本。
    """
    if maximum_samples <= 0:
        raise ValueError(
            "maximum_samples must be greater than zero."
        )

    output_path = ensure_parent_directory(
        output_path
    )

    issue_rows = (
        results_table[
            results_table["is_label_issue"]
        ]
        .sort_values(
            by="issue_rank",
            ascending=True,
        )
        .head(maximum_samples)
    )

    if issue_rows.empty:
        return

    raw_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=None,
    )

    number_of_images = len(issue_rows)
    number_of_columns = 4
    number_of_rows = int(
        np.ceil(
            number_of_images
            / number_of_columns
        )
    )

    figure, axes = plt.subplots(
        nrows=number_of_rows,
        ncols=number_of_columns,
        figsize=(
            14,
            3.6 * number_of_rows,
        ),
    )

    axes_array = np.asarray(
        axes
    ).reshape(-1)

    for axis in axes_array:
        axis.axis("off")

    for axis, (_, row) in zip(
        axes_array,
        issue_rows.iterrows(),
    ):
        sample_index = int(
            row["sample_index"]
        )

        image, _ = raw_dataset[
            sample_index
        ]

        given_label = int(
            row["given_label"]
        )

        predicted_label = int(
            row["predicted_label"]
        )

        quality_score = float(
            row["label_quality_score"]
        )

        predicted_confidence = float(
            row["predicted_confidence"]
        )

        axis.imshow(image)
        axis.axis("off")

        title = (
            f"Index: {sample_index}\n"
            f"Given: {class_names[given_label]}\n"
            f"Pred: {class_names[predicted_label]}\n"
            f"Quality: {quality_score:.3f} | "
            f"Conf: {predicted_confidence:.3f}"
        )

        if "is_true_noise" in row.index:
            true_noise = bool(
                row["is_true_noise"]
            )

            title += (
                "\nInjected noise: "
                + ("Yes" if true_noise else "No")
            )

        axis.set_title(
            title,
            fontsize=9,
        )

    figure.suptitle(
        "Top-Ranked Potential Label Issues",
        fontsize=15,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)