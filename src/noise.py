from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def validate_noise_rate(noise_rate: float) -> None:
    """
    检查噪声比例是否合法。

    noise_rate 应位于 [0, 1) 范围内。

    Examples:
        0.1 表示 10%
        0.2 表示 20%
    """
    if not isinstance(noise_rate, (int, float)):
        raise TypeError("noise_rate must be a numeric value.")

    if noise_rate < 0 or noise_rate >= 1:
        raise ValueError(
            "noise_rate must be greater than or equal to 0 "
            "and smaller than 1."
        )


def validate_labels(
    labels: Sequence[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    验证标签并转换为 NumPy 数组。

    Args:
        labels:
            原始标签。
        num_classes:
            类别数量。

    Returns:
        一维整数 NumPy 数组。
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    labels_array = np.asarray(labels, dtype=np.int64)

    if labels_array.ndim != 1:
        raise ValueError(
            f"labels must be one-dimensional, "
            f"but received shape {labels_array.shape}."
        )

    if labels_array.size == 0:
        raise ValueError("labels cannot be empty.")

    if labels_array.min() < 0:
        raise ValueError("labels cannot contain negative values.")

    if labels_array.max() >= num_classes:
        raise ValueError(
            "labels contain a class index greater than or equal "
            "to num_classes."
        )

    return labels_array


def inject_symmetric_label_noise(
    labels: Sequence[int] | np.ndarray,
    noise_rate: float,
    num_classes: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    向分类标签中注入对称标签噪声。

    对称噪声意味着：
    被选中的样本会随机变成任意一个其他类别，
    并且不会保持原来的标签。

    Args:
        labels:
            干净标签。
        noise_rate:
            噪声比例，例如 0.1、0.2、0.3。
        num_classes:
            类别数量。
        seed:
            随机种子，用于保证实验可复现。

    Returns:
        noisy_labels:
            注入噪声后的标签。

        noise_mask:
            布尔数组。True 表示该样本的标签被修改。

        noisy_indices:
            所有被修改样本的索引。
    """
    validate_noise_rate(noise_rate)

    clean_labels = validate_labels(
        labels=labels,
        num_classes=num_classes,
    )

    total_samples = len(clean_labels)
    number_of_noisy_samples = int(round(total_samples * noise_rate))

    random_generator = np.random.default_rng(seed)

    noisy_indices = random_generator.choice(
        total_samples,
        size=number_of_noisy_samples,
        replace=False,
    )

    noisy_indices = np.sort(noisy_indices)

    noisy_labels = clean_labels.copy()
    noise_mask = np.zeros(total_samples, dtype=bool)
    noise_mask[noisy_indices] = True

    for sample_index in noisy_indices:
        original_label = clean_labels[sample_index]

        # 从 0 到 num_classes - 2 之间随机选择一个数字。
        new_label = random_generator.integers(
            low=0,
            high=num_classes - 1,
        )

        # 将其映射到除 original_label 之外的类别。
        # 这样可以严格保证新标签与原标签不同。
        if new_label >= original_label:
            new_label += 1

        noisy_labels[sample_index] = new_label

    return noisy_labels, noise_mask, noisy_indices


def build_noise_metadata(
    clean_labels: Sequence[int] | np.ndarray,
    noisy_labels: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """
    为每个样本生成标签噪声元数据。

    返回的 DataFrame 中，每一行对应一个训练样本。
    """
    clean_labels_array = validate_labels(
        labels=clean_labels,
        num_classes=len(class_names),
    )

    noisy_labels_array = validate_labels(
        labels=noisy_labels,
        num_classes=len(class_names),
    )

    if len(clean_labels_array) != len(noisy_labels_array):
        raise ValueError(
            "clean_labels and noisy_labels must have the same length."
        )

    noise_mask = clean_labels_array != noisy_labels_array

    metadata = pd.DataFrame(
        {
            "sample_index": np.arange(len(clean_labels_array)),
            "clean_label": clean_labels_array,
            "clean_class_name": [
                class_names[label] for label in clean_labels_array
            ],
            "noisy_label": noisy_labels_array,
            "noisy_class_name": [
                class_names[label] for label in noisy_labels_array
            ],
            "is_noisy": noise_mask,
        }
    )

    return metadata


def save_noise_artifacts(
    output_dir: str | Path,
    clean_labels: Sequence[int] | np.ndarray,
    noisy_labels: Sequence[int] | np.ndarray,
    noise_mask: Sequence[bool] | np.ndarray,
    class_names: Sequence[str],
    noise_rate: float,
    seed: int,
) -> None:
    """
    保存噪声实验需要的全部文件。

    保存内容：
        clean_labels.npy
        noisy_labels.npy
        noise_mask.npy
        noisy_indices.npy
        noise_metadata.csv
        summary.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_labels_array = validate_labels(
        labels=clean_labels,
        num_classes=len(class_names),
    )

    noisy_labels_array = validate_labels(
        labels=noisy_labels,
        num_classes=len(class_names),
    )

    noise_mask_array = np.asarray(noise_mask, dtype=bool)

    if len(clean_labels_array) != len(noisy_labels_array):
        raise ValueError(
            "clean_labels and noisy_labels must have the same length."
        )

    if len(clean_labels_array) != len(noise_mask_array):
        raise ValueError(
            "noise_mask must have the same length as labels."
        )

    calculated_mask = clean_labels_array != noisy_labels_array

    if not np.array_equal(noise_mask_array, calculated_mask):
        raise ValueError(
            "noise_mask does not match the differences between "
            "clean_labels and noisy_labels."
        )

    noisy_indices = np.flatnonzero(noise_mask_array)

    metadata = build_noise_metadata(
        clean_labels=clean_labels_array,
        noisy_labels=noisy_labels_array,
        class_names=class_names,
    )

    np.save(
        output_dir / "clean_labels.npy",
        clean_labels_array,
    )

    np.save(
        output_dir / "noisy_labels.npy",
        noisy_labels_array,
    )

    np.save(
        output_dir / "noise_mask.npy",
        noise_mask_array,
    )

    np.save(
        output_dir / "noisy_indices.npy",
        noisy_indices,
    )

    metadata.to_csv(
        output_dir / "noise_metadata.csv",
        index=False,
        encoding="utf-8",
    )

    total_samples = len(clean_labels_array)
    number_of_noisy_samples = int(noise_mask_array.sum())
    actual_noise_rate = number_of_noisy_samples / total_samples

    summary = {
        "noise_type": "symmetric_label_noise",
        "requested_noise_rate": noise_rate,
        "actual_noise_rate": actual_noise_rate,
        "seed": seed,
        "total_samples": total_samples,
        "number_of_noisy_samples": number_of_noisy_samples,
        "number_of_clean_samples": (
            total_samples - number_of_noisy_samples
        ),
        "num_classes": len(class_names),
        "class_names": list(class_names),
    }

    with open(
        output_dir / "summary.json",
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )


def load_noise_artifacts(
    noise_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从磁盘加载噪声标签和 Ground Truth。

    Returns:
        clean_labels
        noisy_labels
        noise_mask
    """
    noise_dir = Path(noise_dir)

    clean_labels_path = noise_dir / "clean_labels.npy"
    noisy_labels_path = noise_dir / "noisy_labels.npy"
    noise_mask_path = noise_dir / "noise_mask.npy"

    required_paths = [
        clean_labels_path,
        noisy_labels_path,
        noise_mask_path,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

    clean_labels = np.load(clean_labels_path)
    noisy_labels = np.load(noisy_labels_path)
    noise_mask = np.load(noise_mask_path)

    return clean_labels, noisy_labels, noise_mask