from collections import Counter
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def get_class_distribution(
    dataset: Dataset,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """
    统计数据集中每个类别的样本数量和比例。

    Args:
        dataset: PyTorch Dataset，要求包含 targets 属性。
        class_names: 类别名称列表。

    Returns:
        包含 label、class_name、count 和 percentage 的 DataFrame。
    """
    if not hasattr(dataset, "targets"):
        raise AttributeError(
            "Dataset must contain a 'targets' attribute."
        )

    targets = dataset.targets
    counter = Counter(targets)
    total_samples = len(targets)

    records = []

    for label, class_name in enumerate(class_names):
        count = counter.get(label, 0)
        percentage = count / total_samples * 100 if total_samples > 0 else 0

        records.append(
            {
                "label": label,
                "class_name": class_name,
                "count": count,
                "percentage": percentage,
            }
        )

    return pd.DataFrame(records)


def compute_channel_statistics(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算整个图像数据集每个颜色通道的均值和标准差。

    Dataset 中的图片应该已经通过 ToTensor 转换，
    像素范围应为 [0, 1]。

    Args:
        dataset: 图像数据集。
        batch_size: 每个批次的图片数量。
        num_workers: DataLoader 使用的工作进程数。

    Returns:
        mean: shape 为 [channels] 的 Tensor。
        std: shape 为 [channels] 的 Tensor。
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = None
    channel_squared_sum = None
    pixel_count = 0

    for images, _ in loader:
        if images.ndim != 4:
            raise ValueError(
                "Expected image batch shape [B, C, H, W], "
                f"but received {tuple(images.shape)}."
            )

        images = images.float()

        batch_channel_sum = images.sum(dim=(0, 2, 3))
        batch_channel_squared_sum = (images ** 2).sum(dim=(0, 2, 3))

        if channel_sum is None:
            channel_sum = torch.zeros_like(batch_channel_sum)
            channel_squared_sum = torch.zeros_like(
                batch_channel_squared_sum
            )

        channel_sum += batch_channel_sum
        channel_squared_sum += batch_channel_squared_sum

        pixel_count += images.shape[0] * images.shape[2] * images.shape[3]

    if pixel_count == 0:
        raise ValueError("The dataset is empty.")

    mean = channel_sum / pixel_count

    variance = channel_squared_sum / pixel_count - mean ** 2
    variance = torch.clamp(variance, min=0)

    std = torch.sqrt(variance)

    return mean, std


def inspect_image_properties(dataset: Dataset) -> dict:
    """
    检查数据集中第一张图片的基本属性。

    Returns:
        包含图片尺寸、通道数、dtype 和像素范围的字典。
    """
    if len(dataset) == 0:
        raise ValueError("The dataset is empty.")

    image, label = dataset[0]

    if not isinstance(image, torch.Tensor):
        raise TypeError(
            "Expected images to be torch.Tensor. "
            "Please use transforms.ToTensor()."
        )

    if image.ndim != 3:
        raise ValueError(
            f"Expected image shape [C, H, W], received {tuple(image.shape)}."
        )

    channels, height, width = image.shape

    return {
        "channels": channels,
        "height": height,
        "width": width,
        "dtype": str(image.dtype),
        "pixel_min": float(image.min()),
        "pixel_max": float(image.max()),
        "first_label": int(label),
    }


def save_dataset_summary(
    output_path: str | Path,
    dataset_name: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    class_names: Sequence[str],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    """
    将数据集基本信息保存为文本报告。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_properties = inspect_image_properties(train_dataset)

    mean_text = ", ".join(f"{value:.4f}" for value in mean.tolist())
    std_text = ", ".join(f"{value:.4f}" for value in std.tolist())

    report = f"""Dataset EDA Report
==============================

Dataset Name: {dataset_name}

Dataset Size
------------
Training Samples: {len(train_dataset)}
Test Samples: {len(test_dataset)}
Total Samples: {len(train_dataset) + len(test_dataset)}

Classes
-------
Number of Classes: {len(class_names)}
Class Names: {", ".join(class_names)}

Image Properties
----------------
Image Width: {image_properties["width"]}
Image Height: {image_properties["height"]}
Channels: {image_properties["channels"]}
Data Type: {image_properties["dtype"]}
Pixel Minimum: {image_properties["pixel_min"]:.4f}
Pixel Maximum: {image_properties["pixel_max"]:.4f}

Channel Statistics
------------------
Mean: [{mean_text}]
Standard Deviation: [{std_text}]
"""

    output_path.write_text(report, encoding="utf-8")