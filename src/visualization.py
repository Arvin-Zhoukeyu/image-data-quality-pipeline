import random
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    """
    将 [C, H, W] 格式的 Tensor 转换为
    matplotlib 使用的 [H, W, C] NumPy 数组。
    """
    if image.ndim != 3:
        raise ValueError(
            f"Expected image shape [C, H, W], received {tuple(image.shape)}."
        )

    image = image.detach().cpu()

    image = image.permute(1, 2, 0).numpy()

    return np.clip(image, 0.0, 1.0)


def plot_class_distribution(
    distribution_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    绘制并保存类别数量柱状图。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))

    plt.bar(
        distribution_df["class_name"],
        distribution_df["count"],
    )

    plt.title("CIFAR-10 Training Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_random_samples(
    dataset: Dataset,
    class_names: Sequence[str],
    output_path: str | Path,
    number_of_images: int = 16,
    seed: int = 42,
) -> None:
    """
    随机展示数据集中的图片。
    """
    if number_of_images <= 0:
        raise ValueError("number_of_images must be greater than zero.")

    if number_of_images > len(dataset):
        raise ValueError(
            "number_of_images cannot be greater than dataset size."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random_generator = random.Random(seed)
    selected_indices = random_generator.sample(
        range(len(dataset)),
        number_of_images,
    )

    columns = 4
    rows = int(np.ceil(number_of_images / columns))

    plt.figure(figsize=(10, rows * 2.6))

    for position, dataset_index in enumerate(selected_indices):
        image, label = dataset[dataset_index]

        axis = plt.subplot(rows, columns, position + 1)
        axis.imshow(tensor_to_image(image))
        axis.set_title(class_names[label])
        axis.axis("off")

    plt.suptitle("Random CIFAR-10 Training Samples")
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_samples_by_class(
    dataset: Dataset,
    class_names: Sequence[str],
    output_path: str | Path,
    samples_per_class: int = 5,
    seed: int = 42,
) -> None:
    """
    为每个类别随机展示若干张图片。

    每一行代表一个类别。
    """
    if not hasattr(dataset, "targets"):
        raise AttributeError(
            "Dataset must contain a 'targets' attribute."
        )

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be greater than zero.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random_generator = random.Random(seed)

    class_to_indices: dict[int, list[int]] = {
        label: [] for label in range(len(class_names))
    }

    for index, label in enumerate(dataset.targets):
        if label in class_to_indices:
            class_to_indices[label].append(index)

    rows = len(class_names)
    columns = samples_per_class

    plt.figure(figsize=(columns * 2.2, rows * 2.1))

    for label, class_name in enumerate(class_names):
        available_indices = class_to_indices[label]

        if len(available_indices) < samples_per_class:
            raise ValueError(
                f"Class '{class_name}' contains fewer than "
                f"{samples_per_class} samples."
            )

        selected_indices = random_generator.sample(
            available_indices,
            samples_per_class,
        )

        for column, dataset_index in enumerate(selected_indices):
            image, _ = dataset[dataset_index]

            plot_position = label * columns + column + 1
            axis = plt.subplot(rows, columns, plot_position)

            axis.imshow(tensor_to_image(image))
            axis.axis("off")

            if column == 0:
                axis.set_ylabel(
                    class_name,
                    rotation=0,
                    labelpad=45,
                    va="center",
                )

    plt.suptitle("CIFAR-10 Samples by Class")
    plt.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()