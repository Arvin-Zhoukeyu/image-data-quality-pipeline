from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CIFAR10WithCustomLabels(Dataset):
    """
    CIFAR-10 Dataset wrapper。

    图片来自原始 CIFAR-10，标签可以使用：
    1. 原始干净标签
    2. Day 4 生成的 noisy labels
    """

    def __init__(
        self,
        base_dataset: datasets.CIFAR10,
        custom_labels: Optional[Sequence[int] | np.ndarray] = None,
    ) -> None:
        self.base_dataset = base_dataset

        original_labels = np.asarray(
            base_dataset.targets,
            dtype=np.int64,
        )

        if custom_labels is None:
            self.labels = original_labels.copy()
        else:
            custom_labels_array = np.asarray(
                custom_labels,
                dtype=np.int64,
            )

            if custom_labels_array.ndim != 1:
                raise ValueError(
                    "custom_labels must be a one-dimensional array."
                )

            if len(custom_labels_array) != len(base_dataset):
                raise ValueError(
                    "The number of custom labels must match "
                    "the number of dataset samples."
                )

            self.labels = custom_labels_array.copy()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.base_dataset[index]
        label = int(self.labels[index])

        return image, label

    @property
    def targets(self) -> list[int]:
        """
        提供与 torchvision Dataset 类似的 targets 接口。
        """
        return self.labels.tolist()


def get_train_transform() -> transforms.Compose:
    """
    训练集数据增强。

    RandomCrop 和 RandomHorizontalFlip 只用于训练集。
    """
    return transforms.Compose(
        [
            transforms.RandomCrop(
                size=32,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )


def get_evaluation_transform() -> transforms.Compose:
    """
    验证集和测试集不使用随机数据增强。
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR10_MEAN,
                std=CIFAR10_STD,
            ),
        ]
    )


def load_custom_labels(
    noise_root: str | Path,
    noise_level: str,
) -> Optional[np.ndarray]:
    """
    根据实验名称加载训练标签。

    Args:
        noise_root:
            Day 4 噪声标签保存目录。

        noise_level:
            clean、noise_10、noise_20 或 noise_30。

    Returns:
        clean 实验返回 None，表示使用 CIFAR-10 原始标签。
        噪声实验返回 noisy_labels.npy。
    """
    valid_noise_levels = {
        "clean",
        "noise_10",
        "noise_20",
        "noise_30",
    }

    if noise_level not in valid_noise_levels:
        raise ValueError(
            f"Unsupported noise level: {noise_level}. "
            f"Expected one of {sorted(valid_noise_levels)}."
        )

    if noise_level == "clean":
        return None

    labels_path = (
        Path(noise_root)
        / noise_level
        / "noisy_labels.npy"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Noisy labels were not found: {labels_path}\n"
            "Please run: python -m scripts.inject_noise"
        )

    labels = np.load(labels_path)

    return labels.astype(np.int64)


def create_train_validation_indices(
    total_samples: int,
    validation_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    创建可复现的训练集和验证集索引。
    """
    if total_samples <= 0:
        raise ValueError("total_samples must be greater than zero.")

    if validation_ratio <= 0 or validation_ratio >= 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    generator = np.random.default_rng(seed)

    shuffled_indices = generator.permutation(total_samples)

    validation_size = int(
        round(total_samples * validation_ratio)
    )

    validation_indices = shuffled_indices[:validation_size]
    train_indices = shuffled_indices[validation_size:]

    return (
        train_indices.tolist(),
        validation_indices.tolist(),
    )


def create_datasets(
    data_dir: str | Path,
    noise_root: str | Path,
    noise_level: str,
    validation_ratio: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    创建训练集、验证集和测试集。

    训练集和验证集使用相同标签，
    但使用不同图像 transform。
    """
    custom_labels = load_custom_labels(
        noise_root=noise_root,
        noise_level=noise_level,
    )

    train_base_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_train_transform(),
    )

    validation_base_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_evaluation_transform(),
    )

    test_base_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_evaluation_transform(),
    )

    train_full_dataset = CIFAR10WithCustomLabels(
        base_dataset=train_base_dataset,
        custom_labels=custom_labels,
    )

    validation_full_dataset = CIFAR10WithCustomLabels(
        base_dataset=validation_base_dataset,
        custom_labels=custom_labels,
    )

    test_dataset = CIFAR10WithCustomLabels(
        base_dataset=test_base_dataset,
        custom_labels=None,
    )

    train_indices, validation_indices = (
        create_train_validation_indices(
            total_samples=len(train_full_dataset),
            validation_ratio=validation_ratio,
            seed=seed,
        )
    )

    train_dataset = Subset(
        train_full_dataset,
        train_indices,
    )

    validation_dataset = Subset(
        validation_full_dataset,
        validation_indices,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def create_data_loaders(
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    将 Dataset 包装为 DataLoader。

    Dataset 决定一个样本如何读取；
    DataLoader 决定如何组成 batch。
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=num_workers > 0,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, validation_loader, test_loader