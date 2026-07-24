from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_CLASS_NAMES = [
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
]


def get_transforms(train: bool = False) -> transforms.Compose:
    """
    Create image transformations for CIFAR-10.

    Args:
        train:
            Whether the transformations are used for the training set.
            Training augmentation will be added later.

    Returns:
        A torchvision transformation pipeline.
    """
    if train:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616),
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )


def load_cifar10(
    data_dir: str | Path = "data",
    download: bool = True,
) -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """
    Load the CIFAR-10 training and test datasets.

    Args:
        data_dir:
            Directory used to store the dataset.
        download:
            Download CIFAR-10 if it is not already available.

    Returns:
        A tuple containing:
        - training dataset
        - test dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        transform=get_transforms(train=True),
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        transform=get_transforms(train=False),
        download=download,
    )

    return train_dataset, test_dataset


def get_class_distribution(
    dataset: datasets.CIFAR10,
) -> Dict[str, int]:
    """
    Count the number of samples in every class.

    Args:
        dataset:
            A CIFAR-10 dataset.

    Returns:
        A dictionary mapping class names to sample counts.
    """
    label_counts = Counter(dataset.targets)

    return {
        class_name: label_counts[class_index]
        for class_index, class_name in enumerate(CIFAR10_CLASS_NAMES)
    }


def create_dataloaders(
    train_dataset: datasets.CIFAR10,
    test_dataset: datasets.CIFAR10,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test DataLoaders.

    Args:
        train_dataset:
            CIFAR-10 training dataset.
        test_dataset:
            CIFAR-10 test dataset.
        batch_size:
            Number of samples in each batch.
        num_workers:
            Number of worker processes used for data loading.
            Windows beginners should start with 0.
        pin_memory:
            Whether to use pinned memory for faster CPU-to-GPU transfer.

    Returns:
        A tuple containing:
        - training DataLoader
        - test DataLoader
    """
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader