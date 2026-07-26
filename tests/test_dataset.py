import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.dataset import (
    CIFAR10WithCustomLabels,
    create_train_validation_indices,
)


class DummyDataset(Dataset):
    def __init__(self):
        self.targets = [0, 1, 2, 3, 4]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = torch.zeros(
            3,
            32,
            32,
        )

        return image, self.targets[index]


def test_dataset_uses_original_labels():
    base_dataset = DummyDataset()

    dataset = CIFAR10WithCustomLabels(
        base_dataset=base_dataset,
        custom_labels=None,
    )

    _, label = dataset[2]

    assert label == 2


def test_dataset_uses_custom_labels():
    base_dataset = DummyDataset()

    custom_labels = np.array(
        [4, 3, 2, 1, 0],
        dtype=np.int64,
    )

    dataset = CIFAR10WithCustomLabels(
        base_dataset=base_dataset,
        custom_labels=custom_labels,
    )

    _, label = dataset[0]

    assert label == 4


def test_dataset_rejects_wrong_number_of_labels():
    base_dataset = DummyDataset()

    with pytest.raises(ValueError):
        CIFAR10WithCustomLabels(
            base_dataset=base_dataset,
            custom_labels=[0, 1],
        )


def test_train_validation_split_has_no_overlap():
    train_indices, validation_indices = (
        create_train_validation_indices(
            total_samples=100,
            validation_ratio=0.2,
            seed=42,
        )
    )

    assert len(train_indices) == 80
    assert len(validation_indices) == 20

    assert set(train_indices).isdisjoint(
        set(validation_indices)
    )


def test_train_validation_split_is_reproducible():
    first_split = create_train_validation_indices(
        total_samples=100,
        validation_ratio=0.2,
        seed=42,
    )

    second_split = create_train_validation_indices(
        total_samples=100,
        validation_ratio=0.2,
        seed=42,
    )

    assert first_split == second_split