import numpy as np
import pytest

from src.noise import (
    build_noise_metadata,
    inject_symmetric_label_noise,
    validate_noise_rate,
)


def test_validate_noise_rate_accepts_valid_values():
    validate_noise_rate(0.0)
    validate_noise_rate(0.1)
    validate_noise_rate(0.3)
    validate_noise_rate(0.99)


@pytest.mark.parametrize(
    "invalid_rate",
    [-0.1, 1.0, 1.5],
)
def test_validate_noise_rate_rejects_invalid_values(
    invalid_rate,
):
    with pytest.raises(ValueError):
        validate_noise_rate(invalid_rate)


def test_injected_noise_has_expected_number_of_samples():
    labels = np.array(
        [0, 1, 2, 3, 4] * 20,
        dtype=np.int64,
    )

    noisy_labels, noise_mask, noisy_indices = (
        inject_symmetric_label_noise(
            labels=labels,
            noise_rate=0.2,
            num_classes=5,
            seed=42,
        )
    )

    expected_noisy_count = 20

    assert len(noisy_labels) == len(labels)
    assert noise_mask.sum() == expected_noisy_count
    assert len(noisy_indices) == expected_noisy_count


def test_noisy_labels_are_different_from_original_labels():
    labels = np.array(
        [0, 1, 2, 3, 4] * 20,
        dtype=np.int64,
    )

    noisy_labels, noise_mask, noisy_indices = (
        inject_symmetric_label_noise(
            labels=labels,
            noise_rate=0.3,
            num_classes=5,
            seed=42,
        )
    )

    assert np.all(
        noisy_labels[noise_mask] != labels[noise_mask]
    )

    assert np.all(
        noisy_labels[~noise_mask] == labels[~noise_mask]
    )


def test_noise_injection_is_reproducible():
    labels = np.array(
        [0, 1, 2, 3, 4] * 20,
        dtype=np.int64,
    )

    first_result = inject_symmetric_label_noise(
        labels=labels,
        noise_rate=0.2,
        num_classes=5,
        seed=42,
    )

    second_result = inject_symmetric_label_noise(
        labels=labels,
        noise_rate=0.2,
        num_classes=5,
        seed=42,
    )

    for first_array, second_array in zip(
        first_result,
        second_result,
    ):
        assert np.array_equal(
            first_array,
            second_array,
        )


def test_different_seeds_produce_different_noise():
    labels = np.array(
        [0, 1, 2, 3, 4] * 100,
        dtype=np.int64,
    )

    noisy_labels_seed_42, _, _ = (
        inject_symmetric_label_noise(
            labels=labels,
            noise_rate=0.2,
            num_classes=5,
            seed=42,
        )
    )

    noisy_labels_seed_100, _, _ = (
        inject_symmetric_label_noise(
            labels=labels,
            noise_rate=0.2,
            num_classes=5,
            seed=100,
        )
    )

    assert not np.array_equal(
        noisy_labels_seed_42,
        noisy_labels_seed_100,
    )


def test_build_noise_metadata():
    clean_labels = np.array([0, 1, 2, 0])
    noisy_labels = np.array([0, 2, 2, 1])

    class_names = [
        "cat",
        "dog",
        "bird",
    ]

    metadata = build_noise_metadata(
        clean_labels=clean_labels,
        noisy_labels=noisy_labels,
        class_names=class_names,
    )

    assert len(metadata) == 4
    assert metadata["is_noisy"].sum() == 2

    assert metadata.loc[1, "clean_class_name"] == "dog"
    assert metadata.loc[1, "noisy_class_name"] == "bird"