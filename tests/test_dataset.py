from src.dataset import (
    CIFAR10_CLASS_NAMES,
    create_dataloaders,
    get_class_distribution,
    load_cifar10,
)


def test_load_cifar10() -> None:
    train_dataset, test_dataset = load_cifar10(
        data_dir="data",
        download=True,
    )

    assert len(train_dataset) == 50_000
    assert len(test_dataset) == 10_000


def test_class_names() -> None:
    assert len(CIFAR10_CLASS_NAMES) == 10
    assert CIFAR10_CLASS_NAMES[0] == "airplane"
    assert CIFAR10_CLASS_NAMES[3] == "cat"


def test_class_distribution() -> None:
    train_dataset, _ = load_cifar10(
        data_dir="data",
        download=False,
    )

    distribution = get_class_distribution(train_dataset)

    assert len(distribution) == 10
    assert sum(distribution.values()) == 50_000
    assert all(count == 5_000 for count in distribution.values())


def test_create_dataloaders() -> None:
    train_dataset, test_dataset = load_cifar10(
        data_dir="data",
        download=False,
    )

    train_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=64,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    assert images.shape == (64, 3, 32, 32)
    assert labels.shape == (64,)
    assert len(train_loader.dataset) == 50_000
    assert len(test_loader.dataset) == 10_000