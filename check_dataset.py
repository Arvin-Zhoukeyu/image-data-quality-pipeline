from src.dataset import (
    create_dataloaders,
    get_class_distribution,
    load_cifar10,
)


def main() -> None:
    train_dataset, test_dataset = load_cifar10()

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")

    distribution = get_class_distribution(train_dataset)

    print("\nClass distribution:")
    for class_name, count in distribution.items():
        print(f"{class_name:<12}: {count}")

    train_loader, _ = create_dataloaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=128,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("\nFirst batch:")
    print(f"Image tensor shape: {images.shape}")
    print(f"Label tensor shape: {labels.shape}")


if __name__ == "__main__":
    main()