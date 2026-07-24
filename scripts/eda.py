from pathlib import Path

import pandas as pd
from torchvision import datasets, transforms

from src.dataset_statistics import (
    compute_channel_statistics,
    get_class_distribution,
    inspect_image_properties,
    save_dataset_summary,
)
from src.visualization import (
    plot_class_distribution,
    plot_random_samples,
    plot_samples_by_class,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

CLASS_NAMES = [
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


def load_eda_datasets():
    """
    加载用于 EDA 的 CIFAR-10。

    此处只使用 ToTensor，不使用 Normalize，
    这样才能分析原始像素分布。
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def main() -> None:
    print("=" * 50)
    print("Starting CIFAR-10 exploratory data analysis")
    print("=" * 50)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading CIFAR-10...")
    train_dataset, test_dataset = load_eda_datasets()

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("\n[2/6] Inspecting image properties...")
    image_properties = inspect_image_properties(train_dataset)

    for key, value in image_properties.items():
        print(f"{key}: {value}")

    print("\n[3/6] Calculating class distribution...")
    distribution_df = get_class_distribution(
        dataset=train_dataset,
        class_names=CLASS_NAMES,
    )

    distribution_csv_path = (
        REPORT_DIR / "class_distribution.csv"
    )

    distribution_df.to_csv(
        distribution_csv_path,
        index=False,
        encoding="utf-8",
    )

    print(distribution_df.to_string(index=False))

    print("\n[4/6] Calculating RGB channel statistics...")
    mean, std = compute_channel_statistics(
        dataset=train_dataset,
        batch_size=256,
        num_workers=0,
    )

    print(f"RGB mean: {mean.tolist()}")
    print(f"RGB standard deviation: {std.tolist()}")

    statistics_df = pd.DataFrame(
        {
            "channel": ["red", "green", "blue"],
            "mean": mean.tolist(),
            "standard_deviation": std.tolist(),
        }
    )

    statistics_df.to_csv(
        REPORT_DIR / "channel_statistics.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n[5/6] Generating visualisations...")

    plot_class_distribution(
        distribution_df=distribution_df,
        output_path=REPORT_DIR / "class_distribution.png",
    )

    plot_random_samples(
        dataset=train_dataset,
        class_names=CLASS_NAMES,
        output_path=REPORT_DIR / "random_samples.png",
        number_of_images=16,
        seed=42,
    )

    plot_samples_by_class(
        dataset=train_dataset,
        class_names=CLASS_NAMES,
        output_path=REPORT_DIR / "samples_by_class.png",
        samples_per_class=5,
        seed=42,
    )

    print("\n[6/6] Writing dataset summary...")

    save_dataset_summary(
        output_path=REPORT_DIR / "dataset_summary.txt",
        dataset_name="CIFAR-10",
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        class_names=CLASS_NAMES,
        mean=mean,
        std=std,
    )

    print("\nEDA completed successfully.")
    print(f"Reports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()