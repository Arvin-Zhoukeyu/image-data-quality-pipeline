from pathlib import Path

from torchvision import datasets

from src.noise import (
    inject_symmetric_label_noise,
    save_noise_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "noisy_labels"

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

NOISE_RATES = [
    0.10,
    0.20,
    0.30,
]

RANDOM_SEED = 42


def load_clean_training_labels() -> list[int]:
    """
    加载 CIFAR-10 训练集的原始干净标签。

    这里不需要 transform，因为我们目前只读取 labels。
    """
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=None,
    )

    return list(train_dataset.targets)


def generate_noise_experiment(
    clean_labels: list[int],
    noise_rate: float,
) -> None:
    """
    为指定噪声比例生成并保存标签。
    """
    percentage = int(noise_rate * 100)

    experiment_output_dir = (
        OUTPUT_DIR / f"noise_{percentage}"
    )

    noisy_labels, noise_mask, noisy_indices = (
        inject_symmetric_label_noise(
            labels=clean_labels,
            noise_rate=noise_rate,
            num_classes=len(CLASS_NAMES),
            seed=RANDOM_SEED,
        )
    )

    save_noise_artifacts(
        output_dir=experiment_output_dir,
        clean_labels=clean_labels,
        noisy_labels=noisy_labels,
        noise_mask=noise_mask,
        class_names=CLASS_NAMES,
        noise_rate=noise_rate,
        seed=RANDOM_SEED,
    )

    print(f"\nNoise level: {percentage}%")
    print(f"Total samples: {len(clean_labels)}")
    print(f"Noisy samples: {len(noisy_indices)}")
    print(f"Clean samples: {len(clean_labels) - len(noisy_indices)}")
    print(f"Saved to: {experiment_output_dir}")


def main() -> None:
    print("=" * 60)
    print("CIFAR-10 Label Noise Injection")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading original CIFAR-10 labels...")
    clean_labels = load_clean_training_labels()

    print(f"Loaded {len(clean_labels)} clean training labels.")

    for noise_rate in NOISE_RATES:
        generate_noise_experiment(
            clean_labels=clean_labels,
            noise_rate=noise_rate,
        )

    print("\n" + "=" * 60)
    print("All noise experiments completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()