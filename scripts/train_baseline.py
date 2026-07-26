from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.dataset import (
    create_data_loaders,
    create_datasets,
)
from src.model import (
    SimpleCNN,
    count_trainable_parameters,
)
from src.training import evaluate_model, fit
from src.utils import (
    load_yaml_config,
    save_json,
    select_device,
    set_random_seed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline CNN on CIFAR-10."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--noise-level",
        type=str,
        default="clean",
        choices=[
            "clean",
            "noise_10",
            "noise_20",
            "noise_30",
        ],
        help="Training label noise level.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config_path = PROJECT_ROOT / args.config
    config = load_yaml_config(config_path)

    seed = int(config["project"]["seed"])
    set_random_seed(seed)

    device = select_device()

    print("=" * 70)
    print("CIFAR-10 Baseline CNN Training")
    print("=" * 70)
    print(f"Noise level: {args.noise_level}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    scheduler_config = config["scheduler"]
    output_config = config["output"]

    data_dir = PROJECT_ROOT / data_config["data_dir"]
    noise_root = PROJECT_ROOT / data_config["noise_root"]

    experiment_name = (
        f"simple_cnn_{args.noise_level}"
    )

    checkpoint_path = (
        PROJECT_ROOT
        / output_config["checkpoint_dir"]
        / f"{experiment_name}_best.pt"
    )

    experiment_output_dir = (
        PROJECT_ROOT
        / output_config["training_dir"]
        / experiment_name
    )

    history_path = (
        experiment_output_dir
        / "training_history.csv"
    )

    test_metrics_path = (
        experiment_output_dir
        / "test_metrics.json"
    )

    print("\n[1/6] Creating datasets...")

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = create_datasets(
        data_dir=data_dir,
        noise_root=noise_root,
        noise_level=args.noise_level,
        validation_ratio=float(
            data_config["validation_ratio"]
        ),
        seed=seed,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("\n[2/6] Creating DataLoaders...")

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_data_loaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        pin_memory=(
            bool(data_config["pin_memory"])
            and device.type == "cuda"
        ),
        seed=seed,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\n[3/6] Creating model...")

    model = SimpleCNN(
        num_classes=int(model_config["num_classes"]),
        dropout=float(model_config["dropout"]),
    )

    model = model.to(device)

    number_of_parameters = count_trainable_parameters(
        model
    )

    print(model)
    print(
        f"Trainable parameters: "
        f"{number_of_parameters:,}"
    )

    print("\n[4/6] Creating loss, optimizer and scheduler...")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(
            training_config["label_smoothing"]
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(
            training_config["weight_decay"]
        ),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        min_lr=float(
            scheduler_config["minimum_learning_rate"]
        ),
    )

    print(f"Loss: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Scheduler: {scheduler.__class__.__name__}")

    print("\n[5/6] Training model...")

    fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(training_config["epochs"]),
        early_stopping_patience=int(
            training_config["early_stopping_patience"]
        ),
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print("\n[6/6] Evaluating on clean test set...")

    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        description="Testing",
    )

    test_result = {
        "experiment_name": experiment_name,
        "noise_level": args.noise_level,
        "checkpoint_epoch": checkpoint["epoch"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": (
            test_metrics["precision_macro"]
        ),
        "test_recall_macro": (
            test_metrics["recall_macro"]
        ),
        "test_f1_macro": (
            test_metrics["f1_macro"]
        ),
        "confusion_matrix": (
            test_metrics["confusion_matrix"]
        ),
        "class_names": CLASS_NAMES,
    }

    save_json(
        data=test_result,
        output_path=test_metrics_path,
    )

    print("\nTest Results")
    print("-" * 60)
    print(
        f"Test loss: "
        f"{test_metrics['loss']:.4f}"
    )
    print(
        f"Test accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Test precision macro: "
        f"{test_metrics['precision_macro']:.4f}"
    )
    print(
        f"Test recall macro: "
        f"{test_metrics['recall_macro']:.4f}"
    )
    print(
        f"Test F1 macro: "
        f"{test_metrics['f1_macro']:.4f}"
    )

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Metrics: {test_metrics_path}")
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()