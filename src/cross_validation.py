from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from src.dataset import (
    CIFAR10WithCustomLabels,
    get_evaluation_transform,
    get_train_transform,
    load_custom_labels,
)
from src.model import SimpleCNN


@dataclass
class CrossValidationConfig:
    """
    交叉验证训练配置。
    """

    data_dir: str | Path
    noise_root: str | Path
    noise_level: str
    num_classes: int = 10
    num_folds: int = 5
    epochs_per_fold: int = 10
    batch_size: int = 128
    num_workers: int = 0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


def set_random_seed(seed: int) -> None:
    """
    设置随机种子，提高实验可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_cross_validation_datasets(
    data_dir: str | Path,
    noise_root: str | Path,
    noise_level: str,
) -> tuple[Dataset, Dataset, np.ndarray]:
    """
    创建交叉验证使用的两个 Dataset。

    train_dataset:
        使用随机数据增强，用于训练每个 fold。

    evaluation_dataset:
        不使用随机数据增强，用于预测 held-out fold。

    两个 Dataset 使用完全相同的标签和样本顺序。
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

    evaluation_base_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_evaluation_transform(),
    )

    train_dataset = CIFAR10WithCustomLabels(
        base_dataset=train_base_dataset,
        custom_labels=custom_labels,
    )

    evaluation_dataset = CIFAR10WithCustomLabels(
        base_dataset=evaluation_base_dataset,
        custom_labels=custom_labels,
    )

    labels = np.asarray(
        train_dataset.targets,
        dtype=np.int64,
    )

    return train_dataset, evaluation_dataset, labels


def create_model(
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    每个 fold 创建一个全新的模型。

    如果你的 SimpleCNN 构造函数没有 num_classes 参数，
    将下面一行改成：

        model = SimpleCNN()
    """
    model = SimpleCNN(
        num_classes=num_classes,
    )

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    训练一个 epoch。
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    在 held-out fold 上进行验证。
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


@torch.inference_mode()
def predict_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> np.ndarray:
    """
    生成类别预测概率。

    返回形状：

        [样本数量, 类别数量]
    """
    model.eval()

    probability_batches: list[np.ndarray] = []

    for images, _ in data_loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(images)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        probability_batches.append(
            probabilities.cpu().numpy()
        )

    if not probability_batches:
        return np.empty(
            shape=(0, num_classes),
            dtype=np.float32,
        )

    return np.concatenate(
        probability_batches,
        axis=0,
    ).astype(np.float32)


def generate_out_of_sample_predictions(
    config: CrossValidationConfig,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """
    使用 Stratified K-fold Cross Validation，为所有训练样本生成
    out-of-sample predicted probabilities。

    每个样本只由没有训练过该样本的模型预测。
    """
    set_random_seed(config.seed)

    if config.num_folds < 2:
        raise ValueError(
            "num_folds must be at least 2."
        )

    if config.epochs_per_fold <= 0:
        raise ValueError(
            "epochs_per_fold must be greater than zero."
        )

    if config.batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    train_dataset, evaluation_dataset, labels = (
        create_cross_validation_datasets(
            data_dir=config.data_dir,
            noise_root=config.noise_root,
            noise_level=config.noise_level,
        )
    )

    number_of_samples = len(labels)

    out_of_sample_pred_probs = np.zeros(
        shape=(
            number_of_samples,
            config.num_classes,
        ),
        dtype=np.float32,
    )

    prediction_count = np.zeros(
        number_of_samples,
        dtype=np.int64,
    )

    stratified_kfold = StratifiedKFold(
        n_splits=config.num_folds,
        shuffle=True,
        random_state=config.seed,
    )

    fold_histories: list[dict[str, float]] = []

    pin_memory = device.type == "cuda"

    print(
        f"Device: {device}\n"
        f"Noise level: {config.noise_level}\n"
        f"Samples: {number_of_samples}\n"
        f"Folds: {config.num_folds}"
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        stratified_kfold.split(
            np.zeros(number_of_samples),
            labels,
        ),
        start=1,
    ):
        print(
            "\n"
            + "=" * 60
            + f"\nFold {fold_number}/{config.num_folds}"
            + f"\nTrain samples: {len(train_indices)}"
            + f"\nValidation samples: {len(validation_indices)}"
            + "\n"
            + "=" * 60
        )

        fold_seed = config.seed + fold_number
        set_random_seed(fold_seed)

        train_subset = Subset(
            train_dataset,
            train_indices.tolist(),
        )

        validation_subset = Subset(
            evaluation_dataset,
            validation_indices.tolist(),
        )

        data_loader_generator = torch.Generator()
        data_loader_generator.manual_seed(fold_seed)

        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.num_workers > 0,
            generator=data_loader_generator,
        )

        validation_loader = DataLoader(
            dataset=validation_subset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=config.num_workers > 0,
        )

        model = create_model(
            num_classes=config.num_classes,
            device=device,
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        best_validation_loss = float("inf")
        best_model_state: Optional[dict] = None

        for epoch in range(
            1,
            config.epochs_per_fold + 1,
        ):
            train_loss, train_accuracy = train_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            validation_loss, validation_accuracy = (
                evaluate_model(
                    model=model,
                    data_loader=validation_loader,
                    criterion=criterion,
                    device=device,
                )
            )

            scheduler.step(validation_loss)

            learning_rate = optimizer.param_groups[0]["lr"]

            print(
                f"Fold {fold_number} | "
                f"Epoch {epoch:02d}/"
                f"{config.epochs_per_fold:02d} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_accuracy:.4f} | "
                f"Val loss: {validation_loss:.4f} | "
                f"Val acc: {validation_accuracy:.4f} | "
                f"LR: {learning_rate:.6f}"
            )

            fold_histories.append(
                {
                    "fold": float(fold_number),
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "learning_rate": learning_rate,
                }
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                best_model_state = copy.deepcopy(
                    model.state_dict()
                )

        if best_model_state is None:
            raise RuntimeError(
                f"Fold {fold_number} did not produce a model."
            )

        model.load_state_dict(best_model_state)

        fold_pred_probs = predict_probabilities(
            model=model,
            data_loader=validation_loader,
            device=device,
            num_classes=config.num_classes,
        )

        if len(fold_pred_probs) != len(validation_indices):
            raise RuntimeError(
                "The number of predictions does not match "
                "the validation fold size."
            )

        out_of_sample_pred_probs[
            validation_indices
        ] = fold_pred_probs

        prediction_count[
            validation_indices
        ] += 1

        # 显式删除本 fold 使用的 GPU 对象。
        # 单个程序连续训练多个 fold，因此这里进行清理比较合适。
        del model
        del optimizer
        del scheduler

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not np.all(prediction_count == 1):
        incorrect_indices = np.where(
            prediction_count != 1
        )[0]

        raise RuntimeError(
            "Each sample must receive exactly one "
            "out-of-sample prediction. "
            f"Problematic indices: {incorrect_indices[:10]}"
        )

    probability_sums = out_of_sample_pred_probs.sum(
        axis=1
    )

    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Predicted probabilities do not sum to one."
        )

    return (
        out_of_sample_pred_probs,
        labels,
        fold_histories,
    )