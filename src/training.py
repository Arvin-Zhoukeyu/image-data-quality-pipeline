from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation import calculate_classification_metrics
from src.utils import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """
    完成一个 epoch 的训练。
    """
    model.train()

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress_bar = tqdm(
        data_loader,
        desc="Training",
        leave=False,
    )

    for images, targets in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        # 1. 清除上一个 batch 的梯度
        optimizer.zero_grad(set_to_none=True)

        # 2. 前向传播
        logits = model(images)

        # 3. 计算损失
        loss = criterion(
            logits,
            targets,
        )

        # 4. 反向传播
        loss.backward()

        # 5. 更新模型参数
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        all_targets.extend(
            targets.detach().cpu().tolist()
        )

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = total_loss / len(data_loader.dataset)

    metrics = calculate_classification_metrics(
        targets=all_targets,
        predictions=all_predictions,
    )

    return {
        "loss": average_loss,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str = "Evaluating",
) -> dict[str, Any]:
    """
    在验证集或测试集上评估模型。

    @torch.no_grad() 表示不计算梯度，
    可降低显存和计算开销。
    """
    model.eval()

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress_bar = tqdm(
        data_loader,
        desc=description,
        leave=False,
    )

    for images, targets in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            targets,
        )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        all_targets.extend(
            targets.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

    average_loss = total_loss / len(data_loader.dataset)

    metrics = calculate_classification_metrics(
        targets=all_targets,
        predictions=all_predictions,
    )

    return {
        "loss": average_loss,
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int,
    early_stopping_patience: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
) -> pd.DataFrame:
    """
    完整模型训练流程。

    每个 epoch：
    1. 训练
    2. 验证
    3. Scheduler 更新学习率
    4. 保存最佳模型
    5. Early stopping 判断
    """
    history: list[dict[str, float | int]] = []

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)

        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            description="Validation",
        )

        current_learning_rate = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_precision_macro": (
                validation_metrics["precision_macro"]
            ),
            "validation_recall_macro": (
                validation_metrics["recall_macro"]
            ),
            "validation_f1_macro": (
                validation_metrics["f1_macro"]
            ),
        }

        history.append(epoch_record)

        history_df = pd.DataFrame(history)

        history_path = Path(history_path)
        history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        history_df.to_csv(
            history_path,
            index=False,
            encoding="utf-8",
        )

        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Train acc: {train_metrics['accuracy']:.4f} | "
            f"Train F1: {train_metrics['f1_macro']:.4f}"
        )

        print(
            f"Val loss: {validation_metrics['loss']:.4f} | "
            f"Val acc: {validation_metrics['accuracy']:.4f} | "
            f"Val F1: {validation_metrics['f1_macro']:.4f} | "
            f"LR: {current_learning_rate:.6f}"
        )

        # ReduceLROnPlateau 根据验证损失调整学习率
        scheduler.step(
            validation_metrics["loss"]
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                validation_loss=validation_metrics["loss"],
                validation_f1=validation_metrics["f1_macro"],
                output_path=checkpoint_path,
            )

            print("Best model checkpoint saved.")

        else:
            epochs_without_improvement += 1

            print(
                "Validation loss did not improve. "
                f"Patience: {epochs_without_improvement}/"
                f"{early_stopping_patience}"
            )

        if (
            early_stopping_patience > 0
            and epochs_without_improvement
            >= early_stopping_patience
        ):
            print("Early stopping triggered.")
            break

    return pd.DataFrame(history)