from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn


def set_random_seed(seed: int) -> None:
    """
    设置 Python、NumPy 和 PyTorch 随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """
    优先使用 NVIDIA GPU。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_yaml_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """
    加载 YAML 配置文件。
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "The YAML configuration must contain a dictionary."
        )

    return config


def save_json(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    将字典保存为 JSON。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serializable_data = {}

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, np.generic):
            serializable_data[key] = value.item()
        else:
            serializable_data[key] = value

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_data,
            file,
            indent=4,
            ensure_ascii=False,
        )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    validation_loss: float,
    validation_f1: float,
    output_path: str | Path,
) -> None:
    """
    保存训练 checkpoint。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation_loss": validation_loss,
        "validation_f1": validation_f1,
    }

    torch.save(
        checkpoint,
        output_path,
    )