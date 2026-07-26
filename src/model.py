from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    一个可重复使用的卷积模块：

    Conv2d
    → BatchNorm2d
    → ReLU
    → Conv2d
    → BatchNorm2d
    → ReLU
    → MaxPool2d
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """
    CIFAR-10 基础 CNN 分类器。

    不使用现成 ResNet，
    便于理解完整前向传播过程。
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than one."
            )

        if dropout < 0 or dropout >= 1:
            raise ValueError(
                "dropout must be in the range [0, 1)."
            )

        self.features = nn.Sequential(
            ConvBlock(
                input_channels=3,
                output_channels=32,
            ),
            ConvBlock(
                input_channels=32,
                output_channels=64,
            ),
            ConvBlock(
                input_channels=64,
                output_channels=128,
            ),
        )

        self.global_average_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=128,
                out_features=num_classes,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        使用适合 ReLU 的 Kaiming 初始化。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.01,
                )
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_average_pool(x)
        logits = self.classifier(x)

        return logits


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    计算模型中需要训练的参数数量。
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )