"""
Model 2: CNN with Batch Normalization.

Batch Normalization between Conv and ReLU stabilizes training,
enables higher learning rates, and acts as a regularizer —
reducing the need for heavy Dropout.

Architecture improvements over CNNBasic:
    - BatchNorm2d after every Conv2d
    - Deeper feature extraction (5 blocks)
    - Squeeze-and-Excitation (SE) attention after each block
      for channel-wise feature recalibration
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Learns to recalibrate channel-wise feature responses adaptively
    by explicitly modelling inter-channel dependencies.

    Reference: Hu et al. 2018 — "Squeeze-and-Excitation Networks"
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # global average pool
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ---------------------------------------------------------------------------
# BN Conv block
# ---------------------------------------------------------------------------

class ConvBNBlock(nn.Module):
    """Conv → BN → ReLU → optional SE → optional MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool: bool = True,
        use_se: bool = True,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels, se_reduction) if use_se else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.se(x)
        return self.pool(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CNNBatchNorm(nn.Module):
    """
    5-block CNN with Batch Normalization and SE channel attention.

    Args:
        in_channels:    Input channels (1 for grayscale).
        num_classes:    Number of output classes.
        base_filters:   Starting filter count (doubles each block).
        dropout_rate:   Classifier dropout probability.
        fc_hidden_size: Hidden FC layer size.
        use_se:         Enable Squeeze-and-Excitation blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 47,
        base_filters: int = 32,
        dropout_rate: float = 0.25,
        fc_hidden_size: int = 512,
        use_se: bool = True,
    ) -> None:
        super().__init__()

        f = base_filters
        self.features = nn.Sequential(
            ConvBNBlock(in_channels, f,     pool=True,  use_se=use_se),  # → f    × H/2 × W/2
            ConvBNBlock(f,          f * 2,  pool=True,  use_se=use_se),  # → 2f   × H/4 × W/4
            ConvBNBlock(f * 2,      f * 4,  pool=True,  use_se=use_se),  # → 4f   × H/8 × W/8
            ConvBNBlock(f * 4,      f * 8,  pool=False, use_se=use_se),  # → 8f   × H/8 × W/8
            ConvBNBlock(f * 8,      f * 8,  pool=False, use_se=use_se),  # → 8f   × H/8 × W/8
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        classifier_in = f * 8 * 2 * 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_in, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_size, fc_hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(fc_hidden_size // 2, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.adaptive_pool(x)

    @property
    def name(self) -> str:
        return "cnn_batchnorm"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
