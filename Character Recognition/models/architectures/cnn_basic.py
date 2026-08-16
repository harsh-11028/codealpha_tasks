"""
Model 1: Standard CNN for character classification.

Architecture:
    Input (1, H, W)
    → Conv Block 1: Conv2d → ReLU → MaxPool
    → Conv Block 2: Conv2d → ReLU → MaxPool
    → Conv Block 3: Conv2d → ReLU → MaxPool
    → Conv Block 4: Conv2d → ReLU → AdaptiveAvgPool
    → Flatten → FC(256) → Dropout → FC(num_classes)

Designed for:
    - Fast training on MNIST / EMNIST character images
    - Baseline comparison against more complex models
    - CPU-friendly inference
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single convolution block: Conv → ReLU → optional MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.conv(x)))


class CNNBasic(nn.Module):
    """
    Standard 4-block CNN for handwritten character classification.

    Args:
        in_channels:    Input image channels (1 for grayscale).
        num_classes:    Number of output classes.
        base_filters:   Number of filters in the first conv layer.
                        Doubles at each block (32 → 64 → 128 → 256).
        dropout_rate:   Dropout probability before the final classifier.
        fc_hidden_size: Size of the fully connected hidden layer.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 47,
        base_filters: int = 32,
        dropout_rate: float = 0.3,
        fc_hidden_size: int = 256,
    ) -> None:
        super().__init__()

        f = base_filters
        self.features = nn.Sequential(
            ConvBlock(in_channels, f,      kernel_size=3, padding=1, pool=True),   # → f × H/2 × W/2
            ConvBlock(f,          f * 2,   kernel_size=3, padding=1, pool=True),   # → 2f × H/4 × W/4
            ConvBlock(f * 2,      f * 4,   kernel_size=3, padding=1, pool=True),   # → 4f × H/8 × W/8
            ConvBlock(f * 4,      f * 8,   kernel_size=3, padding=1, pool=False),  # → 8f × H/8 × W/8
        )
        # Adaptive pool → 2×2 spatial regardless of input size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        classifier_in = f * 8 * 2 * 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(classifier_in, fc_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_size, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        """Kaiming initialization for conv layers, Xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return CNN feature maps before the classifier (for visualization)."""
        x = self.features(x)
        return self.adaptive_pool(x)

    @property
    def name(self) -> str:
        return "cnn_basic"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
