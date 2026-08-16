"""
Model 3: Residual CNN (ResNet-style) for character classification.

Residual (skip) connections solve the vanishing gradient problem,
allowing much deeper networks to train effectively.

Architecture:
    Input stem (7×7 Conv → BN → ReLU → MaxPool)
    → ResLayer 1: 2× ResBlock (64 filters)
    → ResLayer 2: 2× ResBlock (128 filters) with stride-2 downsampling
    → ResLayer 3: 2× ResBlock (256 filters) with stride-2 downsampling
    → ResLayer 4: 1× ResBlock (512 filters) with stride-2 downsampling
    → AdaptiveAvgPool → FC → num_classes

Lighter than full ResNet-50 but deeper than CNNBatchNorm —
optimal for 32×32 character images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Basic residual block (two 3×3 convolutions + skip connection).

    Pre-activation variant (BN → ReLU → Conv) for improved gradient flow.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        stride:       Stride for the first convolution. stride=2 halves spatial dims.
        dropout_rate: Stochastic depth / dropout inside block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            stride=1, padding=1, bias=False,
        )
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Projection shortcut when dimensions change
        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + identity


class BottleneckBlock(nn.Module):
    """
    Bottleneck residual block (1×1 → 3×3 → 1×1).

    More parameter-efficient for wider layers (≥256 filters).
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        out_channels = mid_channels * self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.conv3(F.relu(self.bn3(out), inplace=True))
        return out + identity


def make_layer(
    block: type,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 1,
    dropout_rate: float = 0.1,
) -> nn.Sequential:
    """Build a sequence of residual blocks."""
    layers = [block(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate)]
    for _ in range(1, num_blocks):
        layers.append(block(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ResidualCNN(nn.Module):
    """
    ResNet-style CNN for handwritten character recognition.

    Designed to fit 32×32 grayscale character images while remaining
    significantly lighter than full ResNet-34/50.

    Args:
        in_channels:   Input channels (1 for grayscale).
        num_classes:   Output class count.
        base_filters:  Filters in layer 1 (doubles per layer).
        layers:        Number of ResBlocks per layer [l1, l2, l3, l4].
        dropout_rate:  Dropout inside each ResBlock.
        fc_dropout:    Dropout before the final classifier.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 47,
        base_filters: int = 32,
        layers: list[int] | None = None,
        dropout_rate: float = 0.1,
        fc_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 1]
        f = base_filters

        # Stem: lightweight initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )

        # Residual layers
        self.layer1 = make_layer(ResBlock, f,     f,     layers[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = make_layer(ResBlock, f,     f * 2, layers[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = make_layer(ResBlock, f * 2, f * 4, layers[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = make_layer(ResBlock, f * 4, f * 8, layers[3], stride=2, dropout_rate=dropout_rate)

        # Final BN
        self.final_bn = nn.BatchNorm2d(f * 8)

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(fc_dropout),
            nn.Linear(f * 8, num_classes),
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.final_bn(x), inplace=True)
        x = self.global_pool(x)
        return self.classifier(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature maps after layer4 (before global pool)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return F.relu(self.final_bn(x), inplace=True)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def name(self) -> str:
        return "residual_cnn"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
