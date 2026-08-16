"""
Model 4: CRNN — Convolutional Recurrent Neural Network.

Architecture designed for sequence recognition (words / lines):
    Input (1, H, W)
    → CNN Backbone (feature extraction)     →  (C, H', W')
    → Column-wise sequence reshape           →  (W', batch, C×H')
    → BiLSTM × 2 layers                     →  (W', batch, 2×rnn_hidden)
    → Linear projection                      →  (W', batch, num_classes+1)
    → CTC Loss during training

The CTC (Connectionist Temporal Classification) loss enables training
without character-level alignment labels — only the full word transcript
is needed as supervision.

This model is the PRIMARY engine for word and sentence recognition.

Reference:
    Shi et al. (2016) — "An End-to-End Trainable Neural Network for
    Image-based Sequence Recognition and Its Application to Scene Text Recognition"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# CNN backbone (extracts spatial features)
# ---------------------------------------------------------------------------

class CRNNBackbone(nn.Module):
    """
    VGG-style CNN backbone adapted for CRNN.

    Produces feature maps of shape (C, H', W') where:
        H' is fixed to 1 (collapsed by aggressive height pooling)
        W' corresponds to the temporal sequence length

    The width dimension becomes the sequence axis for the RNN.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 512) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 64 filters
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                                  # H/2, W/2

            # Block 2 — 128 filters
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                                  # H/4, W/4

            # Block 3 — 256 filters (no pool yet)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                                        # H/8, W/4

            # Block 4 — 512 filters (pool only height)
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                                        # H/16, W/4

            # Block 5 — collapse remaining height
            nn.Conv2d(512, out_channels, 2, padding=0), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            # H/16 - 1 → should be ~1 for H=32 input
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ---------------------------------------------------------------------------
# Bidirectional LSTM
# ---------------------------------------------------------------------------

class BidirectionalLSTM(nn.Module):
    """
    Two-layer Bidirectional LSTM with projection.

    Args:
        input_size:   Feature dimension per time step.
        hidden_size:  LSTM hidden size (per direction).
        output_size:  Output feature dimension after projection.
        num_layers:   Number of stacked BiLSTM layers.
        dropout:      Dropout between layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, batch, input_size)
        Returns:
            (T, batch, output_size)
        """
        output, _ = self.lstm(x)           # (T, batch, 2×hidden)
        return self.linear(output)          # (T, batch, output_size)


# ---------------------------------------------------------------------------
# Full CRNN model
# ---------------------------------------------------------------------------

class CRNN(nn.Module):
    """
    CRNN for word and sentence recognition using CTC.

    Input:  (batch, 1, H, W) — typically H=32, W=128 for word strips
    Output: (T, batch, num_classes+1) — T is the sequence length (time steps)

    The +1 is for the CTC blank token (index 0 by convention).

    Args:
        in_channels:    Input image channels.
        num_classes:    Number of character classes (excluding CTC blank).
        cnn_out_ch:     CNN backbone output channels.
        rnn_hidden:     BiLSTM hidden size per direction.
        rnn_layers:     Number of BiLSTM layers.
        rnn_dropout:    LSTM dropout rate.
        leaky_relu:     Use LeakyReLU instead of ReLU in backbone.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 47,
        cnn_out_ch: int = 512,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # CNN feature extractor
        self.cnn = CRNNBackbone(in_channels, cnn_out_ch)

        # RNN sequence model
        # After CNN, feature is (B, C, 1, W') → reshape to (W', B, C)
        # CTC output includes blank → num_classes + 1
        self.rnn = BidirectionalLSTM(
            input_size=cnn_out_ch,
            hidden_size=rnn_hidden,
            output_size=num_classes + 1,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape (batch, 1, H, W)

        Returns:
            Log-softmax output of shape (T, batch, num_classes+1)
            for use with nn.CTCLoss.
        """
        # CNN → (batch, C, H', W')
        conv_out = self.cnn(x)

        # Collapse height dimension H'→ should be 1 for H=32 input
        batch, c, h, w = conv_out.shape
        assert h == 1 or h <= 2, (
            f"CNN output height={h} is not 1. Check input size or backbone pooling."
        )
        # Merge height into channels if h > 1
        conv_out = conv_out.view(batch, c * h, w)     # (batch, C*H', W')
        conv_out = conv_out.permute(2, 0, 1)           # (W', batch, C*H')

        # RNN → (T, batch, num_classes+1)
        rnn_out = self.rnn(conv_out)

        return F.log_softmax(rnn_out, dim=2)

    def get_sequence_length(self, input_width: int) -> int:
        """
        Compute the output sequence length T for a given input width W.

        Used for CTC loss setup and inference.

        Args:
            input_width: Width of input image in pixels.

        Returns:
            Expected output sequence length T.
        """
        # The backbone applies W/4 × 2 = W/4 through MaxPool(2,1) × 2
        # Then -1 from the 2×1 conv at the end
        return max(1, (input_width // 4) - 1)

    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        blank_index: int = 0,
    ) -> list[list[int]]:
        """
        Greedy CTC decoding (argmax + collapse blanks and repeats).

        Args:
            log_probs:    (T, batch, num_classes+1) log-probability tensor.
            blank_index:  Index of the CTC blank token.

        Returns:
            List of decoded integer sequences (one per batch item).
        """
        # Argmax over class dimension
        indices = log_probs.argmax(dim=2).permute(1, 0)  # (batch, T)
        results: list[list[int]] = []
        for seq in indices:
            decoded: list[int] = []
            prev = blank_index
            for idx in seq.tolist():
                if idx != blank_index and idx != prev:
                    decoded.append(idx)
                prev = idx
            results.append(decoded)
        return results

    @property
    def name(self) -> str:
        return "crnn"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# CTC collate helper (used in DataLoader)
# ---------------------------------------------------------------------------

def ctc_collate_fn(batch):
    """
    Custom DataLoader collate function for CTC training.

    Pads images to the same width and concatenates target sequences.

    Args:
        batch: List of (image_tensor, target_tensor, target_length) tuples.

    Returns:
        (images, targets, input_lengths, target_lengths)
    """
    images, targets, target_lengths = zip(*batch)

    # Pad images to max width in the batch
    max_w = max(img.shape[-1] for img in images)
    padded = torch.zeros(len(images), images[0].shape[0], images[0].shape[1], max_w)
    for i, img in enumerate(images):
        padded[i, :, :, :img.shape[-1]] = img

    targets_cat = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths = torch.tensor(
        [max_w // 4 - 1] * len(images), dtype=torch.long
    )

    return padded, targets_cat, input_lengths, target_lengths
