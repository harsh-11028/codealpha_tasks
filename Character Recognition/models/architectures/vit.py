"""
Model 5: Vision Transformer (ViT) for character classification.

Implements a from-scratch ViT adapted for small (32×32) grayscale images.

Key adaptations from the original ViT (Dosovitskiy et al. 2021):
  - Smaller patch size (4×4 instead of 16×16) to retain spatial detail
    in small character images
  - Lighter architecture (6 layers, 8 heads, embed_dim=256)
  - Pre-LayerNorm (more stable training for small datasets)
  - Stochastic depth (DropPath) regularization
  - No external pre-trained weights needed (trained from scratch on EMNIST)

Reference:
    Dosovitskiy et al. (2021) — "An Image is Worth 16×16 Words:
    Transformers for Image Recognition at Scale"
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stochastic depth (DropPath)
# ---------------------------------------------------------------------------

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """
    Stochastic depth regularization (DropPath).

    Randomly drops entire residual branches during training.
    More effective than regular Dropout for transformer architectures.

    Args:
        x:         Input tensor.
        drop_prob: Probability of dropping the path.
        training:  Apply only during training.

    Returns:
        Tensor with randomly zero-ed residual branches.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob
    random_tensor.floor_()
    return x / keep_prob * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and linearly embeds them.

    For a 32×32 image with patch_size=4:
        Number of patches = (32/4)² = 64
        Patch feature dim = 4×4×1 = 16 (flattened)
        After linear projection → embed_dim

    Args:
        image_size:  (H, W) input image size (assumes square).
        patch_size:  Size of each patch (assumes square).
        in_channels: Input channels (1 for grayscale).
        embed_dim:   Embedding dimension.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size

        # Conv2d with stride=patch_size is equivalent to a linear patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, H, W)

        Returns:
            (batch, num_patches, embed_dim)
        """
        x = self.proj(x)                        # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)        # (B, num_patches, embed_dim)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Efficient multi-head self-attention with optional relative position bias.

    Args:
        embed_dim:        Input/output embedding dimension.
        num_heads:        Number of attention heads.
        attention_dropout: Dropout on attention weights.
        projection_dropout: Dropout on output projection.
        qkv_bias:         Add bias to Q, K, V projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:                (batch, seq_len, embed_dim)
            return_attention:  If True, also return attention weights.

        Returns:
            (output, attention_weights_or_None)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)              # each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))

        return out, (attn if return_attention else None)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block (more stable than post-LN).

    Structure:
        x = x + DropPath(Attention(LN(x)))
        x = x + DropPath(MLP(LN(x)))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_hidden = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attention_dropout, dropout
        )
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Full Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for character classification.

    Adapted for small (32×32) grayscale character images with:
    - Learnable [CLS] token
    - Learnable positional embeddings
    - Stochastic depth regularization
    - Attention rollout for visualization

    Args:
        image_size:       Input image size (assumes square).
        patch_size:       Patch size (assumes square).
        in_channels:      Number of input channels.
        num_classes:      Output class count.
        embed_dim:        Token embedding dimension.
        num_heads:        Number of attention heads.
        num_layers:       Number of transformer blocks.
        mlp_ratio:        MLP expansion ratio.
        dropout_rate:     Dropout in MLP and projections.
        attention_dropout: Dropout on attention weights.
        drop_path_rate:   Max stochastic depth rate (linearly decays).
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 47,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout_rate)

        # Stochastic depth decay rule: linearly increase from 0 to drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout_rate,
                attention_dropout=attention_dropout,
                drop_path_rate=dpr[i],
            )
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        # Positional embedding: standard deviation 0.02
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:                (batch, channels, H, W)
            return_attention:  Ignored; use get_attention_maps() instead.

        Returns:
            Class logits of shape (batch, num_classes).
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)                              # (B, N, embed_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                       # (B, N+1, embed_dim)
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]                                    # [CLS] token output
        return self.head(cls_out)

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract attention weight maps from all layers.

        Used for visualization (attention rollout, head analysis).

        Args:
            x: Input tensor (batch, channels, H, W).

        Returns:
            List of attention weight tensors, one per layer.
            Each tensor has shape (batch, num_heads, N+1, N+1).
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        attention_maps: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, return_attention=True)
            if attn is not None:
                attention_maps.append(attn)

        return attention_maps

    def attention_rollout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention rollout for saliency visualization.

        Multiplies attention maps across layers to show which
        input patches the model focused on for its prediction.

        Args:
            x: Input tensor (1, channels, H, W) — single image.

        Returns:
            2-D saliency map of shape (num_patches,) — one value per patch.
        """
        attention_maps = self.get_attention_maps(x)  # List[(1, heads, N+1, N+1)]

        # Mean over heads
        rollout = torch.eye(attention_maps[0].shape[-1], device=x.device)
        for attn in attention_maps:
            attn_avg = attn.mean(dim=1)[0]  # (N+1, N+1)
            # Add residual identity (attention + identity) / 2
            attn_avg = attn_avg + torch.eye(attn_avg.shape[0], device=x.device)
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            rollout = attn_avg @ rollout

        # CLS token row (index 0) → attention to all patch tokens (indices 1:)
        return rollout[0, 1:]

    @property
    def name(self) -> str:
        return "vit"

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
