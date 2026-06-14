"""Vision Transformer encoder (PyTorch)."""

import torch
import torch.nn as nn

from vit_latex.config import (PATCH_SIZE, NUM_PATCHES, EMBEDDING_DIM,
                              NUM_HEADS, TRANSFORMER_UNITS, TRANSFORMER_LAYERS,
                              DROPOUT_RATE)


class Patches(nn.Module):
    """Split images into patches.

    Input (B, C, H, W) -> output (B, num_patches, patch_size*patch_size*C).
    """

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        batch_size, channels, height, width = images.shape
        p = self.patch_size
        num_patches_h = height // p
        num_patches_w = width // p

        patches = images.unfold(2, p, p).unfold(3, p, p)  # (B, C, nh, nw, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(batch_size, num_patches_h * num_patches_w, channels * p * p)


class PatchEncoder(nn.Module):
    """Linear projection + learnable position embedding."""

    def __init__(self, num_patches, projection_dim, patch_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patch):
        positions = torch.arange(self.num_patches, device=patch.device)
        return self.projection(patch) + self.position_embedding(positions)


class MLP(nn.Module):
    """Multilayer perceptron: Linear + GELU + Dropout per layer."""

    def __init__(self, in_dim, hidden_units, dropout_rate):
        super().__init__()
        layers = []
        dim = in_dim
        for units in hidden_units:
            layers += [nn.Linear(dim, units), nn.GELU(), nn.Dropout(dropout_rate)]
            dim = units
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LN -> MHA -> residual; LN -> MLP -> residual."""

    def __init__(self, dim, num_heads, hidden_units, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, hidden_units, dropout_rate)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_output, _ = self.attention(x1, x1, x1, need_weights=False)
        x2 = x + attn_output
        x3 = self.mlp(self.norm2(x2))
        return x2 + x3


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer encoder producing a (B, num_patches, EMBEDDING_DIM) feature sequence."""

    def __init__(self, input_shape):
        super().__init__()
        height, width, channels = input_shape
        patch_dim = PATCH_SIZE * PATCH_SIZE * channels

        self.patches = Patches(PATCH_SIZE)
        self.patch_encoder = PatchEncoder(NUM_PATCHES, EMBEDDING_DIM, patch_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(EMBEDDING_DIM, NUM_HEADS, TRANSFORMER_UNITS, DROPOUT_RATE)
            for _ in range(TRANSFORMER_LAYERS)
        ])
        self.norm = nn.LayerNorm(EMBEDDING_DIM, eps=1e-6)

    def forward(self, images):
        x = self.patch_encoder(self.patches(images))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


def vision_transformer_encoder(input_shape):
    """Build a Vision Transformer encoder."""
    return VisionTransformerEncoder(input_shape)
