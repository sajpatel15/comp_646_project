"""Model definitions for learned baselines and fusion models."""

from __future__ import annotations

import torch
from torch import nn

from .labels import CLASS_ORDER


class LinearProbe(nn.Module):
    """A simple linear probe over frozen CLIP features."""

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


class MLPProbe(nn.Module):
    """A small MLP ablation for frozen CLIP features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1, num_classes: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CrossAttentionFusionClassifier(nn.Module):
    """Cross-attention over frozen CLIP token states."""

    def __init__(
        self,
        image_input_dim: int,
        text_input_dim: int | None = None,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        resolved_text_dim = text_input_dim if text_input_dim is not None else image_input_dim
        self.image_proj = nn.Linear(image_input_dim, hidden_dim)
        self.text_proj = nn.Linear(resolved_text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_proj(image_tokens)
        text_tokens = self.text_proj(text_tokens)
        attended, _ = self.attention(query=text_tokens, key=image_tokens, value=image_tokens)
        pooled = self.norm(attended + text_tokens).mean(dim=1)
        return self.classifier(pooled)
