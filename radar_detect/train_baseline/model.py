"""3D CNN baseline model for raw-IQ FMCW windows."""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FmcwBaseline3DCNN(nn.Module):
    """Fixed-slot classifier/regressor for real targets."""

    def __init__(
        self,
        in_channels: int = 16,
        max_targets: int = 4,
        num_classes: int = 5,
        tout: int = 8,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_targets = int(max_targets)
        self.num_classes = int(num_classes)
        self.tout = int(tout)

        self.encoder = nn.Sequential(
            ConvBlock3d(in_channels, 32, stride=(1, 2, 2)),
            ConvBlock3d(32, 64, stride=(2, 2, 2)),
            ConvBlock3d(64, 128, stride=(2, 2, 2)),
            ConvBlock3d(128, 192, stride=(2, 2, 2)),
            ConvBlock3d(192, embedding_dim, stride=(1, 2, 2)),
            nn.AdaptiveAvgPool3d(1),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )

        self.class_head = nn.Linear(embedding_dim, self.max_targets * self.num_classes)
        self.state_head = nn.Linear(embedding_dim, self.max_targets * self.tout * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(self.encoder(x))
        class_logits = self.class_head(features).view(
            x.shape[0],
            self.max_targets,
            self.num_classes,
        )
        state_pred = self.state_head(features).view(
            x.shape[0],
            self.max_targets,
            self.tout,
            4,
        )
        return class_logits, state_pred

