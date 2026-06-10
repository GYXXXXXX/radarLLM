"""Patch spatial-temporal Transformer for target intention prediction."""

from __future__ import annotations

import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlotCrossAttention(nn.Module):
    """Learn fixed target-slot queries from temporal scene tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.post_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        query = query + attended
        query = query + self.mlp(self.post_norm(query))
        return query


class FmcwIntentTransformer(nn.Module):
    """Scene-window model with trajectory, class, intent, and threat heads."""

    def __init__(
        self,
        in_channels: int = 16,
        max_targets: int = 4,
        tin: int = 32,
        tout: int = 16,
        nchirp: int = 32,
        nfast: int = 128,
        patch_chirp: int = 8,
        patch_fast: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        spatial_layers: int = 1,
        temporal_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        num_target_classes: int = 4,
        num_intents: int = 5,
        num_threats: int = 4,
    ) -> None:
        super().__init__()
        if nchirp % patch_chirp != 0:
            raise ValueError("nchirp must be divisible by patch_chirp")
        if nfast % patch_fast != 0:
            raise ValueError("nfast must be divisible by patch_fast")

        self.max_targets = int(max_targets)
        self.tin = int(tin)
        self.tout = int(tout)
        self.embed_dim = int(embed_dim)

        self.patch_embed = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(1, patch_chirp, patch_fast),
            stride=(1, patch_chirp, patch_fast),
            bias=True,
        )

        spatial_tokens = (nchirp // patch_chirp) * (nfast // patch_fast)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, spatial_tokens, embed_dim))
        self.temporal_pos_for_spatial = nn.Parameter(torch.zeros(1, tin, 1, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, tin, embed_dim))

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)

        self.slot_queries = nn.Parameter(torch.randn(1, max_targets, embed_dim) * 0.02)
        self.slot_decoder = SlotCrossAttention(embed_dim, num_heads, mlp_ratio, dropout)
        self.head_norm = nn.LayerNorm(embed_dim)

        self.objectness_head = nn.Linear(embed_dim, 1)
        self.state_head = nn.Linear(embed_dim, tout * 4)
        self.target_class_head = nn.Linear(embed_dim, num_target_classes)
        self.intent_head = nn.Linear(embed_dim, num_intents)
        self.threat_head = nn.Linear(embed_dim, num_threats)

        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_for_spatial, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(x.shape)}")

        patches = self.patch_embed(x)
        bsz, dim, tin, hpatch, wpatch = patches.shape
        if tin != self.tin:
            raise ValueError(f"Expected Tin={self.tin}, got {tin}")

        spatial_tokens = hpatch * wpatch
        tokens = patches.permute(0, 2, 3, 4, 1).reshape(bsz, tin, spatial_tokens, dim)
        tokens = tokens + self.spatial_pos[:, :, :spatial_tokens] + self.temporal_pos_for_spatial[:, :tin]

        spatial_in = tokens.reshape(bsz * tin, spatial_tokens, dim)
        spatial_out = self.spatial_encoder(spatial_in)
        frame_tokens = spatial_out.mean(dim=1).reshape(bsz, tin, dim)

        temporal_tokens = self.temporal_encoder(frame_tokens + self.temporal_pos[:, :tin])
        slot_queries = self.slot_queries.expand(bsz, -1, -1)
        slot_features = self.slot_decoder(slot_queries, temporal_tokens)
        slot_features = self.head_norm(slot_features)

        return {
            "objectness_logits": self.objectness_head(slot_features).squeeze(-1),
            "state_pred": self.state_head(slot_features).view(bsz, self.max_targets, self.tout, 4),
            "target_class_logits": self.target_class_head(slot_features),
            "intent_logits": self.intent_head(slot_features),
            "threat_logits": self.threat_head(slot_features),
            "slot_features": slot_features,
        }


class TrackIntentTransformer(nn.Module):
    """Target-track Transformer for intention prediction from radar monitoring results.

    Input shape:
        [B, K, Tin, 4] normalized [x, y, vx, vy]
    """

    def __init__(
        self,
        max_targets: int = 4,
        tin: int = 24,
        tout: int = 8,
        embed_dim: int = 128,
        num_heads: int = 4,
        temporal_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        num_target_classes: int = 4,
        num_intents: int = 5,
        num_threats: int = 4,
    ) -> None:
        super().__init__()
        self.max_targets = int(max_targets)
        self.tin = int(tin)
        self.tout = int(tout)
        self.embed_dim = int(embed_dim)

        self.state_embed = nn.Linear(4, embed_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, 1, tin, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
        self.head_norm = nn.LayerNorm(embed_dim)

        self.objectness_head = nn.Linear(embed_dim, 1)
        self.state_head = nn.Linear(embed_dim, tout * 4)
        self.target_class_head = nn.Linear(embed_dim, num_target_classes)
        self.intent_head = nn.Linear(embed_dim, num_intents)
        self.threat_head = nn.Linear(embed_dim, num_threats)

        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(self, state_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if state_input.ndim != 4:
            raise ValueError(f"Expected [B, K, T, 4], got {tuple(state_input.shape)}")

        bsz, max_targets, tin, dim = state_input.shape
        if max_targets != self.max_targets:
            raise ValueError(f"Expected K={self.max_targets}, got {max_targets}")
        if tin != self.tin:
            raise ValueError(f"Expected Tin={self.tin}, got {tin}")
        if dim != 4:
            raise ValueError(f"Expected state dim=4, got {dim}")

        tokens = self.state_embed(state_input) + self.temporal_pos[:, :, :tin]
        encoded = self.temporal_encoder(tokens.reshape(bsz * max_targets, tin, self.embed_dim))
        features = encoded[:, -1].reshape(bsz, max_targets, self.embed_dim)
        features = self.head_norm(features)

        return {
            "objectness_logits": self.objectness_head(features).squeeze(-1),
            "state_pred": self.state_head(features).view(bsz, self.max_targets, self.tout, 4),
            "target_class_logits": self.target_class_head(features),
            "intent_logits": self.intent_head(features),
            "threat_logits": self.threat_head(features),
            "slot_features": features,
        }
