"""Default constants for the FMCW trajectory baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


STATE_MEAN = np.array(
    [46.0754167, 0.00384204064, 6.73434629, 0.0188028894],
    dtype=np.float32,
)
STATE_STD = np.array(
    [13.9185145, 9.59128023, 3.51218592, 2.15311442],
    dtype=np.float32,
)


@dataclass(frozen=True)
class BaselineConfig:
    dataset_dir: Path = Path("radar_detect/fmcw_traj_dataset_2000scenes")
    tin: int = 16
    tout: int = 8
    stride: int = 1
    max_targets: int = 4
    num_classes: int = 5
    iq_scale: float = 76.0
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    lambda_state: float = 1.0
    train_ratio: float = 0.8
    num_workers: int = 0
    max_cache_scenes: int = 32
    prefetch_factor: int = 1
    pin_memory: bool = True
    log_every: int = 10
    window_shuffle: bool = False
    seed: int = 2026
