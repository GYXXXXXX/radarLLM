"""Default constants for the FMCW trajectory baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


STATE_MEAN = np.array(
    [45.6298792, 0.0279730521, 6.7274895, -0.0123089929],
    dtype=np.float32,
)
STATE_STD = np.array(
    [13.8801432, 9.69236536, 3.50705298, 2.17320712],
    dtype=np.float32,
)


@dataclass(frozen=True)
class BaselineConfig:
    dataset_dir: Path = Path("radar_detect/fmcw_traj_dataset")
    tin: int = 16
    tout: int = 8
    stride: int = 1
    max_targets: int = 4
    num_classes: int = 5
    iq_scale: float = 76.0
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    lambda_state: float = 1.0
    train_ratio: float = 0.8
    num_workers: int = 0
    seed: int = 2026
