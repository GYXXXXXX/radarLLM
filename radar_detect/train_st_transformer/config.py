"""Constants for the scene-first ST-Transformer trajectory experiment."""

from __future__ import annotations

import numpy as np


STATE_MEAN = np.array(
    [46.0754167, 0.00384204064, 6.73434629, 0.0188028894],
    dtype=np.float32,
)
STATE_STD = np.array(
    [13.9185145, 9.59128023, 3.51218592, 2.15311442],
    dtype=np.float32,
)

