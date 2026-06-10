"""Constants for target-intention prediction experiments."""

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

TARGET_CLASS_NAMES = {
    1: "T1_slow_smooth",
    2: "T2_agile",
    3: "T3_small_slow",
    4: "T4_fast_maneuver",
}

INTENT_NAMES = {
    1: "benign_transit",
    2: "approach",
    3: "retreat",
    4: "loiter_patrol",
    5: "intercept",
}

THREAT_NAMES = {
    1: "low",
    2: "guarded",
    3: "elevated",
    4: "high",
}

ACTION_BY_INTENT = {
    "benign_transit": "monitor",
    "approach": "increase_tracking_rate",
    "retreat": "monitor",
    "loiter_patrol": "classify_and_shadow",
    "intercept": "alert_and_allocate_tracker",
}

NUM_TARGET_CLASSES = len(TARGET_CLASS_NAMES)
NUM_INTENTS = len(INTENT_NAMES)
NUM_THREATS = len(THREAT_NAMES)

