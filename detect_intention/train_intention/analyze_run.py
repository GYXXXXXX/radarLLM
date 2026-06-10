#!/usr/bin/env python3
"""Summarize a training metrics.jsonl file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze intention training metrics.")
    parser.add_argument("metrics", type=str)
    return parser.parse_args()


def load_epochs(path: str | Path) -> list[dict[str, Any]]:
    epochs = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "epoch":
                epochs.append(record)
    return epochs


def best_by(epochs: list[dict[str, Any]], key: str, larger: bool = False) -> dict[str, Any]:
    return max(epochs, key=lambda item: item["val"][key]) if larger else min(epochs, key=lambda item: item["val"][key])


def main() -> None:
    args = parse_args()
    epochs = load_epochs(args.metrics)
    if not epochs:
        raise SystemExit("No epoch records found.")

    latest = epochs[-1]
    best_total = best_by(epochs, "loss_total")
    best_intent_loss = best_by(epochs, "loss_intent")
    best_intent_acc = best_by(epochs, "intent_accuracy", larger=True)
    best_ade = best_by(epochs, "ade_m")

    summary = {
        "epoch_count": len(epochs),
        "latest": {
            "epoch": latest["epoch"],
            "train_intent_accuracy": latest["train"]["intent_accuracy"],
            "val_intent_accuracy": latest["val"]["intent_accuracy"],
            "train_threat_accuracy": latest["train"]["threat_accuracy"],
            "val_threat_accuracy": latest["val"]["threat_accuracy"],
            "train_ade_m": latest["train"]["ade_m"],
            "val_ade_m": latest["val"]["ade_m"],
            "val_loss_total": latest["val"]["loss_total"],
        },
        "best_val_loss_total": {
            "epoch": best_total["epoch"],
            "value": best_total["val"]["loss_total"],
        },
        "best_val_intent_loss": {
            "epoch": best_intent_loss["epoch"],
            "value": best_intent_loss["val"]["loss_intent"],
        },
        "best_val_intent_accuracy": {
            "epoch": best_intent_acc["epoch"],
            "value": best_intent_acc["val"]["intent_accuracy"],
        },
        "best_val_ade_m": {
            "epoch": best_ade["epoch"],
            "value": best_ade["val"]["ade_m"],
        },
        "generalization_gap_latest": {
            "intent_accuracy_train_minus_val": (
                latest["train"]["intent_accuracy"] - latest["val"]["intent_accuracy"]
            ),
            "threat_accuracy_train_minus_val": (
                latest["train"]["threat_accuracy"] - latest["val"]["threat_accuracy"]
            ),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
