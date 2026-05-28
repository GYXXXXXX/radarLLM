#!/usr/bin/env python3
"""Plot ST-Transformer training metrics from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics.jsonl curves as a PNG.")
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics.jsonl.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG path. Defaults to metrics directory / metrics_curves.png.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional figure title.",
    )
    return parser.parse_args()


def read_epoch_records(metrics_path: Path) -> list[dict]:
    records = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "epoch":
                records.append(record)
    if not records:
        raise ValueError(f"No epoch records found in {metrics_path}")
    return records


def has_metric(records: list[dict], key: str) -> bool:
    first = records[0]
    return key in first.get("train", {}) and key in first.get("val", {})


def values(records: list[dict], phase: str, key: str) -> list[float]:
    return [float(record[phase][key]) for record in records]


def higher_is_better(key: str) -> bool:
    return "accuracy" in key or key.endswith("_acc")


def select_best_record(records: list[dict], key: str) -> dict:
    if higher_is_better(key):
        return max(records, key=lambda record: float(record["val"][key]))
    return min(records, key=lambda record: float(record["val"][key]))


def plot_pair(
    ax,
    records: list[dict],
    epochs: list[int],
    key: str,
    title: str,
    ylabel: str,
) -> None:
    train_values = values(records, "train", key)
    val_values = values(records, "val", key)
    best_record = select_best_record(records, key)
    best_epoch = int(best_record["epoch"])
    best_value = float(best_record["val"][key])
    direction = "max" if higher_is_better(key) else "min"

    ax.plot(epochs, train_values, marker="o", markersize=3, linewidth=1.7, label=f"train {key}")
    ax.plot(epochs, val_values, marker="o", markersize=3, linewidth=1.7, label=f"val {key}")
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1)
    ax.scatter([best_epoch], [best_value], color="red", s=28, zorder=3, label=f"best val ({direction}) {best_value:.4f}")
    ax.set_title(f"{title} | best val epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    records = read_epoch_records(metrics_path)
    epochs = [int(record["epoch"]) for record in records]

    panels = [
        ("loss_total", "Total Loss", "loss"),
        ("loss_state", "Trajectory Loss", "loss"),
        ("loss_obj", "Objectness Loss", "loss"),
        ("ade_m", "ADE", "m"),
        ("fde_m", "FDE", "m"),
        ("position_mae_m", "Position MAE", "m"),
        ("velocity_mae_mps", "Velocity MAE", "m/s"),
        ("objectness_slot_accuracy", "Objectness Slot Accuracy", "accuracy"),
    ]
    panels = [panel for panel in panels if has_metric(records, panel[0])]

    if not panels:
        raise ValueError("No supported train/val metric keys were found.")

    ncols = 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.0 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = [axes[0], axes[1]]
    else:
        axes = axes.reshape(-1)

    for ax, (key, title, ylabel) in zip(axes, panels):
        plot_pair(ax, records, epochs, key, title, ylabel)

    for ax in axes[len(panels) :]:
        ax.axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=14)
    else:
        fig.suptitle(metrics_path.parent.name, fontsize=14)

    out_path = Path(args.out) if args.out else metrics_path.parent / "metrics_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
