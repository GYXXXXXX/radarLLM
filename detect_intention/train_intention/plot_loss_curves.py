#!/usr/bin/env python3
"""Plot train/validation loss curves from metrics.jsonl.

Usage:
    python detect_intention/train_intention/plot_loss_curves.py path/to/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LOSS_KEYS = ("loss_total", "loss_intent", "loss_threat", "loss_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss curves from a training metrics.jsonl file.")
    parser.add_argument("metrics", type=str, help="Path to metrics.jsonl.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output image path. Default: <metrics_dir>/loss_curves.png",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_metrics(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    epochs: list[dict[str, Any]] = []
    done: dict[str, Any] | None = None

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if record.get("type") == "epoch":
                epochs.append(record)
            elif record.get("type") == "done":
                done = record

    if not epochs:
        raise ValueError(f"No epoch records found in {path}")
    return epochs, done


def series(epochs: list[dict[str, Any]], phase: str, key: str) -> list[float]:
    values = []
    for record in epochs:
        try:
            values.append(float(record[phase][key]))
        except KeyError as exc:
            raise KeyError(f"Missing {phase}.{key} in epoch {record.get('epoch')}") from exc
    return values


def best_epoch_by_val(epochs: list[dict[str, Any]], key: str) -> tuple[int, float]:
    best = min(epochs, key=lambda item: float(item["val"][key]))
    return int(best["epoch"]), float(best["val"][key])


def style_axes(ax: Any, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_losses(epochs: list[dict[str, Any]], done: dict[str, Any] | None, output: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    epoch_ids = [int(item["epoch"]) for item in epochs]
    best_intent_epoch, best_intent_loss = best_epoch_by_val(epochs, "loss_intent")
    best_total_epoch, best_total_loss = best_epoch_by_val(epochs, "loss_total")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)
    axes_flat = axes.flatten()

    colors = {
        "train": "#2563eb",
        "val": "#dc2626",
        "best": "#111827",
    }
    titles = {
        "loss_total": "Total Loss",
        "loss_intent": "Intent Classification Loss",
        "loss_threat": "Threat Classification Loss",
        "loss_state": "Trajectory Regression Loss",
    }

    for ax, key in zip(axes_flat, LOSS_KEYS):
        train_values = series(epochs, "train", key)
        val_values = series(epochs, "val", key)
        ax.plot(epoch_ids, train_values, color=colors["train"], linewidth=2.0, label="train")
        ax.plot(epoch_ids, val_values, color=colors["val"], linewidth=2.0, label="val")

        best_epoch, best_value = best_epoch_by_val(epochs, key)
        ax.scatter([best_epoch], [best_value], color=colors["best"], s=34, zorder=5)
        ax.axvline(best_epoch, color=colors["best"], linestyle=":", linewidth=1.1, alpha=0.55)
        ax.annotate(
            f"best val\nE{best_epoch}: {best_value:.4f}",
            xy=(best_epoch, best_value),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
            color=colors["best"],
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d1d5db", "alpha": 0.9},
        )
        style_axes(ax, titles[key], key)
        ax.legend(frameon=False, fontsize=9)

    subtitle = (
        f"best val loss_intent: epoch {best_intent_epoch}, {best_intent_loss:.6f}    "
        f"best val loss_total: epoch {best_total_epoch}, {best_total_loss:.6f}"
    )
    if done and "best_epoch" in done:
        subtitle += f"    early-stop best_epoch: {done['best_epoch']}"

    fig.suptitle("Training Loss Curves", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.955, subtitle, ha="center", va="center", fontsize=10, color="#374151")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_summary(epochs: list[dict[str, Any]], done: dict[str, Any] | None, output: Path) -> None:
    latest = epochs[-1]
    best_intent_epoch, best_intent_loss = best_epoch_by_val(epochs, "loss_intent")
    best_total_epoch, best_total_loss = best_epoch_by_val(epochs, "loss_total")
    best_state_epoch, best_state_loss = best_epoch_by_val(epochs, "loss_state")

    print(f"Loaded epochs: {len(epochs)}")
    print(f"Latest epoch: {latest['epoch']}")
    print(f"Best val loss_intent: epoch {best_intent_epoch}, {best_intent_loss:.6f}")
    print(f"Best val loss_total : epoch {best_total_epoch}, {best_total_loss:.6f}")
    print(f"Best val loss_state : epoch {best_state_epoch}, {best_state_loss:.6f}")
    if done:
        print(f"Done record: {json.dumps(done, ensure_ascii=False)}")
    print(f"Saved figure: {output}")


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    output = Path(args.output) if args.output else metrics_path.with_name("loss_curves.png")
    epochs, done = load_metrics(metrics_path)
    plot_losses(epochs, done, output, dpi=args.dpi)
    print_summary(epochs, done, output)


if __name__ == "__main__":
    main()

