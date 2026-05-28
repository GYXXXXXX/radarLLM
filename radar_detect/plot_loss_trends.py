import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot loss trends from metrics.jsonl")
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics.jsonl",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="loss_trends.png",
        help="Output image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window",
    )
    return parser.parse_args()


def read_losses(metrics_path: Path):
    epochs = []
    train_total = []
    train_cls = []
    train_state = []
    val_total = []
    val_cls = []
    val_state = []

    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") != "epoch":
                continue
            epoch = record["epoch"]
            train = record["train"]
            val = record["val"]
            epochs.append(epoch)
            train_total.append(train["loss_total"])
            train_cls.append(train["loss_cls"])
            train_state.append(train["loss_state"])
            val_total.append(val["loss_total"])
            val_cls.append(val["loss_cls"])
            val_state.append(val["loss_state"])

    return (
        epochs,
        train_total,
        train_cls,
        train_state,
        val_total,
        val_cls,
        val_state,
    )


def plot_losses(
    epochs,
    train_total,
    train_cls,
    train_state,
    val_total,
    val_cls,
    val_state,
    out_path: Path,
    show: bool,
):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_total, label="train loss_total", linewidth=2)
    plt.plot(epochs, val_total, label="val loss_total", linewidth=2)
    plt.plot(epochs, train_cls, label="train loss_cls", linestyle="--")
    plt.plot(epochs, val_cls, label="val loss_cls", linestyle="--")
    plt.plot(epochs, train_state, label="train loss_state", linestyle=":")
    plt.plot(epochs, val_state, label="val loss_state", linestyle=":")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Trends")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    if show:
        plt.show()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    out_path = Path(args.out)
    (
        epochs,
        train_total,
        train_cls,
        train_state,
        val_total,
        val_cls,
        val_state,
    ) = read_losses(metrics_path)

    if not epochs:
        raise ValueError("No epoch records found in metrics.jsonl")

    plot_losses(
        epochs,
        train_total,
        train_cls,
        train_state,
        val_total,
        val_cls,
        val_state,
        out_path,
        args.show,
    )
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
