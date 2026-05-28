#!/usr/bin/env python3
"""Train and validate the raw-IQ FMCW baseline, logging metrics as JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .config import STATE_MEAN, STATE_STD
    from .dataset import FmcwTrajectoryDataset
    from .model import FmcwBaseline3DCNN
    from .utils import append_jsonl, collect_scene_files, set_seed
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import STATE_MEAN, STATE_STD
    from dataset import FmcwTrajectoryDataset
    from model import FmcwBaseline3DCNN
    from utils import append_jsonl, collect_scene_files, set_seed


class SceneBatchSampler(Sampler[list[int]]):
    """Shuffle scene order while keeping each scene's windows adjacent."""

    def __init__(
        self,
        scene_index_groups: list[list[int]],
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.scene_index_groups = [list(indices) for indices in scene_index_groups if indices]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.sample_count = sum(len(indices) for indices in self.scene_index_groups)

    def __iter__(self):
        order = list(range(len(self.scene_index_groups)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(order)
        self.epoch += 1

        batch: list[int] = []
        for group_idx in order:
            for sample_idx in self.scene_index_groups[group_idx]:
                batch.append(sample_idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return (self.sample_count + self.batch_size - 1) // self.batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train raw-IQ FMCW trajectory baseline.")
    parser.add_argument("--dataset-dir", type=str, default="radar_detect/fmcw_traj_dataset")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--tin", type=int, default=16)
    parser.add_argument("--tout", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--iq-scale", type=float, default=76.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--max-cache-scenes",
        type=int,
        default=32,
        help="Maximum number of loaded scenes to keep in the per-process LRU cache; 0 disables caching.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=1,
        help="Number of batches prefetched per DataLoader worker when num_workers > 0.",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned CPU memory for CUDA transfers.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print running progress every N batches; 0 disables progress output.",
    )
    parser.add_argument(
        "--window-shuffle",
        action="store_true",
        help="Shuffle individual windows instead of scene-local batches. This is slower for .mat/HDF5 datasets.",
    )
    parser.add_argument("--no-cache-scenes", action="store_true")
    parser.add_argument("--print-sample", action="store_true")
    args = parser.parse_args()
    if args.max_cache_scenes < 0:
        parser.error("--max-cache-scenes must be >= 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be >= 1")
    if args.log_every < 0:
        parser.error("--log-every must be >= 0")
    if args.label_smoothing < 0 or args.label_smoothing >= 1:
        parser.error("--label-smoothing must be in [0, 1)")
    if args.dropout < 0 or args.dropout >= 1:
        parser.error("--dropout must be in [0, 1)")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be >= 0")
    if args.early_stop_min_delta < 0:
        parser.error("--early-stop-min-delta must be >= 0")
    if args.lr_patience < 0:
        parser.error("--lr-patience must be >= 0")
    if args.lr_factor <= 0 or args.lr_factor >= 1:
        parser.error("--lr-factor must be in (0, 1)")
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS Conv3d/normalization kernels can be numerically unstable across
    # PyTorch/macOS versions for this workload. Use --device mps explicitly
    # only after verifying finite losses on your local stack.
    return torch.device("cpu")


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("radar_detect/train_baseline/runs") / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def split_scenes(scene_files: list[Path], train_ratio: float) -> tuple[list[Path], list[Path]]:
    split = int(len(scene_files) * train_ratio)
    split = min(max(split, 1), len(scene_files) - 1)
    return scene_files[:split], scene_files[split:]


def make_loader(
    scene_files: list[Path],
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = FmcwTrajectoryDataset(
        scene_files,
        tin=args.tin,
        tout=args.tout,
        stride=args.stride,
        max_targets=args.max_targets,
        iq_scale=args.iq_scale,
        cache_scenes=not args.no_cache_scenes,
        max_cache_scenes=args.max_cache_scenes,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available() and not args.no_pin_memory,
    }
    if shuffle and not args.window_shuffle:
        loader_kwargs["batch_sampler"] = SceneBatchSampler(
            dataset.scene_index_groups,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
    else:
        loader_kwargs["batch_size"] = args.batch_size
        loader_kwargs["shuffle"] = shuffle
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "x": batch["x"].to(device, non_blocking=True).float(),
        "class_label": batch["class_label"].to(device, non_blocking=True).long(),
        "state_label": batch["state_label"].to(device, non_blocking=True).float(),
        "target_mask": batch["target_mask"].to(device, non_blocking=True).float(),
    }


def compute_loss(
    class_logits: torch.Tensor,
    state_pred: torch.Tensor,
    class_label: torch.Tensor,
    state_label: torch.Tensor,
    target_mask: torch.Tensor,
    lambda_state: float,
    label_smoothing: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz, max_targets, num_classes = class_logits.shape
    loss_cls = F.cross_entropy(
        class_logits.reshape(bsz * max_targets, num_classes),
        class_label.reshape(bsz * max_targets),
        label_smoothing=label_smoothing,
    )

    raw_state_loss = F.smooth_l1_loss(state_pred, state_label, reduction="none")
    mask = target_mask[:, :, None, None]
    valid_count = (target_mask.sum() * state_pred.shape[2] * state_pred.shape[3]).clamp_min(1.0)
    loss_state = (raw_state_loss * mask).sum() / valid_count
    loss_total = loss_cls + lambda_state * loss_state
    return loss_total, {"loss_cls": loss_cls, "loss_state": loss_state}


def denormalize_state(state: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(STATE_MEAN, dtype=state.dtype, device=state.device).view(1, 1, 1, 4)
    std = torch.as_tensor(STATE_STD, dtype=state.dtype, device=state.device).view(1, 1, 1, 4)
    return state * std + mean


def batch_metrics(
    class_logits: torch.Tensor,
    state_pred: torch.Tensor,
    class_label: torch.Tensor,
    state_label: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        pred_class = class_logits.argmax(dim=-1)
        slot_correct = (pred_class == class_label).float()
        slot_acc_sum = slot_correct.sum().item()
        slot_count = float(class_label.numel())

        target_count = target_mask.sum().item()
        target_acc_sum = (slot_correct * target_mask).sum().item()

        pred_denorm = denormalize_state(state_pred)
        label_denorm = denormalize_state(state_label)
        diff = pred_denorm - label_denorm
        mask = target_mask[:, :, None, None]

        pos_abs_sum = (diff[..., :2].abs() * mask).sum().item()
        vel_abs_sum = (diff[..., 2:].abs() * mask).sum().item()
        pos_count = target_count * state_pred.shape[2] * 2
        vel_count = target_count * state_pred.shape[2] * 2

        pos_l2 = torch.sqrt(torch.square(diff[..., :2]).sum(dim=-1).clamp_min(0.0))
        frame_mask = target_mask[:, :, None]
        ade_sum = (pos_l2 * frame_mask).sum().item()
        ade_count = target_count * state_pred.shape[2]
        fde_sum = (pos_l2[:, :, -1] * target_mask).sum().item()

    return {
        "slot_acc_sum": slot_acc_sum,
        "slot_count": slot_count,
        "target_acc_sum": target_acc_sum,
        "target_count": max(target_count, 0.0),
        "pos_abs_sum": pos_abs_sum,
        "pos_count": max(pos_count, 0.0),
        "vel_abs_sum": vel_abs_sum,
        "vel_count": max(vel_count, 0.0),
        "ade_sum": ade_sum,
        "ade_count": max(ade_count, 0.0),
        "fde_sum": fde_sum,
        "fde_count": max(target_count, 0.0),
    }


def empty_accumulator() -> dict[str, float]:
    return {
        "loss_total_sum": 0.0,
        "loss_cls_sum": 0.0,
        "loss_state_sum": 0.0,
        "batch_count": 0.0,
        "sample_count": 0.0,
        "slot_acc_sum": 0.0,
        "slot_count": 0.0,
        "target_acc_sum": 0.0,
        "target_count": 0.0,
        "pos_abs_sum": 0.0,
        "pos_count": 0.0,
        "vel_abs_sum": 0.0,
        "vel_count": 0.0,
        "ade_sum": 0.0,
        "ade_count": 0.0,
        "fde_sum": 0.0,
        "fde_count": 0.0,
    }


def add_batch_stats(
    acc: dict[str, float],
    loss_total: torch.Tensor,
    loss_parts: dict[str, torch.Tensor],
    metric_parts: dict[str, float],
    batch_size: int,
) -> None:
    acc["loss_total_sum"] += float(loss_total.detach().cpu()) * batch_size
    acc["loss_cls_sum"] += float(loss_parts["loss_cls"].detach().cpu()) * batch_size
    acc["loss_state_sum"] += float(loss_parts["loss_state"].detach().cpu()) * batch_size
    acc["batch_count"] += 1.0
    acc["sample_count"] += float(batch_size)
    for key, value in metric_parts.items():
        acc[key] += float(value)


def finalize_metrics(acc: dict[str, float]) -> dict[str, float]:
    samples = max(acc["sample_count"], 1.0)

    def ratio(num_key: str, den_key: str) -> float:
        denom = acc[den_key]
        return float(acc[num_key] / denom) if denom > 0 else 0.0

    return {
        "loss_total": acc["loss_total_sum"] / samples,
        "loss_cls": acc["loss_cls_sum"] / samples,
        "loss_state": acc["loss_state_sum"] / samples,
        "slot_accuracy": ratio("slot_acc_sum", "slot_count"),
        "target_accuracy": ratio("target_acc_sum", "target_count"),
        "position_mae_m": ratio("pos_abs_sum", "pos_count"),
        "velocity_mae_mps": ratio("vel_abs_sum", "vel_count"),
        "ade_m": ratio("ade_sum", "ade_count"),
        "fde_m": ratio("fde_sum", "fde_count"),
    }


def print_progress(
    phase: str,
    epoch: int,
    batch_index: int,
    total_batches: int,
    acc: dict[str, float],
) -> None:
    running_metrics = finalize_metrics(acc)
    print(
        f"phase={phase} "
        f"epoch={epoch} "
        f"batch={batch_index} "
        f"total_batch={total_batches} "
        f"loss_total={running_metrics['loss_total']:.6f} "
        f"loss_cls={running_metrics['loss_cls']:.6f} "
        f"loss_state={running_metrics['loss_state']:.6f}",
        flush=True,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_state: float,
    label_smoothing: float,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: str,
    log_every: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    acc = empty_accumulator()
    total_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            class_logits, state_pred = model(batch["x"])
            loss_total, loss_parts = compute_loss(
                class_logits,
                state_pred,
                batch["class_label"],
                batch["state_label"],
                batch["target_mask"],
                lambda_state,
                label_smoothing,
            )
            if not torch.isfinite(loss_total):
                raise FloatingPointError(
                    "Non-finite loss detected. Try --device cpu, lower --lr, "
                    "or inspect input/label normalization."
                )
            if is_train:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()

        metrics = batch_metrics(
            class_logits.detach(),
            state_pred.detach(),
            batch["class_label"],
            batch["state_label"],
            batch["target_mask"],
        )
        add_batch_stats(acc, loss_total, loss_parts, metrics, batch_size=batch["x"].shape[0])

        if log_every and (batch_index % log_every == 0 or batch_index == total_batches):
            print_progress(
                phase,
                epoch,
                batch_index,
                total_batches,
                acc,
            )

    return finalize_metrics(acc)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    val_metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
            "state_mean": STATE_MEAN,
            "state_std": STATE_STD,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = make_run_dir(args)
    log_path = run_dir / "metrics.jsonl"

    scene_files = collect_scene_files(args.dataset_dir)
    train_scenes, val_scenes = split_scenes(scene_files, args.train_ratio)
    train_loader = make_loader(train_scenes, args, shuffle=True)
    val_loader = make_loader(val_scenes, args, shuffle=False)

    model = FmcwBaseline3DCNN(
        in_channels=16,
        max_targets=args.max_targets,
        num_classes=args.num_classes,
        tout=args.tout,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    config_record = {
        "type": "config",
        "args": vars(args),
        "device": str(device),
        "run_dir": str(run_dir),
        "scene_count": len(scene_files),
        "train_scene_count": len(train_scenes),
        "val_scene_count": len(val_scenes),
        "train_window_count": len(train_loader.dataset),
        "val_window_count": len(val_loader.dataset),
        "state_mean": STATE_MEAN.tolist(),
        "state_std": STATE_STD.tolist(),
    }
    append_jsonl(log_path, config_record)

    if args.print_sample:
        sample = train_loader.dataset[0]
        print("sample x shape:", tuple(sample["x"].shape))
        print("sample class_label:", sample["class_label"].tolist())
        print("sample state_label shape:", tuple(sample["state_label"].shape))
        print("sample target_mask:", sample["target_mask"].tolist())

    print(json.dumps(config_record, ensure_ascii=False, indent=2))
    best_val = float("inf")
    best_epoch = 0
    epochs_since_best = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                args.lambda_state,
                args.label_smoothing,
                optimizer,
                epoch=epoch,
                phase="train",
                log_every=args.log_every,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                args.lambda_state,
                args.label_smoothing,
                optimizer=None,
                epoch=epoch,
                phase="val",
                log_every=args.log_every,
            )

            record = {
                "type": "epoch",
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            append_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=False, allow_nan=False))

            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, args, val_metrics)
            current_val = val_metrics["loss_total"]
            scheduler.step(current_val)

            if current_val < best_val - args.early_stop_min_delta:
                best_val = current_val
                best_epoch = epoch
                epochs_since_best = 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, args, val_metrics)
            else:
                epochs_since_best += 1

            if args.early_stop_patience and epochs_since_best >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val {best_val:.6f}).",
                    flush=True,
                )
                break

        append_jsonl(
            log_path,
            {"type": "done", "best_val_loss": best_val, "best_epoch": best_epoch},
        )
    except Exception as exc:
        append_jsonl(log_path, {"type": "error", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
