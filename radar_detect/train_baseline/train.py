#!/usr/bin/env python3
"""Train and validate the raw-IQ FMCW baseline, logging metrics as JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-cache-scenes", action="store_true")
    parser.add_argument("--print-sample", action="store_true")
    return parser.parse_args()


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
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz, max_targets, num_classes = class_logits.shape
    loss_cls = F.cross_entropy(
        class_logits.reshape(bsz * max_targets, num_classes),
        class_label.reshape(bsz * max_targets),
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


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_state: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    acc = empty_accumulator()

    for batch in loader:
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
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, args.lambda_state, optimizer)
            val_metrics = run_epoch(model, val_loader, device, args.lambda_state, optimizer=None)

            record = {
                "type": "epoch",
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            append_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=False, allow_nan=False))

            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, args, val_metrics)
            if val_metrics["loss_total"] < best_val:
                best_val = val_metrics["loss_total"]
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, args, val_metrics)

        append_jsonl(log_path, {"type": "done", "best_val_loss": best_val})
    except Exception as exc:
        append_jsonl(log_path, {"type": "error", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
