#!/usr/bin/env python3
"""Train scene-first patch ST-Transformer with objectness + trajectory heads."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
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
    from .dataset import CachedSceneWindowDataset, SceneWindowDataset, scene_window_collate
    from .mat_utils import append_jsonl, collect_scene_files, set_seed
    from .model import FmcwPatchSTTransformer
except ImportError:  # pragma: no cover
    from config import STATE_MEAN, STATE_STD
    from dataset import CachedSceneWindowDataset, SceneWindowDataset, scene_window_collate
    from mat_utils import append_jsonl, collect_scene_files, set_seed
    from model import FmcwPatchSTTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FMCW patch ST-Transformer.")
    parser.add_argument("--dataset-dir", type=str, default="radar_detect/fmcw_traj_dataset_2000scenes")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--data-mode", choices=("mat_scene", "cached_scene"), default="mat_scene")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--tin", type=int, default=16)
    parser.add_argument("--tout", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--iq-scale", type=float, default=76.0)
    parser.add_argument("--scene-batch-size", type=int, default=2)
    parser.add_argument(
        "--val-scene-batch-size",
        type=int,
        default=0,
        help=(
            "Validation scene batch size. 0 reuses --scene-batch-size. "
            "Use a smaller value when validating all windows."
        ),
    )
    parser.add_argument(
        "--windows-per-scene",
        type=int,
        default=4,
        help="Training windows sampled per scene per batch; 0 uses all windows.",
    )
    parser.add_argument(
        "--val-windows-per-scene",
        type=int,
        default=0,
        help="Validation windows per scene; 0 evaluates all windows.",
    )
    parser.add_argument(
        "--max-train-scenes",
        type=int,
        default=0,
        help="Optional smoke-test cap after train/val split; 0 uses all training scenes.",
    )
    parser.add_argument(
        "--max-val-scenes",
        type=int,
        default=0,
        help="Optional smoke-test cap after train/val split; 0 uses all validation scenes.",
    )
    parser.add_argument(
        "--max-effective-batch-windows",
        type=int,
        default=512,
        help=(
            "Safety cap for scene_batch_size * training windows per scene. "
            "Set 0 to disable after confirming GPU memory."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--lambda-obj", type=float, default=0.1)
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--spatial-layers", type=int, default=1)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--patch-chirp", type=int, default=8)
    parser.add_argument("--patch-fast", type=int, default=16)
    parser.add_argument("--nchirp", type=int, default=32)
    parser.add_argument("--nfast", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument(
        "--scheduler-metric",
        choices=("loss_total", "loss_state", "ade_m"),
        default="loss_state",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-metric",
        choices=("loss_total", "loss_state", "ade_m"),
        default="loss_state",
    )
    parser.add_argument("--print-sample", action="store_true")
    args = parser.parse_args()

    if args.data_mode == "cached_scene" and not args.cache_dir:
        parser.error("--cache-dir is required when --data-mode cached_scene")
    if args.scene_batch_size <= 0:
        parser.error("--scene-batch-size must be > 0")
    if args.val_scene_batch_size < 0:
        parser.error("--val-scene-batch-size must be >= 0")
    if args.windows_per_scene < 0 or args.val_windows_per_scene < 0:
        parser.error("--windows-per-scene and --val-windows-per-scene must be >= 0")
    if args.max_train_scenes < 0 or args.max_val_scenes < 0:
        parser.error("--max-train-scenes and --max-val-scenes must be >= 0")
    if args.max_effective_batch_windows < 0:
        parser.error("--max-effective-batch-windows must be >= 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be >= 1")
    if args.dropout < 0 or args.dropout >= 1:
        parser.error("--dropout must be in [0, 1)")
    if args.lr_factor <= 0 or args.lr_factor >= 1:
        parser.error("--lr-factor must be in (0, 1)")
    if args.lambda_obj < 0 or args.lambda_state < 0:
        parser.error("--lambda-obj and --lambda-state must be >= 0")
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("radar_detect/train_st_transformer/runs") / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def split_scenes(scene_files: list[Path], train_ratio: float) -> tuple[list[Path], list[Path]]:
    split = int(len(scene_files) * train_ratio)
    split = min(max(split, 1), len(scene_files) - 1)
    return scene_files[:split], scene_files[split:]


def apply_scene_limits(
    train_scenes: list[Path],
    val_scenes: list[Path],
    max_train_scenes: int,
    max_val_scenes: int,
) -> tuple[list[Path], list[Path]]:
    if max_train_scenes:
        train_scenes = train_scenes[:max_train_scenes]
    if max_val_scenes:
        val_scenes = val_scenes[:max_val_scenes]
    if not train_scenes:
        raise ValueError("No training scenes after applying --max-train-scenes.")
    if not val_scenes:
        raise ValueError("No validation scenes after applying --max-val-scenes.")
    return train_scenes, val_scenes


def estimate_windows_per_scene(tin: int, tout: int, stride: int, nframes: int = 48) -> int:
    return max(0, (int(nframes) - int(tin) - int(tout)) // int(stride) + 1)


def validate_effective_batch(args: argparse.Namespace) -> None:
    all_windows_per_scene = estimate_windows_per_scene(args.tin, args.tout, args.stride)
    train_windows_per_scene = (
        all_windows_per_scene
        if args.windows_per_scene == 0
        else min(args.windows_per_scene, all_windows_per_scene)
    )
    effective_batch_windows = args.scene_batch_size * train_windows_per_scene
    if (
        args.max_effective_batch_windows
        and effective_batch_windows > args.max_effective_batch_windows
    ):
        raise ValueError(
            "Requested scene batch is too large for the scene-first loader: "
            f"scene_batch_size={args.scene_batch_size}, "
            f"train_windows_per_scene={train_windows_per_scene}, "
            f"effective_window_batch={effective_batch_windows}. "
            f"Reduce --scene-batch-size or raise --max-effective-batch-windows. "
            "For a smoke test, try --scene-batch-size 2 --windows-per-scene 2 "
            "--max-train-scenes 8 --max-val-scenes 2."
        )


def make_loader(
    scene_files: list[Path],
    args: argparse.Namespace,
    shuffle: bool,
    windows_per_scene: int,
    random_windows: bool,
    scene_batch_size: int | None = None,
) -> DataLoader:
    if args.data_mode == "cached_scene":
        dataset = CachedSceneWindowDataset(
            scene_files,
            cache_dir=args.cache_dir,
            tin=args.tin,
            tout=args.tout,
            stride=args.stride,
        )
    else:
        dataset = SceneWindowDataset(
            scene_files,
            tin=args.tin,
            tout=args.tout,
            stride=args.stride,
            max_targets=args.max_targets,
            iq_scale=args.iq_scale,
        )

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.scene_batch_size if scene_batch_size is None else scene_batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available() and not args.no_pin_memory,
        "collate_fn": partial(
            scene_window_collate,
            windows_per_scene=windows_per_scene,
            random_windows=random_windows,
        ),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "x": batch["x"].to(device, non_blocking=True).float(),
        "state_label": batch["state_label"].to(device, non_blocking=True).float(),
        "target_mask": batch["target_mask"].to(device, non_blocking=True).float(),
        "class_label": batch["class_label"].to(device, non_blocking=True).long(),
    }


def compute_loss(
    objectness_logits: torch.Tensor,
    state_pred: torch.Tensor,
    state_label: torch.Tensor,
    target_mask: torch.Tensor,
    lambda_obj: float,
    lambda_state: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_obj = F.binary_cross_entropy_with_logits(objectness_logits, target_mask)

    raw_state_loss = F.smooth_l1_loss(state_pred, state_label, reduction="none")
    mask = target_mask[:, :, None, None]
    valid_count = (target_mask.sum() * state_pred.shape[2] * state_pred.shape[3]).clamp_min(1.0)
    loss_state = (raw_state_loss * mask).sum() / valid_count
    loss_total = lambda_obj * loss_obj + lambda_state * loss_state
    return loss_total, {"loss_obj": loss_obj, "loss_state": loss_state}


def denormalize_state(state: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(STATE_MEAN, dtype=state.dtype, device=state.device).view(1, 1, 1, 4)
    std = torch.as_tensor(STATE_STD, dtype=state.dtype, device=state.device).view(1, 1, 1, 4)
    return state * std + mean


def batch_metrics(
    objectness_logits: torch.Tensor,
    state_pred: torch.Tensor,
    state_label: torch.Tensor,
    target_mask: torch.Tensor,
    objectness_threshold: float,
) -> dict[str, float]:
    with torch.no_grad():
        pred_obj = (torch.sigmoid(objectness_logits) >= objectness_threshold).float()
        slot_correct = (pred_obj == target_mask).float()
        slot_acc_sum = slot_correct.sum().item()
        slot_count = float(target_mask.numel())

        target_count = target_mask.sum().item()
        target_obj_correct = (slot_correct * target_mask).sum().item()
        bg_mask = 1.0 - target_mask
        bg_count = bg_mask.sum().item()
        bg_obj_correct = (slot_correct * bg_mask).sum().item()

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
        "obj_slot_acc_sum": slot_acc_sum,
        "obj_slot_count": slot_count,
        "obj_target_acc_sum": target_obj_correct,
        "target_count": max(target_count, 0.0),
        "obj_bg_acc_sum": bg_obj_correct,
        "bg_count": max(bg_count, 0.0),
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
        "loss_obj_sum": 0.0,
        "loss_state_sum": 0.0,
        "batch_count": 0.0,
        "sample_count": 0.0,
        "obj_slot_acc_sum": 0.0,
        "obj_slot_count": 0.0,
        "obj_target_acc_sum": 0.0,
        "target_count": 0.0,
        "obj_bg_acc_sum": 0.0,
        "bg_count": 0.0,
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
    acc["loss_obj_sum"] += float(loss_parts["loss_obj"].detach().cpu()) * batch_size
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
        "loss_obj": acc["loss_obj_sum"] / samples,
        "loss_state": acc["loss_state_sum"] / samples,
        "objectness_slot_accuracy": ratio("obj_slot_acc_sum", "obj_slot_count"),
        "objectness_target_accuracy": ratio("obj_target_acc_sum", "target_count"),
        "objectness_background_accuracy": ratio("obj_bg_acc_sum", "bg_count"),
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
    metrics = finalize_metrics(acc)
    print(
        f"phase={phase} epoch={epoch} batch={batch_index} total_batch={total_batches} "
        f"loss_total={metrics['loss_total']:.6f} "
        f"loss_obj={metrics['loss_obj']:.6f} "
        f"loss_state={metrics['loss_state']:.6f} "
        f"ade_m={metrics['ade_m']:.4f}",
        flush=True,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_obj: float,
    lambda_state: float,
    objectness_threshold: float,
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
            objectness_logits, state_pred = model(batch["x"])
            loss_total, loss_parts = compute_loss(
                objectness_logits,
                state_pred,
                batch["state_label"],
                batch["target_mask"],
                lambda_obj=lambda_obj,
                lambda_state=lambda_state,
            )
            if not torch.isfinite(loss_total):
                raise FloatingPointError("Non-finite loss detected.")
            if is_train:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()

        metrics = batch_metrics(
            objectness_logits.detach(),
            state_pred.detach(),
            batch["state_label"],
            batch["target_mask"],
            objectness_threshold=objectness_threshold,
        )
        add_batch_stats(acc, loss_total, loss_parts, metrics, batch_size=batch["x"].shape[0])

        if log_every and (batch_index % log_every == 0 or batch_index == total_batches):
            print_progress(phase, epoch, batch_index, total_batches, acc)

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
            "model_type": "patch_st_transformer_objectness",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
            "state_mean": STATE_MEAN,
            "state_std": STATE_STD,
        },
        path,
    )


def make_model(args: argparse.Namespace) -> FmcwPatchSTTransformer:
    return FmcwPatchSTTransformer(
        in_channels=16,
        max_targets=args.max_targets,
        tin=args.tin,
        tout=args.tout,
        nchirp=args.nchirp,
        nfast=args.nfast,
        patch_chirp=args.patch_chirp,
        patch_fast=args.patch_fast,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        spatial_layers=args.spatial_layers,
        temporal_layers=args.temporal_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )


def maybe_save_best(
    value: float,
    best_value: float,
    checkpoint_name: str,
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    val_metrics: dict[str, float],
    min_delta: float = 0.0,
) -> tuple[float, bool]:
    if value < best_value - min_delta:
        save_checkpoint(run_dir / checkpoint_name, model, optimizer, epoch, args, val_metrics)
        return value, True
    return best_value, False


def main() -> None:
    args = parse_args()
    validate_effective_batch(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = make_run_dir(args)
    log_path = run_dir / "metrics.jsonl"

    scene_files = collect_scene_files(args.dataset_dir)
    train_scenes, val_scenes = split_scenes(scene_files, args.train_ratio)
    train_scenes, val_scenes = apply_scene_limits(
        train_scenes,
        val_scenes,
        args.max_train_scenes,
        args.max_val_scenes,
    )
    train_loader = make_loader(
        train_scenes,
        args,
        shuffle=True,
        windows_per_scene=args.windows_per_scene,
        random_windows=True,
    )
    val_loader = make_loader(
        val_scenes,
        args,
        shuffle=False,
        windows_per_scene=args.val_windows_per_scene,
        random_windows=False,
        scene_batch_size=args.val_scene_batch_size or args.scene_batch_size,
    )

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
        "train_window_count_available": train_loader.dataset.window_count,
        "val_window_count_available": val_loader.dataset.window_count,
        "train_effective_windows_per_epoch": (
            len(train_scenes)
            * (
                estimate_windows_per_scene(args.tin, args.tout, args.stride)
                if args.windows_per_scene == 0
                else min(args.windows_per_scene, estimate_windows_per_scene(args.tin, args.tout, args.stride))
            )
        ),
        "train_effective_batch_windows": (
            args.scene_batch_size
            * (
                estimate_windows_per_scene(args.tin, args.tout, args.stride)
                if args.windows_per_scene == 0
                else min(args.windows_per_scene, estimate_windows_per_scene(args.tin, args.tout, args.stride))
            )
        ),
        "val_effective_windows_per_epoch": (
            len(val_scenes)
            * (
                estimate_windows_per_scene(args.tin, args.tout, args.stride)
                if args.val_windows_per_scene == 0
                else min(args.val_windows_per_scene, estimate_windows_per_scene(args.tin, args.tout, args.stride))
            )
        ),
        "val_effective_batch_windows": (
            (args.val_scene_batch_size or args.scene_batch_size)
            * (
                estimate_windows_per_scene(args.tin, args.tout, args.stride)
                if args.val_windows_per_scene == 0
                else min(args.val_windows_per_scene, estimate_windows_per_scene(args.tin, args.tout, args.stride))
            )
        ),
        "model_type": "patch_st_transformer_objectness",
        "state_mean": STATE_MEAN.tolist(),
        "state_std": STATE_STD.tolist(),
    }
    append_jsonl(log_path, config_record)

    if args.print_sample:
        sample = train_loader.dataset[0]
        print("scene item x shape:", tuple(sample["x"].shape))
        print("scene item state_label shape:", tuple(sample["state_label"].shape))
        print("scene item target_mask shape:", tuple(sample["target_mask"].shape))
        print("scene item starts:", sample["start"][:5].tolist(), "...")

    print(json.dumps(config_record, ensure_ascii=False, indent=2))
    best_total = float("inf")
    best_state = float("inf")
    best_ade = float("inf")
    best_early = float("inf")
    best_epoch = 0
    epochs_since_best = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                args.lambda_obj,
                args.lambda_state,
                args.objectness_threshold,
                optimizer,
                epoch=epoch,
                phase="train",
                log_every=args.log_every,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                args.lambda_obj,
                args.lambda_state,
                args.objectness_threshold,
                optimizer=None,
                epoch=epoch,
                phase="val",
                log_every=args.log_every,
            )

            record = {"type": "epoch", "epoch": epoch, "train": train_metrics, "val": val_metrics}
            append_jsonl(log_path, record)
            print(json.dumps(record, ensure_ascii=False, allow_nan=False))

            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, args, val_metrics)
            scheduler.step(val_metrics[args.scheduler_metric])

            best_total, total_improved = maybe_save_best(
                val_metrics["loss_total"],
                best_total,
                "best.pt",
                run_dir,
                model,
                optimizer,
                epoch,
                args,
                val_metrics,
                min_delta=args.early_stop_min_delta if args.early_stop_metric == "loss_total" else 0.0,
            )
            best_state, state_improved = maybe_save_best(
                val_metrics["loss_state"],
                best_state,
                "best_by_loss_state.pt",
                run_dir,
                model,
                optimizer,
                epoch,
                args,
                val_metrics,
                min_delta=args.early_stop_min_delta if args.early_stop_metric == "loss_state" else 0.0,
            )
            best_ade, ade_improved = maybe_save_best(
                val_metrics["ade_m"],
                best_ade,
                "best_by_ade.pt",
                run_dir,
                model,
                optimizer,
                epoch,
                args,
                val_metrics,
                min_delta=args.early_stop_min_delta if args.early_stop_metric == "ade_m" else 0.0,
            )

            current_early = val_metrics[args.early_stop_metric]
            if current_early < best_early - args.early_stop_min_delta:
                best_early = current_early
                best_epoch = epoch
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            if total_improved or state_improved or ade_improved:
                append_jsonl(
                    log_path,
                    {
                        "type": "checkpoint",
                        "epoch": epoch,
                        "best_total": best_total,
                        "best_loss_state": best_state,
                        "best_ade": best_ade,
                    },
                )

            if args.early_stop_patience and epochs_since_best >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best {args.early_stop_metric} at epoch {best_epoch}: {best_early:.6f}).",
                    flush=True,
                )
                break

        append_jsonl(
            log_path,
            {
                "type": "done",
                "best_val_loss": best_total,
                "best_val_loss_state": best_state,
                "best_val_ade": best_ade,
                "best_epoch": best_epoch,
            },
        )
    except Exception as exc:
        append_jsonl(log_path, {"type": "error", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
