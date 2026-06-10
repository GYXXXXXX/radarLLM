#!/usr/bin/env python3
"""Train a radar Transformer for target intention prediction."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .config import (
        INTENT_NAMES,
        NUM_INTENTS,
        NUM_TARGET_CLASSES,
        NUM_THREATS,
        STATE_MEAN,
        STATE_STD,
        TARGET_CLASS_NAMES,
        THREAT_NAMES,
    )
    from .dataset import SceneWindowDataset, scene_window_collate
    from .mat_utils import append_jsonl, collect_scene_files, set_seed
    from .model import FmcwIntentTransformer, TrackIntentTransformer
except ImportError:  # pragma: no cover
    from config import (
        INTENT_NAMES,
        NUM_INTENTS,
        NUM_TARGET_CLASSES,
        NUM_THREATS,
        STATE_MEAN,
        STATE_STD,
        TARGET_CLASS_NAMES,
        THREAT_NAMES,
    )
    from dataset import SceneWindowDataset, scene_window_collate
    from mat_utils import append_jsonl, collect_scene_files, set_seed
    from model import FmcwIntentTransformer, TrackIntentTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FMCW Transformer for target intention.")
    parser.add_argument("--dataset-dir", type=str, default="detect_intention/intention_dataset_compact")
    parser.add_argument(
        "--input-mode",
        choices=("track", "radar"),
        default="track",
        help="track uses radar-derived target trajectories; radar uses raw IQ end-to-end.",
    )
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--tin", type=int, default=0)
    parser.add_argument("--tout", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--iq-scale", type=float, default=76.0)
    parser.add_argument("--scene-batch-size", type=int, default=1)
    parser.add_argument("--val-scene-batch-size", type=int, default=0)
    parser.add_argument("--windows-per-scene", type=int, default=0)
    parser.add_argument("--val-windows-per-scene", type=int, default=0)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-val-scenes", type=int, default=0)
    parser.add_argument("--max-effective-batch-windows", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--lambda-intent", type=float, default=1.0)
    parser.add_argument("--lambda-threat", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--spatial-layers", type=int, default=1)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--patch-chirp", type=int, default=0)
    parser.add_argument("--patch-fast", type=int, default=0)
    parser.add_argument("--nchirp", type=int, default=0)
    parser.add_argument("--nfast", type=int, default=0)
    parser.add_argument("--nrx", type=int, default=0)
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
        choices=("loss_total", "loss_intent", "ade_m"),
        default="loss_intent",
    )
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stop-metric",
        choices=("loss_total", "loss_intent", "ade_m"),
        default="loss_intent",
    )
    parser.add_argument("--print-sample", action="store_true")
    args = parser.parse_args()

    if args.scene_batch_size <= 0:
        parser.error("--scene-batch-size must be > 0")
    if args.val_scene_batch_size < 0:
        parser.error("--val-scene-batch-size must be >= 0")
    if args.windows_per_scene < 0 or args.val_windows_per_scene < 0:
        parser.error("--windows-per-scene and --val-windows-per-scene must be >= 0")
    if args.max_train_scenes < 0 or args.max_val_scenes < 0:
        parser.error("--max-train-scenes and --max-val-scenes must be >= 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be >= 1")
    if args.dropout < 0 or args.dropout >= 1:
        parser.error("--dropout must be in [0, 1)")
    if args.lr_factor <= 0 or args.lr_factor >= 1:
        parser.error("--lr-factor must be in (0, 1)")
    if args.label_smoothing < 0 or args.label_smoothing >= 1:
        parser.error("--label-smoothing must be in [0, 1)")
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dataset_dir(dataset_dir: str) -> str:
    if dataset_dir != "auto":
        return dataset_dir

    candidates = [
        Path("detect_intention/intention_dataset_compact"),
        Path("detect_intention/intention_dataset"),
    ]
    available: list[tuple[int, Path]] = []
    for candidate in candidates:
        if candidate.exists():
            scene_count = len(list(candidate.glob("scene_*.mat")))
            if scene_count > 0:
                available.append((scene_count, candidate))
    if available:
        available.sort(key=lambda item: item[0], reverse=True)
        return str(available[0][1])
    raise FileNotFoundError(
        "No intention dataset found. Generate one first, for example: "
        "generate_intention_fmcw_dataset(5000, [], 'compact')"
    )


def _mat_int(value: Any) -> int:
    import numpy as np

    return int(np.asarray(value).reshape(-1)[0])


def infer_dataset_params(args: argparse.Namespace, scene_files: list[Path]) -> None:
    try:
        from .mat_utils import load_scene
    except ImportError:  # pragma: no cover
        from mat_utils import load_scene

    scene = load_scene(scene_files[0])
    nfast = _mat_int(scene["p"]["Nfast"])
    nchirp = _mat_int(scene["p"]["Nchirp"])
    nrx = _mat_int(scene["p"]["Nrx"])
    nframes = _mat_int(scene["p"]["Nframes"])

    if args.nfast == 0:
        args.nfast = nfast
    if args.nchirp == 0:
        args.nchirp = nchirp
    if args.nrx == 0:
        args.nrx = nrx

    if args.tin == 0 and args.tout == 0:
        args.tin = max(1, int(round(nframes * 0.75)))
        args.tout = nframes - args.tin
    elif args.tin == 0:
        args.tin = nframes - args.tout
    elif args.tout == 0:
        args.tout = nframes - args.tin

    if args.tin <= 0 or args.tout <= 0 or args.tin + args.tout > nframes:
        raise ValueError(
            f"Invalid Tin/Tout for dataset: Nframes={nframes}, tin={args.tin}, tout={args.tout}."
        )

    if args.patch_chirp == 0:
        args.patch_chirp = 8 if args.nchirp % 8 == 0 else args.nchirp
    if args.patch_fast == 0:
        args.patch_fast = 16 if args.nfast % 16 == 0 else args.nfast

    if args.nchirp % args.patch_chirp != 0:
        raise ValueError(f"nchirp={args.nchirp} must be divisible by patch_chirp={args.patch_chirp}.")
    if args.nfast % args.patch_fast != 0:
        raise ValueError(f"nfast={args.nfast} must be divisible by patch_fast={args.patch_fast}.")

    args.inferred_nframes = nframes
    args.inferred_from_scene = str(scene_files[0])


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("detect_intention/train_intention/runs") / f"run_{stamp}"
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
    all_windows = estimate_windows_per_scene(
        args.tin,
        args.tout,
        args.stride,
        nframes=getattr(args, "inferred_nframes", 48),
    )
    train_windows = all_windows if args.windows_per_scene == 0 else min(args.windows_per_scene, all_windows)
    effective = args.scene_batch_size * train_windows
    if args.max_effective_batch_windows and effective > args.max_effective_batch_windows:
        raise ValueError(
            "Requested scene batch is too large: "
            f"scene_batch_size={args.scene_batch_size}, windows_per_scene={train_windows}, "
            f"effective_window_batch={effective}."
        )


def make_loader(
    scene_files: list[Path],
    args: argparse.Namespace,
    shuffle: bool,
    windows_per_scene: int,
    random_windows: bool,
    scene_batch_size: int | None = None,
) -> DataLoader:
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
        "state_input": batch["state_input"].to(device, non_blocking=True).float(),
        "state_label": batch["state_label"].to(device, non_blocking=True).float(),
        "target_mask": batch["target_mask"].to(device, non_blocking=True).float(),
        "target_class_label": batch["target_class_label"].to(device, non_blocking=True).long(),
        "intent_label": batch["intent_label"].to(device, non_blocking=True).long(),
        "threat_label": batch["threat_label"].to(device, non_blocking=True).long(),
    }


def normalize_state_input(state_input: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(STATE_MEAN, dtype=state_input.dtype, device=state_input.device).view(1, 1, 1, 4)
    std = torch.as_tensor(STATE_STD, dtype=state_input.dtype, device=state_input.device).view(1, 1, 1, 4)
    return (state_input - mean) / std


def forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], input_mode: str) -> dict[str, torch.Tensor]:
    if input_mode == "track":
        return model(normalize_state_input(batch["state_input"]))
    return model(batch["x"])


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    valid = target_mask > 0.5
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], labels[valid], label_smoothing=label_smoothing)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_mask = batch["target_mask"]

    raw_state_loss = F.smooth_l1_loss(outputs["state_pred"], batch["state_label"], reduction="none")
    mask = target_mask[:, :, None, None]
    valid_count = (target_mask.sum() * outputs["state_pred"].shape[2] * outputs["state_pred"].shape[3]).clamp_min(1.0)
    loss_state = (raw_state_loss * mask).sum() / valid_count

    loss_intent = masked_cross_entropy(
        outputs["intent_logits"],
        batch["intent_label"],
        target_mask,
        label_smoothing=args.label_smoothing,
    )
    loss_threat = masked_cross_entropy(
        outputs["threat_logits"],
        batch["threat_label"],
        target_mask,
        label_smoothing=args.label_smoothing,
    )

    loss_total = (
        args.lambda_state * loss_state
        + args.lambda_intent * loss_intent
        + args.lambda_threat * loss_threat
    )
    return loss_total, {
        "loss_state": loss_state,
        "loss_intent": loss_intent,
        "loss_threat": loss_threat,
    }


def denormalize_state(state: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(STATE_MEAN, dtype=state.dtype, device=state.device).view(1, 1, 1, 4)
    std = torch.as_tensor(STATE_STD, dtype=state.dtype, device=state.device).view(1, 1, 1, 4)
    return state * std + mean


def masked_accuracy_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        valid = target_mask > 0.5
        count = float(valid.sum().item())
        if count <= 0:
            return 0.0, 0.0
        pred = torch.argmax(logits, dim=-1)
        correct = ((pred == labels) & valid).sum().item()
        return float(correct), count


def batch_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    objectness_threshold: float,
) -> dict[str, float]:
    with torch.no_grad():
        target_mask = batch["target_mask"]
        target_count = target_mask.sum().item()

        intent_correct, intent_count = masked_accuracy_sum(
            outputs["intent_logits"],
            batch["intent_label"],
            target_mask,
        )
        threat_correct, threat_count = masked_accuracy_sum(
            outputs["threat_logits"],
            batch["threat_label"],
            target_mask,
        )

        pred_denorm = denormalize_state(outputs["state_pred"])
        label_denorm = denormalize_state(batch["state_label"])
        diff = pred_denorm - label_denorm
        mask = target_mask[:, :, None, None]

        pos_abs_sum = (diff[..., :2].abs() * mask).sum().item()
        vel_abs_sum = (diff[..., 2:].abs() * mask).sum().item()
        pos_count = target_count * outputs["state_pred"].shape[2] * 2
        vel_count = target_count * outputs["state_pred"].shape[2] * 2

        pos_l2 = torch.sqrt(torch.square(diff[..., :2]).sum(dim=-1).clamp_min(0.0))
        frame_mask = target_mask[:, :, None]
        ade_sum = (pos_l2 * frame_mask).sum().item()
        ade_count = target_count * outputs["state_pred"].shape[2]
        fde_sum = (pos_l2[:, :, -1] * target_mask).sum().item()

    return {
        "target_count": max(target_count, 0.0),
        "intent_correct": intent_correct,
        "intent_count": intent_count,
        "threat_correct": threat_correct,
        "threat_count": threat_count,
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
    keys = [
        "loss_total_sum",
        "loss_state_sum",
        "loss_intent_sum",
        "loss_threat_sum",
        "batch_count",
        "sample_count",
        "target_count",
        "intent_correct",
        "intent_count",
        "threat_correct",
        "threat_count",
        "pos_abs_sum",
        "pos_count",
        "vel_abs_sum",
        "vel_count",
        "ade_sum",
        "ade_count",
        "fde_sum",
        "fde_count",
    ]
    return {key: 0.0 for key in keys}


def add_batch_stats(
    acc: dict[str, float],
    loss_total: torch.Tensor,
    loss_parts: dict[str, torch.Tensor],
    metric_parts: dict[str, float],
    batch_size: int,
) -> None:
    acc["loss_total_sum"] += float(loss_total.detach().cpu()) * batch_size
    for name in ["loss_state", "loss_intent", "loss_threat"]:
        acc[f"{name}_sum"] += float(loss_parts[name].detach().cpu()) * batch_size
    acc["batch_count"] += 1.0
    acc["sample_count"] += float(batch_size)
    for key, value in metric_parts.items():
        acc[key] += float(value)


def finalize_metrics(acc: dict[str, float]) -> dict[str, float]:
    samples = max(acc["sample_count"], 1.0)

    def ratio(num_key: str, den_key: str) -> float:
        denom = acc[den_key]
        return float(acc[num_key] / denom) if denom > 0 else 0.0

    metrics = {
        "loss_total": acc["loss_total_sum"] / samples,
        "loss_state": acc["loss_state_sum"] / samples,
        "loss_intent": acc["loss_intent_sum"] / samples,
        "loss_threat": acc["loss_threat_sum"] / samples,
        "intent_accuracy": ratio("intent_correct", "intent_count"),
        "threat_accuracy": ratio("threat_correct", "threat_count"),
        "position_mae_m": ratio("pos_abs_sum", "pos_count"),
        "velocity_mae_mps": ratio("vel_abs_sum", "vel_count"),
        "ade_m": ratio("ade_sum", "ade_count"),
        "fde_m": ratio("fde_sum", "fde_count"),
    }
    return metrics


def print_progress(
    phase: str,
    epoch: int,
    batch_index: int,
    total_batches: int,
    acc: dict[str, float],
) -> None:
    metrics = finalize_metrics(acc)
    print(
        f"phase={phase} epoch={epoch} batch={batch_index}/{total_batches} "
        f"loss_total={metrics['loss_total']:.6f} "
        f"loss_intent={metrics['loss_intent']:.6f} "
        f"intent_acc={metrics['intent_accuracy']:.4f} "
        f"threat_acc={metrics['threat_accuracy']:.4f} "
        f"ade_m={metrics['ade_m']:.4f}",
        flush=True,
    )


def print_epoch_summary(epoch: int, epochs: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    print(
        f"[epoch {epoch:03d}/{epochs:03d}] "
        f"train loss={train_metrics['loss_total']:.6f} "
        f"intent_acc={train_metrics['intent_accuracy']:.4f} "
        f"threat_acc={train_metrics['threat_accuracy']:.4f} "
        f"ade={train_metrics['ade_m']:.3f} | "
        f"val loss={val_metrics['loss_total']:.6f} "
        f"intent_acc={val_metrics['intent_accuracy']:.4f} "
        f"threat_acc={val_metrics['threat_accuracy']:.4f} "
        f"ade={val_metrics['ade_m']:.3f}",
        flush=True,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: str,
    log_every: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    acc = empty_accumulator()
    total_batches = len(loader)

    for batch_index, raw_batch in enumerate(loader, start=1):
        batch = move_batch(raw_batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = forward_model(model, batch, args.input_mode)
            loss_total, loss_parts = compute_loss(outputs, batch, args)
            if not torch.isfinite(loss_total):
                raise FloatingPointError("Non-finite loss detected.")
            if is_train:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()

        metrics = batch_metrics(
            {key: value.detach() for key, value in outputs.items() if isinstance(value, torch.Tensor)},
            batch,
            objectness_threshold=args.objectness_threshold,
        )
        add_batch_stats(acc, loss_total, loss_parts, metrics, batch_size=batch["x"].shape[0])

        if log_every and (batch_index % log_every == 0 or batch_index == total_batches):
            print_progress(phase, epoch, batch_index, total_batches, acc)

    return finalize_metrics(acc)


def make_model(args: argparse.Namespace) -> FmcwIntentTransformer:
    if args.input_mode == "track":
        return TrackIntentTransformer(
            max_targets=args.max_targets,
            tin=args.tin,
            tout=args.tout,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            temporal_layers=args.temporal_layers,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            num_target_classes=NUM_TARGET_CLASSES,
            num_intents=NUM_INTENTS,
            num_threats=NUM_THREATS,
        )

    return FmcwIntentTransformer(
        in_channels=2 * args.nrx,
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
        num_target_classes=NUM_TARGET_CLASSES,
        num_intents=NUM_INTENTS,
        num_threats=NUM_THREATS,
    )


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
            "model_type": "track_intent_transformer" if args.input_mode == "track" else "fmcw_intent_transformer",
            "input_mode": args.input_mode,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
            "state_mean": STATE_MEAN,
            "state_std": STATE_STD,
            "target_class_names": TARGET_CLASS_NAMES,
            "intent_names": INTENT_NAMES,
            "threat_names": THREAT_NAMES,
        },
        path,
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
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = parse_args()
    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = make_run_dir(args)
    log_path = run_dir / "metrics.jsonl"

    scene_files = collect_scene_files(args.dataset_dir)
    infer_dataset_params(args, scene_files)
    validate_effective_batch(args)
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

    all_windows = estimate_windows_per_scene(
        args.tin,
        args.tout,
        args.stride,
        nframes=getattr(args, "inferred_nframes", 48),
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
        "windows_per_scene_available": all_windows,
        "model_type": "track_intent_transformer" if args.input_mode == "track" else "fmcw_intent_transformer",
        "state_mean": STATE_MEAN.tolist(),
        "state_std": STATE_STD.tolist(),
        "target_class_names": TARGET_CLASS_NAMES,
        "intent_names": INTENT_NAMES,
        "threat_names": THREAT_NAMES,
    }
    append_jsonl(log_path, config_record)

    if args.print_sample:
        sample = train_loader.dataset[0]
        print("scene item x shape:", tuple(sample["x"].shape))
        print("scene item state_label shape:", tuple(sample["state_label"].shape))
        print("scene item intent_label:", sample["intent_label"][0].tolist())
        print("scene item target_mask:", sample["target_mask"][0].tolist())

    print(json.dumps(config_record, ensure_ascii=False, indent=2))
    best_total = float("inf")
    best_intent = float("inf")
    best_ade = float("inf")
    best_intent_acc = -float("inf")
    best_early = float("inf")
    best_epoch = 0
    epochs_since_best = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                args,
                optimizer,
                epoch=epoch,
                phase="train",
                log_every=args.log_every,
            )
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                args,
                optimizer=None,
                epoch=epoch,
                phase="val",
                log_every=args.log_every,
            )

            record = {"type": "epoch", "epoch": epoch, "train": train_metrics, "val": val_metrics}
            append_jsonl(log_path, record)
            print_epoch_summary(epoch, args.epochs, train_metrics, val_metrics)
            print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)

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
            best_intent, intent_improved = maybe_save_best(
                val_metrics["loss_intent"],
                best_intent,
                "best_by_intent.pt",
                run_dir,
                model,
                optimizer,
                epoch,
                args,
                val_metrics,
                min_delta=args.early_stop_min_delta if args.early_stop_metric == "loss_intent" else 0.0,
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
            intent_acc_improved = val_metrics["intent_accuracy"] > best_intent_acc + 1e-8
            if intent_acc_improved:
                best_intent_acc = val_metrics["intent_accuracy"]
                save_checkpoint(
                    run_dir / "best_by_intent_accuracy.pt",
                    model,
                    optimizer,
                    epoch,
                    args,
                    val_metrics,
                )

            current_early = val_metrics[args.early_stop_metric]
            if current_early < best_early - args.early_stop_min_delta:
                best_early = current_early
                best_epoch = epoch
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            if total_improved or intent_improved or ade_improved or intent_acc_improved:
                append_jsonl(
                    log_path,
                    {
                        "type": "checkpoint",
                        "epoch": epoch,
                        "best_total": best_total,
                        "best_loss_intent": best_intent,
                        "best_ade": best_ade,
                        "best_intent_accuracy": best_intent_acc,
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
                "best_val_loss_intent": best_intent,
                "best_val_ade": best_ade,
                "best_val_intent_accuracy": best_intent_acc,
                "best_epoch": best_epoch,
            },
        )
    except Exception as exc:
        append_jsonl(log_path, {"type": "error", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
