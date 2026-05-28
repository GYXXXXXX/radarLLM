#!/usr/bin/env python3
"""Compare one scene's ground-truth and stitched predicted trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'matplotlib'. Install with: pip install matplotlib") from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'. Install train requirements.") from exc

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .config import STATE_MEAN, STATE_STD
    from .dataset import record_to_windows, scene_to_tensor_record
    from .mat_utils import collect_scene_files
    from .model import FmcwPatchSTTransformer
except ImportError:  # pragma: no cover
    from config import STATE_MEAN, STATE_STD
    from dataset import record_to_windows, scene_to_tensor_record
    from mat_utils import collect_scene_files
    from model import FmcwPatchSTTransformer


CLASS_NAMES = {
    0: "background",
    1: "T1_slow_smooth",
    2: "T2_uav_agile",
    3: "T3_pedestrian",
    4: "T4_fast_maneuver",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one scene's stitched trajectory prediction."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="radar_detect/fmcw_traj_dataset_2000scenes")
    parser.add_argument("--data-mode", choices=("mat_scene", "cached_scene"), default="mat_scene")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--scene-index", type=int, default=1600)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_model(checkpoint: dict[str, Any], device: torch.device) -> FmcwPatchSTTransformer:
    ckpt_args = checkpoint.get("args", {})
    model = FmcwPatchSTTransformer(
        in_channels=16,
        max_targets=int(ckpt_args.get("max_targets", 4)),
        tin=int(ckpt_args.get("tin", 16)),
        tout=int(ckpt_args.get("tout", 8)),
        nchirp=int(ckpt_args.get("nchirp", 32)),
        nfast=int(ckpt_args.get("nfast", 128)),
        patch_chirp=int(ckpt_args.get("patch_chirp", 8)),
        patch_fast=int(ckpt_args.get("patch_fast", 16)),
        embed_dim=int(ckpt_args.get("embed_dim", 128)),
        num_heads=int(ckpt_args.get("num_heads", 4)),
        spatial_layers=int(ckpt_args.get("spatial_layers", 1)),
        temporal_layers=int(ckpt_args.get("temporal_layers", 2)),
        mlp_ratio=float(ckpt_args.get("mlp_ratio", 2.0)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def denormalize_state_np(state: np.ndarray) -> np.ndarray:
    return state * STATE_STD.reshape(1, 1, 4) + STATE_MEAN.reshape(1, 1, 4)


def load_scene_record(scene_file: Path, checkpoint: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ckpt_args = checkpoint.get("args", {})
    if args.data_mode == "cached_scene":
        if not args.cache_dir:
            raise ValueError("--cache-dir is required with --data-mode cached_scene")
        return safe_torch_load(Path(args.cache_dir) / f"{scene_file.stem}.pt")
    return scene_to_tensor_record(
        scene_file,
        max_targets=int(ckpt_args.get("max_targets", 4)),
        iq_scale=float(ckpt_args.get("iq_scale", 76.0)),
    )


def predict_windows(
    model: FmcwPatchSTTransformer,
    x: torch.Tensor,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    objectness_probs = []
    state_preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], eval_batch_size):
            xb = x[start : start + eval_batch_size].to(device).float()
            objectness_logits, state_pred = model(xb)
            objectness_probs.append(torch.sigmoid(objectness_logits).cpu().numpy())
            state_preds.append(state_pred.cpu().numpy())
    return np.concatenate(objectness_probs, axis=0), np.concatenate(state_preds, axis=0)


def stitch_window_predictions(
    windows: dict[str, Any],
    pred_state_norm: np.ndarray,
    objectness_probs: np.ndarray,
    nframes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nwin, max_targets, tout, state_dim = pred_state_norm.shape
    pred_state = denormalize_state_np(pred_state_norm.reshape(-1, max_targets, tout, state_dim))
    pred_sum = np.zeros((max_targets, nframes, state_dim), dtype=np.float64)
    pred_count = np.zeros((max_targets, nframes), dtype=np.float64)

    starts = windows["start"].cpu().numpy()
    tin = int(windows["x"].shape[2])
    for win_idx in range(nwin):
        frames = np.arange(starts[win_idx] + tin, starts[win_idx] + tin + tout)
        pred_sum[:, frames, :] += pred_state[win_idx]
        pred_count[:, frames] += 1.0

    pred_full = np.full((max_targets, nframes, state_dim), np.nan, dtype=np.float32)
    covered = pred_count > 0
    pred_full[covered] = (pred_sum[covered] / pred_count[covered][:, None]).astype(np.float32)
    mean_objectness = objectness_probs.mean(axis=0)
    return pred_full, pred_count, mean_objectness


def compute_metrics(
    pred_full: np.ndarray,
    gt_full: np.ndarray,
    pred_count: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    per_slot = []
    ade_values = []
    fde_values = []
    position_values = []
    velocity_values = []

    for slot, is_target in enumerate(target_mask.astype(bool)):
        if not is_target:
            continue
        frame_mask = pred_count[slot] > 0
        if not np.any(frame_mask):
            continue

        diff = pred_full[slot, frame_mask] - gt_full[slot, frame_mask]
        pos_diff = diff[:, :2]
        vel_diff = diff[:, 2:]
        pos_l2 = np.linalg.norm(pos_diff, axis=1)
        frames = np.where(frame_mask)[0]

        per_slot.append(
            {
                "slot": int(slot),
                "covered_frame_start": int(frames[0]),
                "covered_frame_end": int(frames[-1]),
                "covered_frame_count": int(frames.size),
                "mean_overlap_count": float(np.mean(pred_count[slot, frame_mask])),
                "position_mae_m": float(np.mean(np.abs(pos_diff))),
                "velocity_mae_mps": float(np.mean(np.abs(vel_diff))),
                "ade_m": float(np.mean(pos_l2)),
                "fde_m": float(pos_l2[-1]),
            }
        )
        ade_values.append(pos_l2)
        fde_values.append(np.array([pos_l2[-1]], dtype=np.float32))
        position_values.append(np.abs(pos_diff).reshape(-1))
        velocity_values.append(np.abs(vel_diff).reshape(-1))

    if not per_slot:
        return {
            "per_slot": [],
            "position_mae_m": 0.0,
            "velocity_mae_mps": 0.0,
            "ade_m": 0.0,
            "fde_m": 0.0,
        }

    return {
        "per_slot": per_slot,
        "position_mae_m": float(np.mean(np.concatenate(position_values))),
        "velocity_mae_mps": float(np.mean(np.concatenate(velocity_values))),
        "ade_m": float(np.mean(np.concatenate(ade_values))),
        "fde_m": float(np.mean(np.concatenate(fde_values))),
    }


def plot_scene(
    out_path: Path,
    scene_name: str,
    pred_full: np.ndarray,
    gt_full: np.ndarray,
    pred_count: np.ndarray,
    target_mask: np.ndarray,
    class_label: np.ndarray,
    objectness: np.ndarray,
) -> None:
    nframes = gt_full.shape[1]
    frames = np.arange(nframes)
    colors = plt.cm.tab10(np.arange(target_mask.shape[0]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax_xy, ax_x, ax_y, ax_speed = axes.reshape(-1)

    for slot, is_target in enumerate(target_mask.astype(bool)):
        if not is_target:
            continue

        color = colors[slot]
        cls_name = CLASS_NAMES.get(int(class_label[slot]), str(class_label[slot]))
        label = f"slot {slot} {cls_name} obj={objectness[slot]:.2f}"
        covered = pred_count[slot] > 0

        ax_xy.plot(gt_full[slot, :, 0], gt_full[slot, :, 1], "-", color=color, label=f"GT {label}")
        ax_xy.plot(pred_full[slot, :, 0], pred_full[slot, :, 1], "--", color=color, label=f"Pred {label}")
        ax_xy.scatter(gt_full[slot, 0, 0], gt_full[slot, 0, 1], marker="s", color=color, s=55)
        if np.any(covered):
            first = int(np.where(covered)[0][0])
            ax_xy.scatter(pred_full[slot, first, 0], pred_full[slot, first, 1], marker="x", color=color, s=70)

        ax_x.plot(frames, gt_full[slot, :, 0], "-", color=color, label=f"GT slot {slot}")
        ax_x.plot(frames, pred_full[slot, :, 0], "--", color=color, label=f"Pred slot {slot}")

        ax_y.plot(frames, gt_full[slot, :, 1], "-", color=color, label=f"GT slot {slot}")
        ax_y.plot(frames, pred_full[slot, :, 1], "--", color=color, label=f"Pred slot {slot}")

        gt_speed = np.linalg.norm(gt_full[slot, :, 2:], axis=1)
        pred_speed = np.linalg.norm(pred_full[slot, :, 2:], axis=1)
        ax_speed.plot(frames, gt_speed, "-", color=color, label=f"GT slot {slot}")
        ax_speed.plot(frames, pred_speed, "--", color=color, label=f"Pred slot {slot}")

    ax_xy.set_title(f"Scene trajectory XY: {scene_name}")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8)

    ax_x.set_title("x over frames")
    ax_x.set_xlabel("Frame")
    ax_x.set_ylabel("x (m)")
    ax_x.grid(True, alpha=0.3)
    ax_x.legend(fontsize=8)

    ax_y.set_title("y over frames")
    ax_y.set_xlabel("Frame")
    ax_y.set_ylabel("y (m)")
    ax_y.grid(True, alpha=0.3)
    ax_y.legend(fontsize=8)

    ax_speed.set_title("speed over frames")
    ax_speed.set_xlabel("Frame")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = choose_device(args.device)
    checkpoint = safe_torch_load(checkpoint_path, device)
    ckpt_args = checkpoint.get("args", {})

    scene_files = collect_scene_files(args.dataset_dir)
    if args.scene_index < 0 or args.scene_index >= len(scene_files):
        raise IndexError(f"--scene-index must be in [0, {len(scene_files) - 1}]")
    scene_file = scene_files[args.scene_index]

    record = load_scene_record(scene_file, checkpoint, args)
    windows = record_to_windows(
        record,
        tin=int(ckpt_args.get("tin", 16)),
        tout=int(ckpt_args.get("tout", 8)),
        stride=int(ckpt_args.get("stride", 1)),
    )

    model = make_model(checkpoint, device)
    objectness_probs, pred_state_norm = predict_windows(
        model,
        windows["x"],
        device,
        eval_batch_size=args.eval_batch_size,
    )

    gt_full = record["state_full"].cpu().numpy()
    class_label = record["class_label"].cpu().numpy()
    target_mask = record["target_mask"].cpu().numpy()
    pred_full, pred_count, mean_objectness = stitch_window_predictions(
        windows,
        pred_state_norm,
        objectness_probs,
        nframes=gt_full.shape[1],
    )
    metrics = compute_metrics(pred_full, gt_full, pred_count, target_mask)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = checkpoint_path.parent / "prediction_plots" / f"{scene_file.stem}_scene_compare.png"

    plot_scene(
        out_path,
        scene_file.name,
        pred_full,
        gt_full,
        pred_count,
        target_mask,
        class_label,
        mean_objectness,
    )

    summary = {
        "scene": scene_file.name,
        "scene_index": args.scene_index,
        "checkpoint": str(checkpoint_path),
        "out": str(out_path),
        "covered_frame_start": int(np.where(pred_count.max(axis=0) > 0)[0][0]),
        "covered_frame_end": int(np.where(pred_count.max(axis=0) > 0)[0][-1]),
        "class_label": class_label.tolist(),
        "target_mask": target_mask.tolist(),
        "objectness_prob": mean_objectness.round(4).tolist(),
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
