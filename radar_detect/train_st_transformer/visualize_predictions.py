#!/usr/bin/env python3
"""Visualize ST-Transformer objectness and trajectory predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
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
    parser = argparse.ArgumentParser(description="Plot ST-Transformer trajectory predictions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="radar_detect/fmcw_traj_dataset_2000scenes")
    parser.add_argument("--cache-dir", type=str, default="")
    parser.add_argument("--data-mode", choices=("mat_scene", "cached_scene"), default="mat_scene")
    parser.add_argument("--scene-index", type=int, default=1600)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--stitch-scene", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--show", action="store_true")
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
    obj_probs = []
    states = []
    with torch.no_grad():
        for start in range(0, x.shape[0], eval_batch_size):
            xb = x[start : start + eval_batch_size].to(device).float()
            objectness_logits, state_pred = model(xb)
            obj_probs.append(torch.sigmoid(objectness_logits).cpu().numpy())
            states.append(state_pred.cpu().numpy())
    return np.concatenate(obj_probs, axis=0), np.concatenate(states, axis=0)


def compute_window_metrics(pred_state: np.ndarray, label_state: np.ndarray, target_mask: np.ndarray) -> dict:
    per_slot = []
    pos_abs_values = []
    vel_abs_values = []
    ade_values = []
    fde_values = []
    for slot, is_target in enumerate(target_mask.astype(bool)):
        if not is_target:
            continue
        diff = pred_state[slot] - label_state[slot]
        pos_diff = diff[:, :2]
        vel_diff = diff[:, 2:]
        pos_l2 = np.linalg.norm(pos_diff, axis=1)
        per_slot.append(
            {
                "slot": int(slot),
                "position_mae_m": float(np.mean(np.abs(pos_diff))),
                "velocity_mae_mps": float(np.mean(np.abs(vel_diff))),
                "ade_m": float(np.mean(pos_l2)),
                "fde_m": float(pos_l2[-1]),
            }
        )
        pos_abs_values.append(np.abs(pos_diff).reshape(-1))
        vel_abs_values.append(np.abs(vel_diff).reshape(-1))
        ade_values.append(pos_l2)
        fde_values.append(np.array([pos_l2[-1]], dtype=np.float32))
    if not per_slot:
        return {"per_slot": [], "position_mae_m": 0.0, "velocity_mae_mps": 0.0, "ade_m": 0.0, "fde_m": 0.0}
    return {
        "per_slot": per_slot,
        "position_mae_m": float(np.mean(np.concatenate(pos_abs_values))),
        "velocity_mae_mps": float(np.mean(np.concatenate(vel_abs_values))),
        "ade_m": float(np.mean(np.concatenate(ade_values))),
        "fde_m": float(np.mean(np.concatenate(fde_values))),
    }


def plot_window(
    out_path: Path,
    scene_name: str,
    start_frame: int,
    class_label: np.ndarray,
    target_mask: np.ndarray,
    obj_prob: np.ndarray,
    pred_state: np.ndarray,
    label_state: np.ndarray,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    ax_xy, ax_speed = axes
    colors = plt.cm.tab10(np.arange(target_mask.shape[0]))
    frames = np.arange(start_frame, start_frame + label_state.shape[1])

    for slot, is_target in enumerate(target_mask.astype(bool)):
        if not is_target:
            continue
        color = colors[slot]
        gt_name = CLASS_NAMES.get(int(class_label[slot]), str(class_label[slot]))
        label = f"slot {slot} gt={gt_name} obj={obj_prob[slot]:.2f}"
        ax_xy.plot(label_state[slot, :, 0], label_state[slot, :, 1], "o-", color=color, label=f"GT {label}")
        ax_xy.plot(pred_state[slot, :, 0], pred_state[slot, :, 1], "x--", color=color, label=f"Pred {label}")
        ax_speed.plot(frames, np.linalg.norm(label_state[slot, :, 2:], axis=1), "-", color=color, label=f"GT slot {slot}")
        ax_speed.plot(frames, np.linalg.norm(pred_state[slot, :, 2:], axis=1), "--", color=color, label=f"Pred slot {slot}")

    ax_xy.set_title(f"Future trajectory: {scene_name}, frame {start_frame}")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis("equal")
    ax_xy.legend(fontsize=8)
    ax_speed.set_title("Future speed")
    ax_speed.set_xlabel("Frame")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def stitch_predictions(
    windows: dict[str, Any],
    pred_states_norm: np.ndarray,
    obj_probs: np.ndarray,
    state_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nwin, max_targets, tout, _ = pred_states_norm.shape
    nframes = state_full.shape[1]
    pred_sum = np.zeros((max_targets, nframes, 4), dtype=np.float64)
    pred_count = np.zeros((max_targets, nframes), dtype=np.float64)
    pred_states = denormalize_state_np(pred_states_norm.reshape(-1, max_targets, tout, 4))
    starts = windows["start"].numpy()
    tin = int(windows["x"].shape[2])

    for win_idx in range(nwin):
        frames = np.arange(starts[win_idx] + tin, starts[win_idx] + tin + tout)
        pred_sum[:, frames, :] += pred_states[win_idx]
        pred_count[:, frames] += 1.0

    pred_full = np.full((max_targets, nframes, 4), np.nan, dtype=np.float32)
    covered = pred_count > 0
    pred_full[covered] = (pred_sum[covered] / pred_count[covered][:, None]).astype(np.float32)
    return pred_full, pred_count, obj_probs.mean(axis=0)


def compute_stitched_metrics(
    pred_full: np.ndarray,
    state_full: np.ndarray,
    pred_count: np.ndarray,
    target_mask: np.ndarray,
) -> dict:
    per_slot = []
    ade_values = []
    pos_values = []
    vel_values = []
    final_values = []
    for slot, is_target in enumerate(target_mask.astype(bool)):
        if not is_target:
            continue
        frame_mask = pred_count[slot] > 0
        if not np.any(frame_mask):
            continue
        diff = pred_full[slot, frame_mask] - state_full[slot, frame_mask]
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
                "final_frame_error_m": float(pos_l2[-1]),
            }
        )
        pos_values.append(np.abs(pos_diff).reshape(-1))
        vel_values.append(np.abs(vel_diff).reshape(-1))
        ade_values.append(pos_l2)
        final_values.append(np.array([pos_l2[-1]], dtype=np.float32))
    if not per_slot:
        return {"per_slot": [], "position_mae_m": 0.0, "velocity_mae_mps": 0.0, "ade_m": 0.0, "final_frame_error_m": 0.0}
    return {
        "per_slot": per_slot,
        "position_mae_m": float(np.mean(np.concatenate(pos_values))),
        "velocity_mae_mps": float(np.mean(np.concatenate(vel_values))),
        "ade_m": float(np.mean(np.concatenate(ade_values))),
        "final_frame_error_m": float(np.mean(np.concatenate(final_values))),
    }


def plot_stitched(
    out_path: Path,
    scene_name: str,
    class_label: np.ndarray,
    target_mask: np.ndarray,
    obj_prob: np.ndarray,
    pred_full: np.ndarray,
    state_full: np.ndarray,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    ax_xy, ax_speed = axes
    colors = plt.cm.tab10(np.arange(target_mask.shape[0]))
    frames = np.arange(state_full.shape[1])
    for slot, is_target in enumerate(target_mask.astype(bool)):
        if not is_target:
            continue
        color = colors[slot]
        gt_name = CLASS_NAMES.get(int(class_label[slot]), str(class_label[slot]))
        label = f"slot {slot} {gt_name} obj={obj_prob[slot]:.2f}"
        ax_xy.plot(state_full[slot, :, 0], state_full[slot, :, 1], "-", color=color, label=f"GT {label}")
        ax_xy.plot(pred_full[slot, :, 0], pred_full[slot, :, 1], "--", color=color, label=f"Pred {label}")
        ax_speed.plot(frames, np.linalg.norm(state_full[slot, :, 2:], axis=1), "-", color=color, label=f"GT slot {slot}")
        ax_speed.plot(frames, np.linalg.norm(pred_full[slot, :, 2:], axis=1), "--", color=color, label=f"Pred slot {slot}")
    ax_xy.set_title(f"Stitched scene trajectory: {scene_name}")
    ax_xy.set_xlabel("x (m)")
    ax_xy.set_ylabel("y (m)")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis("equal")
    ax_xy.legend(fontsize=8)
    ax_speed.set_title("Speed over scene")
    ax_speed.set_xlabel("Frame")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = choose_device(args.device)
    checkpoint = safe_torch_load(checkpoint_path, device)
    model = make_model(checkpoint, device)
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
    obj_probs, pred_states_norm = predict_windows(model, windows["x"], device, args.eval_batch_size)

    class_label = record["class_label"].numpy()
    target_mask = record["target_mask"].numpy()
    state_full = record["state_full"].numpy()

    if args.stitch_scene:
        pred_full, pred_count, mean_obj_prob = stitch_predictions(
            windows,
            pred_states_norm,
            obj_probs,
            state_full,
        )
        metrics = compute_stitched_metrics(pred_full, state_full, pred_count, target_mask)
        out_path = Path(args.out) if args.out else checkpoint_path.parent / "prediction_plots" / f"{scene_file.stem}_stitched.png"
        plot_stitched(out_path, scene_file.name, class_label, target_mask, mean_obj_prob, pred_full, state_full, args.show)
        covered = np.where(pred_count.max(axis=0) > 0)[0]
        summary = {
            "mode": "stitch_scene",
            "scene": scene_file.name,
            "out": str(out_path),
            "window_count": int(windows["x"].shape[0]),
            "covered_frame_start": int(covered[0]) if covered.size else None,
            "covered_frame_end": int(covered[-1]) if covered.size else None,
            "objectness_prob": mean_obj_prob.round(4).tolist(),
            "target_mask": target_mask.tolist(),
            "class_label": class_label.tolist(),
            "metrics": metrics,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.window_index < 0 or args.window_index >= windows["x"].shape[0]:
        raise IndexError(f"--window-index must be in [0, {windows['x'].shape[0] - 1}]")
    idx = args.window_index
    pred_state = denormalize_state_np(pred_states_norm[idx])
    label_state = denormalize_state_np(windows["state_label"][idx].numpy())
    metrics = compute_window_metrics(pred_state, label_state, target_mask)
    start_frame = int(windows["start"][idx].item()) + int(ckpt_args.get("tin", 16))
    out_path = Path(args.out) if args.out else checkpoint_path.parent / "prediction_plots" / f"{scene_file.stem}_window_{idx:03d}.png"
    plot_window(
        out_path,
        scene_file.name,
        start_frame,
        class_label,
        target_mask,
        obj_probs[idx],
        pred_state,
        label_state,
        args.show,
    )
    summary = {
        "mode": "window",
        "scene": scene_file.name,
        "out": str(out_path),
        "start_frame": start_frame,
        "objectness_prob": obj_probs[idx].round(4).tolist(),
        "target_mask": target_mask.tolist(),
        "class_label": class_label.tolist(),
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

