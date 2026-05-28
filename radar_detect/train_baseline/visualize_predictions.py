#!/usr/bin/env python3
"""Visualize fixed-slot FMCW trajectory predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'matplotlib'. Install with: pip install matplotlib"
    ) from exc

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'torch'. Install train_baseline requirements.") from exc

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .config import STATE_MEAN, STATE_STD
    from .dataset import FmcwTrajectoryDataset
    from .model import FmcwBaseline3DCNN
    from .utils import collect_scene_files
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import STATE_MEAN, STATE_STD
    from dataset import FmcwTrajectoryDataset
    from model import FmcwBaseline3DCNN
    from utils import collect_scene_files


CLASS_NAMES = {
    0: "background",
    1: "T1_slow_smooth",
    2: "T2_uav_agile",
    3: "T3_pedestrian",
    4: "T4_fast_maneuver",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot predicted vs. ground-truth future trajectories."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="radar_detect/fmcw_traj_dataset_2000scenes",
        help="Directory containing scene_*.mat files.",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Zero-based index into the sorted scene_*.mat list.",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Zero-based sliding-window index within the selected scene.",
    )
    parser.add_argument(
        "--stitch-scene",
        action="store_true",
        help="Run every window in the selected scene and average overlapping future predictions.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="", help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show an interactive plot window.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def denormalize_state_np(state: np.ndarray) -> np.ndarray:
    return state * STATE_STD.reshape(1, 1, 4) + STATE_MEAN.reshape(1, 1, 4)


def make_model(checkpoint: dict, device: torch.device) -> FmcwBaseline3DCNN:
    ckpt_args = checkpoint.get("args", {})
    model = FmcwBaseline3DCNN(
        in_channels=16,
        max_targets=int(ckpt_args.get("max_targets", 4)),
        num_classes=int(ckpt_args.get("num_classes", 5)),
        tout=int(ckpt_args.get("tout", 8)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def build_dataset(scene_file: Path, checkpoint: dict) -> FmcwTrajectoryDataset:
    ckpt_args = checkpoint.get("args", {})
    return FmcwTrajectoryDataset(
        [scene_file],
        tin=int(ckpt_args.get("tin", 16)),
        tout=int(ckpt_args.get("tout", 8)),
        stride=int(ckpt_args.get("stride", 1)),
        max_targets=int(ckpt_args.get("max_targets", 4)),
        iq_scale=float(ckpt_args.get("iq_scale", 76.0)),
        cache_scenes=True,
        max_cache_scenes=1,
    )


def plot_prediction(
    out_path: Path,
    sample: dict,
    pred_class: np.ndarray,
    pred_prob: np.ndarray,
    pred_state: np.ndarray,
    label_state: np.ndarray,
    show: bool,
) -> None:
    target_mask = sample["target_mask"].numpy().astype(bool)
    class_label = sample["class_label"].numpy().astype(int)
    start = int(sample["start"])
    scene_name = str(sample["scene_file"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    ax_xy, ax_speed = axes

    colors = plt.cm.tab10(np.arange(len(target_mask)))
    tin = int(sample["x"].shape[1])
    future_frames = np.arange(start + tin, start + tin + label_state.shape[1])

    for slot, is_target in enumerate(target_mask):
        if not is_target:
            continue
        color = colors[slot]
        gt_xy = label_state[slot, :, :2]
        pred_xy = pred_state[slot, :, :2]
        gt_vel = label_state[slot, :, 2:]
        pred_vel = pred_state[slot, :, 2:]
        gt_speed = np.linalg.norm(gt_vel, axis=1)
        pred_speed = np.linalg.norm(pred_vel, axis=1)

        gt_name = CLASS_NAMES.get(int(class_label[slot]), str(class_label[slot]))
        pred_name = CLASS_NAMES.get(int(pred_class[slot]), str(pred_class[slot]))
        label = (
            f"slot {slot} gt={gt_name} pred={pred_name} "
            f"p={pred_prob[slot]:.2f}"
        )

        ax_xy.plot(gt_xy[:, 0], gt_xy[:, 1], "o-", color=color, label=f"GT {label}")
        ax_xy.plot(pred_xy[:, 0], pred_xy[:, 1], "x--", color=color, label=f"Pred {label}")
        ax_xy.scatter(gt_xy[0, 0], gt_xy[0, 1], marker="s", color=color, s=70)

        ax_speed.plot(future_frames, gt_speed, "-", color=color, label=f"GT slot {slot}")
        ax_speed.plot(future_frames, pred_speed, "--", color=color, label=f"Pred slot {slot}")

    ax_xy.set_title(f"Future trajectory: {scene_name}, start={start}")
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


def run_sample_prediction(
    model: FmcwBaseline3DCNN,
    sample: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = sample["x"].unsqueeze(0).to(device).float()
    with torch.no_grad():
        class_logits, state_pred = model(x)
        prob = torch.softmax(class_logits, dim=-1)

    pred_class = class_logits.argmax(dim=-1).squeeze(0).cpu().numpy().astype(int)
    prob_full = prob.squeeze(0).cpu().numpy()
    pred_prob = prob_full.max(axis=-1)
    pred_state = denormalize_state_np(state_pred.squeeze(0).cpu().numpy())
    return pred_class, pred_prob, pred_state, prob_full


def compute_sample_metrics(
    pred_state: np.ndarray,
    label_state: np.ndarray,
    target_mask: np.ndarray,
) -> dict:
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

        pos_mae = float(np.mean(np.abs(pos_diff)))
        vel_mae = float(np.mean(np.abs(vel_diff)))
        ade = float(np.mean(pos_l2))
        fde = float(pos_l2[-1])

        per_slot.append(
            {
                "slot": int(slot),
                "position_mae_m": pos_mae,
                "velocity_mae_mps": vel_mae,
                "ade_m": ade,
                "fde_m": fde,
            }
        )
        pos_abs_values.append(np.abs(pos_diff).reshape(-1))
        vel_abs_values.append(np.abs(vel_diff).reshape(-1))
        ade_values.append(pos_l2)
        fde_values.append(np.array([pos_l2[-1]], dtype=np.float32))

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
        "position_mae_m": float(np.mean(np.concatenate(pos_abs_values))),
        "velocity_mae_mps": float(np.mean(np.concatenate(vel_abs_values))),
        "ade_m": float(np.mean(np.concatenate(ade_values))),
        "fde_m": float(np.mean(np.concatenate(fde_values))),
    }


def scene_ground_truth(dataset: FmcwTrajectoryDataset, scene_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene = dataset._load_scene(scene_file)
    gt = scene["gt"]
    is_target = np.asarray(gt["isTarget"], dtype=bool).reshape(-1)
    target_idx = np.where(is_target)[0]
    class_ids = np.asarray(gt["targetClassId"], dtype=np.int64).reshape(-1)
    pos = np.asarray(gt["pos"], dtype=np.float32)
    vel = np.asarray(gt["vel"], dtype=np.float32)

    max_targets = dataset.max_targets
    nframes = pos.shape[1]
    state = np.full((max_targets, nframes, 4), np.nan, dtype=np.float32)
    class_label = np.zeros((max_targets,), dtype=np.int64)
    target_mask = np.zeros((max_targets,), dtype=bool)

    for slot, obj_idx in enumerate(target_idx[:max_targets]):
        state[slot] = np.concatenate([pos[obj_idx], vel[obj_idx]], axis=-1)
        class_label[slot] = int(class_ids[obj_idx])
        target_mask[slot] = True

    return state, class_label, target_mask


def stitch_scene_predictions(
    dataset: FmcwTrajectoryDataset,
    scene_file: Path,
    model: FmcwBaseline3DCNN,
    device: torch.device,
) -> dict:
    gt_state, class_label, target_mask = scene_ground_truth(dataset, scene_file)
    max_targets, nframes, state_dim = gt_state.shape

    pred_sum = np.zeros((max_targets, nframes, state_dim), dtype=np.float64)
    pred_count = np.zeros((max_targets, nframes), dtype=np.float64)
    prob_sum = None
    prob_count = 0

    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        _, _, pred_state, prob = run_sample_prediction(model, sample, device)

        if prob_sum is None:
            prob_sum = np.zeros_like(prob, dtype=np.float64)
        prob_sum += prob
        prob_count += 1

        start = int(sample["start"])
        tin = int(sample["x"].shape[1])
        tout = pred_state.shape[1]
        frames = np.arange(start + tin, start + tin + tout)
        pred_sum[:, frames, :] += pred_state
        pred_count[:, frames] += 1.0

    pred_state_full = np.full_like(gt_state, np.nan, dtype=np.float32)
    covered = pred_count > 0
    pred_state_full[covered] = (pred_sum[covered] / pred_count[covered][:, None]).astype(np.float32)

    if prob_sum is None or prob_count == 0:
        mean_prob = np.zeros((max_targets, 1), dtype=np.float32)
    else:
        mean_prob = (prob_sum / prob_count).astype(np.float32)
    pred_class = mean_prob.argmax(axis=-1).astype(int)
    pred_prob = mean_prob.max(axis=-1)

    return {
        "gt_state": gt_state,
        "class_label": class_label,
        "target_mask": target_mask,
        "pred_state": pred_state_full,
        "pred_count": pred_count,
        "pred_class": pred_class,
        "pred_prob": pred_prob,
        "window_count": len(dataset),
    }


def compute_stitched_metrics(
    pred_state: np.ndarray,
    gt_state: np.ndarray,
    pred_count: np.ndarray,
    target_mask: np.ndarray,
) -> dict:
    per_slot = []
    pos_abs_values = []
    vel_abs_values = []
    ade_values = []
    final_errors = []

    for slot, is_target in enumerate(target_mask.astype(bool)):
        if not is_target:
            continue
        frame_mask = pred_count[slot] > 0
        if not np.any(frame_mask):
            continue

        diff = pred_state[slot, frame_mask] - gt_state[slot, frame_mask]
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

        pos_abs_values.append(np.abs(pos_diff).reshape(-1))
        vel_abs_values.append(np.abs(vel_diff).reshape(-1))
        ade_values.append(pos_l2)
        final_errors.append(np.array([pos_l2[-1]], dtype=np.float32))

    if not per_slot:
        return {
            "per_slot": [],
            "position_mae_m": 0.0,
            "velocity_mae_mps": 0.0,
            "ade_m": 0.0,
            "final_frame_error_m": 0.0,
        }

    return {
        "per_slot": per_slot,
        "position_mae_m": float(np.mean(np.concatenate(pos_abs_values))),
        "velocity_mae_mps": float(np.mean(np.concatenate(vel_abs_values))),
        "ade_m": float(np.mean(np.concatenate(ade_values))),
        "final_frame_error_m": float(np.mean(np.concatenate(final_errors))),
    }


def plot_stitched_scene(
    out_path: Path,
    scene_name: str,
    stitched: dict,
    show: bool,
) -> None:
    gt_state = stitched["gt_state"]
    pred_state = stitched["pred_state"]
    pred_count = stitched["pred_count"]
    class_label = stitched["class_label"]
    target_mask = stitched["target_mask"]
    pred_class = stitched["pred_class"]
    pred_prob = stitched["pred_prob"]

    nframes = gt_state.shape[1]
    frames = np.arange(nframes)
    colors = plt.cm.tab10(np.arange(gt_state.shape[0]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    ax_xy, ax_speed = axes

    for slot, is_target in enumerate(target_mask.astype(bool)):
        if not is_target:
            continue

        color = colors[slot]
        gt_xy = gt_state[slot, :, :2]
        pred_xy = pred_state[slot, :, :2]
        gt_speed = np.linalg.norm(gt_state[slot, :, 2:], axis=1)
        pred_speed = np.linalg.norm(pred_state[slot, :, 2:], axis=1)
        covered = pred_count[slot] > 0

        gt_name = CLASS_NAMES.get(int(class_label[slot]), str(class_label[slot]))
        pred_name = CLASS_NAMES.get(int(pred_class[slot]), str(pred_class[slot]))

        ax_xy.plot(gt_xy[:, 0], gt_xy[:, 1], "-", color=color, label=f"GT slot {slot} {gt_name}")
        ax_xy.plot(
            pred_xy[:, 0],
            pred_xy[:, 1],
            "--",
            color=color,
            label=f"Pred slot {slot} {pred_name} p={pred_prob[slot]:.2f}",
        )
        ax_xy.scatter(gt_xy[0, 0], gt_xy[0, 1], marker="s", color=color, s=60)
        if np.any(covered):
            first = int(np.where(covered)[0][0])
            ax_xy.scatter(pred_xy[first, 0], pred_xy[first, 1], marker="x", color=color, s=70)

        ax_speed.plot(frames, gt_speed, "-", color=color, label=f"GT slot {slot}")
        ax_speed.plot(frames, pred_speed, "--", color=color, label=f"Pred slot {slot}")

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

    scene_files = collect_scene_files(args.dataset_dir)
    if args.scene_index < 0 or args.scene_index >= len(scene_files):
        raise IndexError(f"--scene-index must be in [0, {len(scene_files) - 1}]")

    scene_file = scene_files[args.scene_index]
    dataset = build_dataset(scene_file, checkpoint)
    model = make_model(checkpoint, device)

    if args.stitch_scene:
        stitched = stitch_scene_predictions(dataset, scene_file, model, device)
        metrics = compute_stitched_metrics(
            stitched["pred_state"],
            stitched["gt_state"],
            stitched["pred_count"],
            stitched["target_mask"],
        )

        if args.out:
            out_path = Path(args.out)
        else:
            out_path = checkpoint_path.parent / "prediction_plots" / f"{scene_file.stem}_stitched.png"

        plot_stitched_scene(out_path, scene_file.name, stitched, show=args.show)

        covered_counts = stitched["pred_count"].max(axis=0)
        covered_frames = np.where(covered_counts > 0)[0]
        summary = {
            "mode": "stitch_scene",
            "scene": scene_file.name,
            "out": str(out_path),
            "window_count": stitched["window_count"],
            "covered_frame_start": int(covered_frames[0]) if covered_frames.size else None,
            "covered_frame_end": int(covered_frames[-1]) if covered_frames.size else None,
            "class_label": stitched["class_label"].tolist(),
            "pred_class": stitched["pred_class"].tolist(),
            "pred_prob": stitched["pred_prob"].round(4).tolist(),
            "target_mask": stitched["target_mask"].astype(float).tolist(),
            "metrics": metrics,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.window_index < 0 or args.window_index >= len(dataset):
        raise IndexError(f"--window-index must be in [0, {len(dataset) - 1}] for {scene_file.name}")

    sample = dataset[args.window_index]
    pred_class, pred_prob, pred_state, _ = run_sample_prediction(model, sample, device)
    label_state = denormalize_state_np(sample["state_label"].numpy())
    sample_metrics = compute_sample_metrics(
        pred_state,
        label_state,
        sample["target_mask"].numpy(),
    )

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            checkpoint_path.parent
            / "prediction_plots"
            / f"{scene_file.stem}_window_{args.window_index:03d}.png"
        )

    plot_prediction(
        out_path,
        sample,
        pred_class,
        pred_prob,
        pred_state,
        label_state,
        show=args.show,
    )

    summary = {
        "mode": "window",
        "scene": scene_file.name,
        "start": int(sample["start"]),
        "out": str(out_path),
        "class_label": sample["class_label"].tolist(),
        "pred_class": pred_class.tolist(),
        "pred_prob": pred_prob.round(4).tolist(),
        "target_mask": sample["target_mask"].tolist(),
        "metrics": sample_metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
