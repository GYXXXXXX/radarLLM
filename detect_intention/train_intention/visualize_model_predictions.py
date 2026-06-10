#!/usr/bin/env python3
"""Visualize intention-model predictions from a trained checkpoint.

Usage:
    python detect_intention/train_intention/visualize_model_predictions.py \
      --checkpoint detect_intention/train_intention/runs/run_xxx/best_by_intent.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .config import ACTION_BY_INTENT, INTENT_NAMES, STATE_MEAN, STATE_STD, THREAT_NAMES
    from .dataset import record_to_windows, scene_to_tensor_record
    from .mat_utils import collect_scene_files, load_scene
    from .train import choose_device, forward_model, infer_dataset_params, make_model
except ImportError:  # pragma: no cover
    from config import ACTION_BY_INTENT, INTENT_NAMES, STATE_MEAN, STATE_STD, THREAT_NAMES
    from dataset import record_to_windows, scene_to_tensor_record
    from mat_utils import collect_scene_files, load_scene
    from train import choose_device, forward_model, infer_dataset_params, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize trained target-intention predictions.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--max-scenes", type=int, default=8)
    parser.add_argument("--scene-index", type=int, default=0, help="0 means use the first selected scenes.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--protected-x", type=float, default=72.0)
    parser.add_argument("--protected-y", type=float, default=0.0)
    parser.add_argument("--protected-radius", type=float, default=10.0)
    return parser.parse_args()


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def id_to_name(mapping: dict[int, str], zero_based_id: int) -> str:
    return mapping[int(zero_based_id) + 1]


def denormalize_state_np(state: np.ndarray) -> np.ndarray:
    return state * STATE_STD.reshape(1, 1, 1, 4) + STATE_MEAN.reshape(1, 1, 1, 4)


def split_scene_files(scene_files: list[Path], train_ratio: float, split: str) -> list[Path]:
    if split == "all":
        return scene_files
    cut = int(len(scene_files) * train_ratio)
    cut = min(max(cut, 1), len(scene_files) - 1)
    if split == "train":
        return scene_files[:cut]
    return scene_files[cut:]


def build_window_record(scene_file: Path, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = load_scene(scene_file)
    record = scene_to_tensor_record(scene_file, max_targets=args.max_targets, iq_scale=args.iq_scale)
    windows = record_to_windows(record, tin=args.tin, tout=args.tout, stride=args.stride)
    return scene, windows


def move_window_to_device(windows: dict[str, Any], device: torch.device, index: int = 0) -> dict[str, torch.Tensor]:
    return {
        "x": windows["x"][index : index + 1].to(device).float(),
        "state_input": windows["state_input"][index : index + 1].to(device).float(),
        "state_label": windows["state_label"][index : index + 1].to(device).float(),
        "target_mask": windows["target_mask"][index : index + 1].to(device).float(),
        "target_class_label": windows["target_class_label"][index : index + 1].to(device).long(),
        "intent_label": windows["intent_label"][index : index + 1].to(device).long(),
        "threat_label": windows["threat_label"][index : index + 1].to(device).long(),
    }


def draw_protected_area(ax: Any, protected_point: np.ndarray, radius: float) -> None:
    theta = np.linspace(0, 2 * np.pi, 160)
    ax.plot(protected_point[0], protected_point[1], marker="*", color="#a21caf", markersize=13)
    ax.plot(
        protected_point[0] + radius * np.cos(theta),
        protected_point[1] + radius * np.sin(theta),
        linestyle=":",
        color="#a21caf",
        linewidth=1.2,
        label="protected area",
    )


def draw_scene_prediction(
    scene_file: Path,
    scene: dict[str, Any],
    windows: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    window_index: int,
    output_path: Path,
    protected_point: np.ndarray,
    protected_radius: float,
    dpi: int,
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    start = int(windows["start"][window_index].item())
    tin = int(windows["x"].shape[2])
    tout = int(windows["state_label"].shape[2])
    observed_slice = slice(start, start + tin)
    future_slice = slice(start + tin, start + tin + tout)

    target_mask = windows["target_mask"][window_index].numpy()
    true_intent = windows["intent_label"][window_index].numpy()
    true_threat = windows["threat_label"][window_index].numpy()
    true_future = denormalize_state_np(windows["state_label"][window_index : window_index + 1].numpy())[0]
    pred_future = denormalize_state_np(outputs["state_pred"].detach().cpu().numpy())[0]
    intent_prob = F.softmax(outputs["intent_logits"], dim=-1).detach().cpu().numpy()[0]
    threat_prob = F.softmax(outputs["threat_logits"], dim=-1).detach().cpu().numpy()[0]

    gt = scene["gt"]
    pos = np.asarray(gt["pos"], dtype=np.float32)
    n_targets = int(target_mask.sum())

    fig, ax = plt.subplots(figsize=(10.5, 8.0), constrained_layout=True)
    draw_protected_area(ax, protected_point, protected_radius)

    colors = ["#2563eb", "#dc2626", "#059669", "#d97706"]
    records: list[dict[str, Any]] = []

    for slot in range(len(target_mask)):
        if target_mask[slot] <= 0.5:
            continue

        color = colors[slot % len(colors)]
        full_xy = pos[slot]
        obs_xy = full_xy[observed_slice]
        true_xy = true_future[slot, :, :2]
        pred_xy = pred_future[slot, :, :2]

        pred_intent_id0 = int(intent_prob[slot].argmax())
        pred_threat_id0 = int(threat_prob[slot].argmax())
        true_intent_id0 = int(true_intent[slot])
        true_threat_id0 = int(true_threat[slot])
        pred_intent_name = id_to_name(INTENT_NAMES, pred_intent_id0)
        pred_threat_name = id_to_name(THREAT_NAMES, pred_threat_id0)
        true_intent_name = id_to_name(INTENT_NAMES, true_intent_id0)
        true_threat_name = id_to_name(THREAT_NAMES, true_threat_id0)
        action = ACTION_BY_INTENT[pred_intent_name]

        ax.plot(full_xy[:, 0], full_xy[:, 1], color=color, alpha=0.18, linewidth=1.0)
        ax.plot(obs_xy[:, 0], obs_xy[:, 1], "-o", color=color, linewidth=2.0, markersize=3)
        ax.plot(true_xy[:, 0], true_xy[:, 1], "--", color=color, linewidth=2.0, label=f"T{slot + 1} true future")
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "-.", color=color, linewidth=2.5, label=f"T{slot + 1} predicted")
        ax.scatter(obs_xy[0, 0], obs_xy[0, 1], marker="o", color=color, edgecolor="white", s=55)
        ax.scatter(pred_xy[-1, 0], pred_xy[-1, 1], marker="x", color=color, s=80, linewidth=2.0)

        label = (
            f"T{slot + 1}: pred={pred_intent_name} ({intent_prob[slot, pred_intent_id0]:.2f}), "
            f"true={true_intent_name}\n"
            f"threat={pred_threat_name} ({threat_prob[slot, pred_threat_id0]:.2f}), action={action}"
        )
        ax.text(
            obs_xy[-1, 0],
            obs_xy[-1, 1],
            "  " + label,
            fontsize=8,
            color=color,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.86},
        )

        records.append(
            {
                "slot": slot + 1,
                "pred_intent": pred_intent_name,
                "pred_intent_confidence": float(intent_prob[slot, pred_intent_id0]),
                "true_intent": true_intent_name,
                "pred_threat": pred_threat_name,
                "pred_threat_confidence": float(threat_prob[slot, pred_threat_id0]),
                "true_threat": true_threat_name,
                "recommended_action": action,
                "pred_final_xy": pred_xy[-1].tolist(),
                "true_final_xy": true_xy[-1].tolist(),
            }
        )

    ax.set_title(
        f"{scene_file.name} | window start={start + 1}, Tin={tin}, Tout={tout}, targets={n_targets}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.axis("equal")
    ax.legend(frameon=False, fontsize=8, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return records


def main() -> None:
    args = parse_args()
    checkpoint = safe_torch_load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**checkpoint["args"])
    if not hasattr(train_args, "input_mode"):
        train_args.input_mode = "radar"
    if args.dataset_dir:
        train_args.dataset_dir = args.dataset_dir

    scene_files = collect_scene_files(train_args.dataset_dir)
    infer_dataset_params(train_args, scene_files)
    selected = split_scene_files(scene_files, train_args.train_ratio, args.split)
    if args.scene_index > 0:
        selected = selected[args.scene_index - 1 :]
    selected = selected[: args.max_scenes]
    if not selected:
        raise ValueError("No scenes selected.")

    device = choose_device(args.device)
    model = make_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent / "prediction_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    protected_point = np.array([args.protected_x, args.protected_y], dtype=np.float32)

    all_records = []
    with torch.no_grad():
        for scene_file in selected:
            scene, windows = build_window_record(scene_file, train_args)
            batch = move_window_to_device(windows, device, index=0)
            outputs = forward_model(model, batch, train_args.input_mode)
            image_path = out_dir / f"{scene_file.stem}_prediction.png"
            records = draw_scene_prediction(
                scene_file,
                scene,
                windows,
                outputs,
                window_index=0,
                output_path=image_path,
                protected_point=protected_point,
                protected_radius=args.protected_radius,
                dpi=args.dpi,
            )
            all_records.append(
                {
                    "scene_file": scene_file.name,
                    "image": str(image_path),
                    "records": records,
                }
            )
            print(f"Saved: {image_path}")

    summary_path = out_dir / "prediction_visualization_summary.json"
    summary_path.write_text(json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

