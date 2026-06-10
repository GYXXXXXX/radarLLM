#!/usr/bin/env python3
"""Evaluate an intention checkpoint and export structured target predictions."""

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
    from .config import ACTION_BY_INTENT, INTENT_NAMES, TARGET_CLASS_NAMES, THREAT_NAMES
    from .mat_utils import collect_scene_files, sanitize_for_json
    from .train import (
        apply_scene_limits,
        batch_metrics,
        choose_device,
        compute_loss,
        denormalize_state,
        empty_accumulator,
        finalize_metrics,
        forward_model,
        make_loader,
        make_model,
        split_scenes,
    )
except ImportError:  # pragma: no cover
    from config import ACTION_BY_INTENT, INTENT_NAMES, TARGET_CLASS_NAMES, THREAT_NAMES
    from mat_utils import collect_scene_files, sanitize_for_json
    from train import (
        apply_scene_limits,
        batch_metrics,
        choose_device,
        compute_loss,
        denormalize_state,
        empty_accumulator,
        finalize_metrics,
        forward_model,
        make_loader,
        make_model,
        split_scenes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FMCW intention checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--scene-batch-size", type=int, default=1)
    parser.add_argument("--windows-per-scene", type=int, default=0)
    parser.add_argument("--objectness-threshold", type=float, default=0.5)
    parser.add_argument("--protected-x", type=float, default=72.0)
    parser.add_argument("--protected-y", type=float, default=0.0)
    parser.add_argument("--include-trajectories", action="store_true")
    return parser.parse_args()


def safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def update_accumulator(
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


def id_to_name(mapping: dict[int, str], zero_based_id: int) -> str:
    return mapping[int(zero_based_id) + 1]


def trend_label(delta: float, eps: float = 1.0) -> str:
    if delta < -eps:
        return "decreasing"
    if delta > eps:
        return "increasing"
    return "stable"


def state_summary(
    observed_state: np.ndarray,
    predicted_state: np.ndarray,
    protected_point: np.ndarray,
    include_trajectories: bool,
) -> dict[str, Any]:
    obs_xy = observed_state[:, :2]
    pred_xy = predicted_state[:, :2]
    obs_speed = np.linalg.norm(observed_state[:, 2:], axis=-1)
    pred_speed = np.linalg.norm(predicted_state[:, 2:], axis=-1)
    obs_range = np.linalg.norm(obs_xy, axis=-1)
    pred_range = np.linalg.norm(pred_xy, axis=-1)
    obs_protected_dist = np.linalg.norm(obs_xy - protected_point[None, :], axis=-1)
    pred_protected_dist = np.linalg.norm(pred_xy - protected_point[None, :], axis=-1)

    summary: dict[str, Any] = {
        "observed_range_start_m": float(obs_range[0]),
        "observed_range_end_m": float(obs_range[-1]),
        "observed_range_delta_m": float(obs_range[-1] - obs_range[0]),
        "observed_range_trend": trend_label(float(obs_range[-1] - obs_range[0])),
        "observed_speed_mean_mps": float(obs_speed.mean()),
        "observed_protected_distance_start_m": float(obs_protected_dist[0]),
        "observed_protected_distance_end_m": float(obs_protected_dist[-1]),
        "observed_protected_distance_delta_m": float(obs_protected_dist[-1] - obs_protected_dist[0]),
        "observed_protected_distance_trend": trend_label(float(obs_protected_dist[-1] - obs_protected_dist[0])),
        "predicted_range_final_m": float(pred_range[-1]),
        "predicted_range_delta_m": float(pred_range[-1] - obs_range[-1]),
        "predicted_range_trend": trend_label(float(pred_range[-1] - obs_range[-1])),
        "predicted_speed_mean_mps": float(pred_speed.mean()),
        "predicted_protected_distance_final_m": float(pred_protected_dist[-1]),
        "predicted_protected_distance_delta_m": float(pred_protected_dist[-1] - obs_protected_dist[-1]),
        "predicted_protected_distance_trend": trend_label(
            float(pred_protected_dist[-1] - obs_protected_dist[-1])
        ),
    }
    if include_trajectories:
        summary["observed_state"] = observed_state.tolist()
        summary["predicted_state"] = predicted_state.tolist()
    return summary


def confusion_matrix(records: list[dict[str, Any]], true_key: str, pred_key: str, nclass: int) -> list[list[int]]:
    mat = np.zeros((nclass, nclass), dtype=np.int64)
    for item in records:
        true_id = item.get(true_key)
        pred_id = item.get(pred_key)
        if true_id is None or pred_id is None:
            continue
        mat[int(true_id) - 1, int(pred_id) - 1] += 1
    return mat.tolist()


def make_prediction_records(
    raw_batch: dict[str, Any],
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    objectness_prob = torch.sigmoid(outputs["objectness_logits"]).detach().cpu().numpy()
    class_prob = F.softmax(outputs["target_class_logits"], dim=-1).detach().cpu().numpy()
    intent_prob = F.softmax(outputs["intent_logits"], dim=-1).detach().cpu().numpy()
    threat_prob = F.softmax(outputs["threat_logits"], dim=-1).detach().cpu().numpy()
    state_pred = denormalize_state(outputs["state_pred"]).detach().cpu().numpy()

    target_mask = batch["target_mask"].detach().cpu().numpy()
    true_class = batch["target_class_label"].detach().cpu().numpy()
    true_intent = batch["intent_label"].detach().cpu().numpy()
    true_threat = batch["threat_label"].detach().cpu().numpy()
    state_input = raw_batch["state_input"].numpy()
    starts = raw_batch["start"].numpy()
    protected_point = np.array([args.protected_x, args.protected_y], dtype=np.float32)

    records: list[dict[str, Any]] = []
    bsz, max_targets = objectness_prob.shape
    for bidx in range(bsz):
        for slot in range(max_targets):
            if target_mask[bidx, slot] <= 0.5:
                continue

            pred_class_zero = int(class_prob[bidx, slot].argmax())
            pred_intent_zero = int(intent_prob[bidx, slot].argmax())
            pred_threat_zero = int(threat_prob[bidx, slot].argmax())
            pred_intent_name = id_to_name(INTENT_NAMES, pred_intent_zero)

            record = {
                "scene_file": raw_batch["scene_file"][bidx],
                "start_frame": int(starts[bidx]) + 1,
                "slot": slot + 1,
                "objectness_prob": float(objectness_prob[bidx, slot]),
                "pred_target_class_id": pred_class_zero + 1,
                "pred_target_class_name": id_to_name(TARGET_CLASS_NAMES, pred_class_zero),
                "pred_target_class_confidence": float(class_prob[bidx, slot, pred_class_zero]),
                "pred_intent_id": pred_intent_zero + 1,
                "pred_intent_name": pred_intent_name,
                "pred_intent_confidence": float(intent_prob[bidx, slot, pred_intent_zero]),
                "pred_threat_level": pred_threat_zero + 1,
                "pred_threat_name": id_to_name(THREAT_NAMES, pred_threat_zero),
                "pred_threat_confidence": float(threat_prob[bidx, slot, pred_threat_zero]),
                "recommended_action_rule": ACTION_BY_INTENT[pred_intent_name],
                "target_present_label": bool(target_mask[bidx, slot] > 0.5),
            }
            if target_mask[bidx, slot] > 0.5:
                record.update(
                    {
                        "true_target_class_id": int(true_class[bidx, slot]) + 1,
                        "true_target_class_name": id_to_name(TARGET_CLASS_NAMES, int(true_class[bidx, slot])),
                        "true_intent_id": int(true_intent[bidx, slot]) + 1,
                        "true_intent_name": id_to_name(INTENT_NAMES, int(true_intent[bidx, slot])),
                        "true_threat_level": int(true_threat[bidx, slot]) + 1,
                        "true_threat_name": id_to_name(THREAT_NAMES, int(true_threat[bidx, slot])),
                    }
                )
            record.update(
                state_summary(
                    state_input[bidx, slot],
                    state_pred[bidx, slot],
                    protected_point,
                    include_trajectories=args.include_trajectories,
                )
            )
            records.append(record)
    return records


def main() -> None:
    args = parse_args()
    checkpoint = safe_torch_load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**checkpoint["args"])
    if not hasattr(train_args, "nrx"):
        train_args.nrx = 8
    if not hasattr(train_args, "input_mode"):
        train_args.input_mode = "radar"
    if args.dataset_dir:
        train_args.dataset_dir = args.dataset_dir
    train_args.scene_batch_size = args.scene_batch_size
    train_args.val_scene_batch_size = args.scene_batch_size
    train_args.val_windows_per_scene = args.windows_per_scene
    train_args.windows_per_scene = args.windows_per_scene
    train_args.no_pin_memory = False
    train_args.num_workers = 0
    train_args.prefetch_factor = 1
    train_args.objectness_threshold = args.objectness_threshold

    scene_files = collect_scene_files(train_args.dataset_dir)
    train_scenes, val_scenes = split_scenes(scene_files, train_args.train_ratio)
    if args.split == "train":
        selected = train_scenes
    elif args.split == "val":
        selected = val_scenes
    else:
        selected = scene_files
    if args.max_scenes:
        selected = selected[: args.max_scenes]
    if not selected:
        raise ValueError("No scenes selected for evaluation.")

    loader = make_loader(
        selected,
        train_args,
        shuffle=False,
        windows_per_scene=args.windows_per_scene,
        random_windows=False,
        scene_batch_size=args.scene_batch_size,
    )

    device = choose_device(args.device)
    model = make_model(train_args).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "eval_metrics.json"

    acc = empty_accumulator()
    all_records: list[dict[str, Any]] = []
    with pred_path.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for raw_batch in loader:
                batch = {
                    "x": raw_batch["x"].to(device).float(),
                    "state_input": raw_batch["state_input"].to(device).float(),
                    "state_label": raw_batch["state_label"].to(device).float(),
                    "target_mask": raw_batch["target_mask"].to(device).float(),
                    "target_class_label": raw_batch["target_class_label"].to(device).long(),
                    "intent_label": raw_batch["intent_label"].to(device).long(),
                    "threat_label": raw_batch["threat_label"].to(device).long(),
                }
                outputs = forward_model(model, batch, train_args.input_mode)
                loss_total, loss_parts = compute_loss(outputs, batch, train_args)
                metric_parts = batch_metrics(outputs, batch, args.objectness_threshold)
                update_accumulator(acc, loss_total, loss_parts, metric_parts, batch_size=batch["x"].shape[0])

                records = make_prediction_records(raw_batch, batch, outputs, args)
                for record in records:
                    all_records.append(record)
                    handle.write(json.dumps(sanitize_for_json(record), ensure_ascii=False) + "\n")

    metrics = finalize_metrics(acc)
    metrics.update(
        {
            "checkpoint": str(args.checkpoint),
            "dataset_dir": str(train_args.dataset_dir),
            "split": args.split,
            "scene_count": len(selected),
            "prediction_count": len(all_records),
            "intent_confusion_matrix": confusion_matrix(
                all_records, "true_intent_id", "pred_intent_id", len(INTENT_NAMES)
            ),
            "target_class_confusion_matrix": confusion_matrix(
                all_records, "true_target_class_id", "pred_target_class_id", len(TARGET_CLASS_NAMES)
            ),
            "threat_confusion_matrix": confusion_matrix(
                all_records, "true_threat_level", "pred_threat_level", len(THREAT_NAMES)
            ),
        }
    )
    metrics_path.write_text(json.dumps(sanitize_for_json(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(sanitize_for_json(metrics), ensure_ascii=False, indent=2))
    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
