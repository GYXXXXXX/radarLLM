"""Scene-first sliding-window dataset for radar intention prediction."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .config import STATE_MEAN, STATE_STD
    from .mat_utils import load_scene, read_scene_nframes
except ImportError:  # pragma: no cover
    from config import STATE_MEAN, STATE_STD
    from mat_utils import load_scene, read_scene_nframes


def make_full_input(iq: np.ndarray, iq_scale: float) -> torch.Tensor:
    """Convert [Nfast, Nchirp, Nrx, Nframes] IQ to [2*Nrx, Nframes, Nchirp, Nfast]."""
    if iq.ndim != 4:
        raise ValueError(f"iq must be 4-D, got {iq.shape}")

    nfast, nchirp, nrx, nframes = iq.shape
    iq = iq / float(iq_scale)
    real = np.real(iq).astype(np.float32, copy=False)
    imag = np.imag(iq).astype(np.float32, copy=False)

    x_scene = np.empty((2 * nrx, nframes, nchirp, nfast), dtype=np.float32)
    for rx in range(nrx):
        x_scene[2 * rx] = np.transpose(real[:, :, rx, :], (2, 1, 0))
        x_scene[2 * rx + 1] = np.transpose(imag[:, :, rx, :], (2, 1, 0))
    return torch.from_numpy(x_scene)


def make_full_labels(gt: dict[str, Any], max_targets: int) -> dict[str, torch.Tensor]:
    is_target = np.asarray(gt["isTarget"], dtype=bool).reshape(-1)
    target_idx = np.where(is_target)[0]
    if target_idx.size > max_targets:
        raise ValueError(f"Scene has {target_idx.size} targets, max_targets={max_targets}")

    class_ids = np.asarray(gt["targetClassId"], dtype=np.int64).reshape(-1)
    intent_ids = np.asarray(gt["intentId"], dtype=np.int64).reshape(-1)
    threat_levels = np.asarray(gt["threatLevel"], dtype=np.int64).reshape(-1)
    response_action_ids = np.asarray(gt["responseActionId"], dtype=np.int64).reshape(-1)
    pos = np.asarray(gt["pos"], dtype=np.float32)
    vel = np.asarray(gt["vel"], dtype=np.float32)
    nframes = pos.shape[1]

    target_class_label = np.zeros((max_targets,), dtype=np.int64)
    intent_label = np.zeros((max_targets,), dtype=np.int64)
    threat_label = np.zeros((max_targets,), dtype=np.int64)
    response_action_label = np.zeros((max_targets,), dtype=np.int64)
    state_full = np.zeros((max_targets, nframes, 4), dtype=np.float32)
    target_mask = np.zeros((max_targets,), dtype=np.float32)

    for slot, obj_idx in enumerate(target_idx):
        class_id = int(class_ids[obj_idx])
        intent_id = int(intent_ids[obj_idx])
        threat_level = int(threat_levels[obj_idx])
        response_action_id = int(response_action_ids[obj_idx])
        if class_id < 1 or class_id > 4:
            raise ValueError(f"targetClassId must be 1..4, got {class_id}")
        if intent_id < 1 or intent_id > 5:
            raise ValueError(f"intentId must be 1..5, got {intent_id}")
        if threat_level < 1 or threat_level > 4:
            raise ValueError(f"threatLevel must be 1..4, got {threat_level}")

        state_full[slot] = np.concatenate([pos[obj_idx], vel[obj_idx]], axis=-1)
        target_class_label[slot] = class_id - 1
        intent_label[slot] = intent_id - 1
        threat_label[slot] = threat_level - 1
        response_action_label[slot] = max(response_action_id - 1, 0)
        target_mask[slot] = 1.0

    return {
        "target_class_label": torch.from_numpy(target_class_label),
        "intent_label": torch.from_numpy(intent_label),
        "threat_label": torch.from_numpy(threat_label),
        "response_action_label": torch.from_numpy(response_action_label),
        "state_full": torch.from_numpy(state_full),
        "target_mask": torch.from_numpy(target_mask),
    }


def scene_to_tensor_record(
    scene_file: str | Path,
    max_targets: int,
    iq_scale: float,
) -> dict[str, Any]:
    scene_file = Path(scene_file)
    scene = load_scene(scene_file)
    labels = make_full_labels(scene["gt"], max_targets)
    labels.update(
        {
            "scene_file": scene_file.name,
            "x_scene": make_full_input(scene["iq"], iq_scale),
        }
    )
    return labels


def record_to_windows(
    record: dict[str, Any],
    tin: int,
    tout: int,
    stride: int,
    window_indices: list[int] | torch.Tensor | None = None,
) -> dict[str, Any]:
    x_scene = record["x_scene"].float()
    state_full = record["state_full"].float()
    target_mask = record["target_mask"].float()

    nframes = int(x_scene.shape[1])
    last_start = nframes - int(tin) - int(tout)
    if last_start < 0:
        raise ValueError(f"Scene {record['scene_file']} is too short for Tin/Tout.")

    all_starts = list(range(0, last_start + 1, int(stride)))
    if window_indices is None:
        starts = all_starts
    else:
        requested = [int(item) for item in window_indices]
        starts = []
        for window_index in requested:
            if window_index < 0 or window_index >= len(all_starts):
                raise IndexError(f"window index {window_index} out of range 0..{len(all_starts) - 1}")
            starts.append(all_starts[window_index])

    mean = torch.as_tensor(STATE_MEAN, dtype=torch.float32).view(1, 1, 4)
    std = torch.as_tensor(STATE_STD, dtype=torch.float32).view(1, 1, 4)

    x_windows = []
    state_input_windows = []
    state_label_windows = []
    for start in starts:
        x_windows.append(x_scene[:, start : start + tin])
        state_input_windows.append(state_full[:, start : start + tin, :])
        pred_slice = slice(start + tin, start + tin + tout)
        state_label_windows.append((state_full[:, pred_slice, :] - mean) / std)

    nwin = len(starts)
    expand = lambda tensor: tensor.unsqueeze(0).expand(nwin, -1).contiguous()
    return {
        "x": torch.stack(x_windows, dim=0),
        "state_input": torch.stack(state_input_windows, dim=0),
        "state_label": torch.stack(state_label_windows, dim=0),
        "target_mask": expand(target_mask),
        "target_class_label": expand(record["target_class_label"].long()),
        "intent_label": expand(record["intent_label"].long()),
        "threat_label": expand(record["threat_label"].long()),
        "response_action_label": expand(record["response_action_label"].long()),
        "start": torch.as_tensor(starts, dtype=torch.long),
        "scene_file": record["scene_file"],
    }


def sample_window_indices(nwin: int, windows_per_scene: int, random_windows: bool) -> list[int]:
    if windows_per_scene <= 0 or windows_per_scene >= nwin:
        return list(range(nwin))
    if random_windows:
        return sorted(random.sample(range(nwin), windows_per_scene))
    return list(range(windows_per_scene))


def select_windows(
    item: dict[str, Any],
    windows_per_scene: int = 0,
    random_windows: bool = False,
) -> dict[str, Any]:
    nwin = int(item["x"].shape[0])
    indices = sample_window_indices(nwin, int(windows_per_scene), bool(random_windows))
    if len(indices) == nwin:
        return item

    idx = torch.as_tensor(indices, dtype=torch.long)
    indexed_keys = {
        "x",
        "state_input",
        "state_label",
        "target_mask",
        "target_class_label",
        "intent_label",
        "threat_label",
        "response_action_label",
        "start",
    }
    out = {"scene_file": item["scene_file"]}
    for key in indexed_keys:
        out[key] = item[key].index_select(0, idx)
    return out


class SceneWindowDataset(Dataset):
    """Dataset item is one scene expanded into sliding windows."""

    def __init__(
        self,
        scene_files: list[str | Path],
        tin: int = 32,
        tout: int = 16,
        stride: int = 1,
        max_targets: int = 4,
        iq_scale: float = 76.0,
    ) -> None:
        self.scene_files = [Path(path) for path in scene_files]
        self.tin = int(tin)
        self.tout = int(tout)
        self.stride = int(stride)
        self.max_targets = int(max_targets)
        self.iq_scale = float(iq_scale)
        self.window_count = 0
        for scene_file in self.scene_files:
            nframes = read_scene_nframes(scene_file)
            self.window_count += max(0, (nframes - self.tin - self.tout) // self.stride + 1)
        if self.window_count <= 0:
            raise ValueError("No sliding-window samples were created. Check Tin/Tout/stride.")

    def __len__(self) -> int:
        return len(self.scene_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = scene_to_tensor_record(
            self.scene_files[index],
            max_targets=self.max_targets,
            iq_scale=self.iq_scale,
        )
        return record_to_windows(record, self.tin, self.tout, self.stride)


def scene_window_collate(
    items: list[dict[str, Any]],
    windows_per_scene: int = 0,
    random_windows: bool = False,
) -> dict[str, Any]:
    items = [
        select_windows(item, windows_per_scene=windows_per_scene, random_windows=random_windows)
        for item in items
    ]
    scene_files: list[str] = []
    for item in items:
        scene_files.extend([str(item["scene_file"])] * int(item["x"].shape[0]))

    keys = [
        "x",
        "state_input",
        "state_label",
        "target_mask",
        "target_class_label",
        "intent_label",
        "threat_label",
        "response_action_label",
        "start",
    ]
    batch = {key: torch.cat([item[key] for item in items], dim=0) for key in keys}
    batch["scene_file"] = scene_files
    return batch

