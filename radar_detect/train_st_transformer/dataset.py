"""Scene-first datasets and tensor-cache helpers for FMCW windows."""

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


def _safe_torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_full_input(iq: np.ndarray, iq_scale: float) -> torch.Tensor:
    """Convert [Nfast, Nchirp, Nrx, Nframes] complex IQ to [2*Nrx, Nframes, Nchirp, Nfast]."""
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


def make_full_labels(gt: dict, max_targets: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_target = np.asarray(gt["isTarget"], dtype=bool).reshape(-1)
    target_idx = np.where(is_target)[0]
    if target_idx.size > max_targets:
        raise ValueError(f"Scene has {target_idx.size} real targets, max_targets={max_targets}")

    class_ids = np.asarray(gt["targetClassId"], dtype=np.int64).reshape(-1)
    pos = np.asarray(gt["pos"], dtype=np.float32)
    vel = np.asarray(gt["vel"], dtype=np.float32)
    nframes = pos.shape[1]

    class_label = np.zeros((max_targets,), dtype=np.int64)
    state_full = np.zeros((max_targets, nframes, 4), dtype=np.float32)
    target_mask = np.zeros((max_targets,), dtype=np.float32)

    for slot, obj_idx in enumerate(target_idx):
        class_id = int(class_ids[obj_idx])
        if class_id < 1 or class_id > 4:
            raise ValueError(f"Real target class id must be 1..4, got {class_id}")

        state_full[slot] = np.concatenate([pos[obj_idx], vel[obj_idx]], axis=-1)
        class_label[slot] = class_id
        target_mask[slot] = 1.0

    return (
        torch.from_numpy(class_label),
        torch.from_numpy(state_full),
        torch.from_numpy(target_mask),
    )


def scene_to_tensor_record(
    scene_file: str | Path,
    max_targets: int,
    iq_scale: float,
) -> dict[str, Any]:
    scene_file = Path(scene_file)
    scene = load_scene(scene_file)
    class_label, state_full, target_mask = make_full_labels(scene["gt"], max_targets)
    return {
        "scene_file": scene_file.name,
        "x_scene": make_full_input(scene["iq"], iq_scale),
        "class_label": class_label,
        "state_full": state_full,
        "target_mask": target_mask,
    }


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
    class_label = record["class_label"].long()

    nframes = int(x_scene.shape[1])
    last_start = nframes - int(tin) - int(tout)
    if last_start < 0:
        raise ValueError(f"Scene {record['scene_file']} is too short for Tin/Tout.")

    all_starts = list(range(0, last_start + 1, int(stride)))
    if window_indices is None:
        starts = all_starts
    else:
        if isinstance(window_indices, torch.Tensor):
            requested = [int(item) for item in window_indices.reshape(-1).tolist()]
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
    state_windows = []
    for start in starts:
        x_windows.append(x_scene[:, start : start + tin])
        pred_slice = slice(start + tin, start + tin + tout)
        state_windows.append((state_full[:, pred_slice, :] - mean) / std)

    nwin = len(starts)
    return {
        "x": torch.stack(x_windows, dim=0),
        "state_label": torch.stack(state_windows, dim=0),
        "target_mask": target_mask.unsqueeze(0).expand(nwin, -1).contiguous(),
        "class_label": class_label.unsqueeze(0).expand(nwin, -1).contiguous(),
        "start": torch.as_tensor(starts, dtype=torch.long),
        "scene_file": record["scene_file"],
    }


def sample_window_indices(
    nwin: int,
    windows_per_scene: int,
    random_windows: bool,
) -> list[int]:
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
    return {
        "x": item["x"].index_select(0, idx),
        "state_label": item["state_label"].index_select(0, idx),
        "target_mask": item["target_mask"].index_select(0, idx),
        "class_label": item["class_label"].index_select(0, idx),
        "start": item["start"].index_select(0, idx),
        "scene_file": item["scene_file"],
    }


class SceneWindowDataset(Dataset):
    """Dataset item is one scene expanded into all sliding windows."""

    def __init__(
        self,
        scene_files: list[str | Path],
        tin: int = 16,
        tout: int = 8,
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


class CachedSceneWindowDataset(Dataset):
    """Scene-first dataset backed by preprocessed per-scene tensor files."""

    def __init__(
        self,
        scene_files: list[str | Path],
        cache_dir: str | Path,
        tin: int = 16,
        tout: int = 8,
        stride: int = 1,
    ) -> None:
        self.scene_files = [Path(path) for path in scene_files]
        self.cache_dir = Path(cache_dir)
        self.tin = int(tin)
        self.tout = int(tout)
        self.stride = int(stride)
        self.cache_files = [self.cache_dir / f"{path.stem}.pt" for path in self.scene_files]
        missing = [path for path in self.cache_files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} cached scene tensors. First missing: {missing[0]}"
            )

        self.window_count = 0
        for scene_file in self.scene_files:
            nframes = read_scene_nframes(scene_file)
            self.window_count += max(0, (nframes - self.tin - self.tout) // self.stride + 1)
        if self.window_count <= 0:
            raise ValueError("No sliding-window samples were created. Check Tin/Tout/stride.")

    def __len__(self) -> int:
        return len(self.cache_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = _safe_torch_load(self.cache_files[index])
        return record_to_windows(record, self.tin, self.tout, self.stride)


def scene_window_collate(
    items: list[dict[str, Any]],
    windows_per_scene: int = 0,
    random_windows: bool = False,
) -> dict[str, Any]:
    items = [
        select_windows(
            item,
            windows_per_scene=windows_per_scene,
            random_windows=random_windows,
        )
        for item in items
    ]
    scene_files: list[str] = []
    for item in items:
        scene_files.extend([str(item["scene_file"])] * int(item["x"].shape[0]))

    return {
        "x": torch.cat([item["x"] for item in items], dim=0),
        "state_label": torch.cat([item["state_label"] for item in items], dim=0),
        "target_mask": torch.cat([item["target_mask"] for item in items], dim=0),
        "class_label": torch.cat([item["class_label"] for item in items], dim=0),
        "start": torch.cat([item["start"] for item in items], dim=0),
        "scene_file": scene_files,
    }
