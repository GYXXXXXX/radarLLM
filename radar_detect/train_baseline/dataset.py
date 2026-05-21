"""Sliding-window Dataset for raw-IQ FMCW trajectory prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .config import STATE_MEAN, STATE_STD
    from .utils import load_scene, read_scene_nframes
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import STATE_MEAN, STATE_STD
    from utils import load_scene, read_scene_nframes


class FmcwTrajectoryDataset(Dataset):
    """Return raw IQ windows and fixed-slot labels for real targets only."""

    def __init__(
        self,
        scene_files: list[str | Path],
        tin: int = 16,
        tout: int = 8,
        stride: int = 1,
        max_targets: int = 4,
        iq_scale: float = 76.0,
        cache_scenes: bool = True,
    ) -> None:
        self.scene_files = [Path(path) for path in scene_files]
        self.tin = int(tin)
        self.tout = int(tout)
        self.stride = int(stride)
        self.max_targets = int(max_targets)
        self.iq_scale = float(iq_scale)
        self.cache_scenes = bool(cache_scenes)
        self._scene_cache: dict[Path, dict] = {}

        self.samples: list[tuple[Path, int]] = []
        for scene_file in self.scene_files:
            nframes = read_scene_nframes(scene_file)
            last_start = nframes - self.tin - self.tout
            if last_start < 0:
                continue
            for start in range(0, last_start + 1, self.stride):
                self.samples.append((scene_file, start))

        if not self.samples:
            raise ValueError("No sliding-window samples were created. Check Tin/Tout/stride.")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_scene(self, scene_file: Path) -> dict:
        if not self.cache_scenes:
            return load_scene(scene_file)
        scene = self._scene_cache.get(scene_file)
        if scene is None:
            scene = load_scene(scene_file)
            self._scene_cache[scene_file] = scene
        return scene

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        scene_file, start = self.samples[index]
        scene = self._load_scene(scene_file)
        iq = scene["iq"]
        gt = scene["gt"]

        input_slice = slice(start, start + self.tin)
        pred_slice = slice(start + self.tin, start + self.tin + self.tout)

        x = self._make_input(iq[:, :, :, input_slice])
        class_label, state_label, target_mask = self._make_labels(gt, pred_slice)

        return {
            "x": torch.from_numpy(x),
            "class_label": torch.from_numpy(class_label),
            "state_label": torch.from_numpy(state_label),
            "target_mask": torch.from_numpy(target_mask),
            "scene_file": scene_file.name,
            "start": start,
        }

    def _make_input(self, iq_window: np.ndarray) -> np.ndarray:
        """Convert [Nfast, Nchirp, Nrx, Tin] complex IQ to [2*Nrx, Tin, Nchirp, Nfast]."""
        if iq_window.ndim != 4:
            raise ValueError(f"iq_window must be 4-D, got {iq_window.shape}")

        nfast, nchirp, nrx, tin = iq_window.shape
        if tin != self.tin:
            raise ValueError(f"Expected Tin={self.tin}, got {tin}")

        iq_window = iq_window / self.iq_scale
        real = np.real(iq_window).astype(np.float32, copy=False)
        imag = np.imag(iq_window).astype(np.float32, copy=False)

        x = np.empty((2 * nrx, tin, nchirp, nfast), dtype=np.float32)
        for rx in range(nrx):
            x[2 * rx] = np.transpose(real[:, :, rx, :], (2, 1, 0))
            x[2 * rx + 1] = np.transpose(imag[:, :, rx, :], (2, 1, 0))
        return x

    def _make_labels(
        self,
        gt: dict,
        pred_slice: slice,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        is_target = np.asarray(gt["isTarget"], dtype=bool).reshape(-1)
        target_idx = np.where(is_target)[0]
        if target_idx.size > self.max_targets:
            raise ValueError(f"Scene has {target_idx.size} real targets, max_targets={self.max_targets}")

        class_ids = np.asarray(gt["targetClassId"], dtype=np.int64).reshape(-1)
        pos = np.asarray(gt["pos"], dtype=np.float32)
        vel = np.asarray(gt["vel"], dtype=np.float32)

        class_label = np.zeros((self.max_targets,), dtype=np.int64)
        state_label = np.zeros((self.max_targets, self.tout, 4), dtype=np.float32)
        target_mask = np.zeros((self.max_targets,), dtype=np.float32)

        for slot, obj_idx in enumerate(target_idx):
            class_id = int(class_ids[obj_idx])
            if class_id < 1 or class_id > 4:
                raise ValueError(f"Real target class id must be 1..4, got {class_id}")

            raw_state = np.concatenate(
                [pos[obj_idx, pred_slice, :], vel[obj_idx, pred_slice, :]],
                axis=-1,
            )
            if raw_state.shape != (self.tout, 4):
                raise ValueError(f"Expected state shape {(self.tout, 4)}, got {raw_state.shape}")

            class_label[slot] = class_id
            state_label[slot] = (raw_state - STATE_MEAN) / STATE_STD
            target_mask[slot] = 1.0

        return class_label, state_label, target_mask
