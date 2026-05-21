"""Shared utilities for MATLAB v7.3 loading, normalization, metrics, and logging."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    from .config import STATE_MEAN, STATE_STD
except ImportError:  # pragma: no cover - direct script execution fallback
    from config import STATE_MEAN, STATE_STD


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def decode_matlab_char(data: np.ndarray) -> str:
    arr = np.asarray(data)
    flat = arr.flatten(order="F")
    chars = [chr(int(x)) for x in flat if int(x) != 0]
    return "".join(chars)


def matlab_to_numpy(data: np.ndarray) -> np.ndarray:
    """Convert h5py-loaded MATLAB arrays to their MATLAB semantic axis order."""
    if data.ndim <= 1:
        return data
    axes = tuple(reversed(range(data.ndim)))
    return np.transpose(data, axes=axes)


def read_matlab_dataset(handle: h5py.File, dataset: h5py.Dataset) -> Any:
    """Read a MATLAB v7.3 HDF5 dataset, including chars, refs, logicals, and complex."""
    matlab_class = dataset.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("ascii", errors="ignore")

    if matlab_class == "char":
        return decode_matlab_char(dataset[()])

    ref_dtype = h5py.check_dtype(ref=dataset.dtype)
    if ref_dtype is not None:
        refs = np.asarray(dataset[()]).flatten(order="F")
        values = []
        for ref in refs:
            values.append(None if not ref else read_matlab_dataset(handle, handle[ref]))
        return values

    data = dataset[()]
    if dataset.dtype.names == ("real", "imag"):
        data = data["real"] + 1j * data["imag"]

    data = np.asarray(data)
    data = matlab_to_numpy(data)

    if matlab_class == "logical":
        data = data.astype(bool)

    if data.shape == ():
        return data.item()
    if data.size == 1:
        return data.item()
    return data


def _as_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def load_scene(scene_path: str | Path) -> dict[str, Any]:
    """Load one scene and assert semantic shapes used by the baseline."""
    scene_path = Path(scene_path)
    with h5py.File(scene_path, "r") as handle:
        p = {field: read_matlab_dataset(handle, handle["p"][field]) for field in handle["p"].keys()}
        gt = {
            field: read_matlab_dataset(handle, handle["gt"][field])
            for field in handle["gt"].keys()
        }
        iq = read_matlab_dataset(handle, handle["iq"])

    nfast = _as_int(p["Nfast"])
    nchirp = _as_int(p["Nchirp"])
    nrx = _as_int(p["Nrx"])
    nframes = _as_int(p["Nframes"])
    expected_iq = (nfast, nchirp, nrx, nframes)

    if iq.shape != expected_iq:
        reversed_iq = tuple(reversed(expected_iq))
        if iq.shape == reversed_iq:
            iq = np.transpose(iq, (3, 2, 1, 0))
        else:
            raise ValueError(
                f"{scene_path.name}: iq shape {iq.shape} does not match expected "
                f"{expected_iq} after MATLAB axis conversion."
            )

    pos = np.asarray(gt["pos"], dtype=np.float32)
    vel = np.asarray(gt["vel"], dtype=np.float32)
    if pos.ndim != 3 or pos.shape[1] != nframes or pos.shape[2] != 2:
        raise ValueError(f"{scene_path.name}: gt.pos must be [Nobject, Nframes, 2], got {pos.shape}")
    if vel.ndim != 3 or vel.shape[1] != nframes or vel.shape[2] != 2:
        raise ValueError(f"{scene_path.name}: gt.vel must be [Nobject, Nframes, 2], got {vel.shape}")

    gt["pos"] = pos
    gt["vel"] = vel
    gt["isTarget"] = np.asarray(gt["isTarget"], dtype=bool).reshape(-1)
    gt["targetClassId"] = np.asarray(gt["targetClassId"], dtype=np.int64).reshape(-1)

    return {"scene_path": scene_path, "p": p, "gt": gt, "iq": iq.astype(np.complex64, copy=False)}


def read_scene_nframes(scene_path: str | Path) -> int:
    scene_path = Path(scene_path)
    with h5py.File(scene_path, "r") as handle:
        if "p" in handle and "Nframes" in handle["p"]:
            return _as_int(read_matlab_dataset(handle, handle["p"]["Nframes"]))
        shape = handle["iq"].shape
        return int(max(shape))


def standardize_state_np(state: np.ndarray) -> np.ndarray:
    return (state.astype(np.float32) - STATE_MEAN) / STATE_STD


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                sanitize_for_json(record),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def collect_scene_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    scene_files = sorted(dataset_dir.glob("scene_*.mat"))
    if not scene_files:
        raise FileNotFoundError(f"No scene_*.mat files found in: {dataset_dir}")
    return scene_files


def sanitize_for_json(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats into strict JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
