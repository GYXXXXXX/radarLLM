"""MATLAB v7.3/HDF5 helpers for intention scenes."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np


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
    if data.ndim <= 1:
        return data
    axes = tuple(reversed(range(data.ndim)))
    return np.transpose(data, axes=axes)


def read_matlab_dataset(handle: h5py.File, dataset: h5py.Dataset) -> Any:
    matlab_class = dataset.attrs.get("MATLAB_class", b"")
    if isinstance(matlab_class, bytes):
        matlab_class = matlab_class.decode("ascii", errors="ignore")

    if matlab_class == "char":
        return decode_matlab_char(dataset[()])

    ref_dtype = h5py.check_dtype(ref=dataset.dtype)
    if ref_dtype is not None:
        refs = np.asarray(dataset[()]).flatten(order="F")
        return [None if not ref else read_matlab_dataset(handle, handle[ref]) for ref in refs]

    data = dataset[()]
    if dataset.dtype.names == ("real", "imag"):
        data = data["real"] + 1j * data["imag"]

    data = matlab_to_numpy(np.asarray(data))
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
    scene_path = Path(scene_path)
    with h5py.File(scene_path, "r") as handle:
        p = {field: read_matlab_dataset(handle, handle["p"][field]) for field in handle["p"].keys()}
        gt = {field: read_matlab_dataset(handle, handle["gt"][field]) for field in handle["gt"].keys()}
        meta = {
            field: read_matlab_dataset(handle, handle["meta"][field])
            for field in handle["meta"].keys()
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
                f"{scene_path.name}: iq shape {iq.shape} does not match expected {expected_iq}."
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
    gt["intentId"] = np.asarray(gt["intentId"], dtype=np.int64).reshape(-1)
    gt["threatLevel"] = np.asarray(gt["threatLevel"], dtype=np.int64).reshape(-1)
    gt["responseActionId"] = np.asarray(gt["responseActionId"], dtype=np.int64).reshape(-1)

    return {
        "scene_path": scene_path,
        "p": p,
        "gt": gt,
        "meta": meta,
        "iq": iq.astype(np.complex64, copy=False),
    }


def read_scene_nframes(scene_path: str | Path) -> int:
    with h5py.File(scene_path, "r") as handle:
        if "p" in handle and "Nframes" in handle["p"]:
            return _as_int(read_matlab_dataset(handle, handle["p"]["Nframes"]))
        return int(max(handle["iq"].shape))


def collect_scene_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    scene_files = sorted(dataset_dir.glob("scene_*.mat"))
    if not scene_files:
        raise FileNotFoundError(f"No scene_*.mat files found in: {dataset_dir}")
    return scene_files


def sanitize_for_json(value: Any) -> Any:
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

