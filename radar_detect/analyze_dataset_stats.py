#!/usr/bin/env python3
"""Analyze label and input normalization statistics for FMCW trajectory scenes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'h5py'. Install with: pip install h5py numpy") from exc


class RunningStats:
    """Streaming scalar statistics for arrays."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sumsq += float(np.square(arr).sum())
        self.min = min(self.min, float(arr.min()))
        self.max = max(self.max, float(arr.max()))

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        if not self.count:
            return float("nan")
        var = max(self.sumsq / self.count - self.mean**2, 0.0)
        return float(np.sqrt(var))


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


def read_matlab_dataset(handle: h5py.File, dataset: h5py.Dataset):
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


def collect_scene_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    scene_files = sorted(dataset_dir.glob("scene_*.mat"))
    if not scene_files:
        raise FileNotFoundError(f"No scene_*.mat files found in: {dataset_dir}")
    return scene_files


def read_index_counts(index_csv: Path) -> tuple[dict[int, int], dict[int, int]]:
    target_counts: dict[int, int] = {}
    interferer_counts: dict[int, int] = {}
    if not index_csv.exists():
        return target_counts, interferer_counts

    with index_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            nt = int(row["n_targets"])
            ni = int(row["n_interferers"])
            target_counts[nt] = target_counts.get(nt, 0) + 1
            interferer_counts[ni] = interferer_counts.get(ni, 0) + 1
    return target_counts, interferer_counts


def update_label_stats(handle: h5py.File, stats: dict[str, RunningStats]) -> tuple[int, int]:
    gt = {field: read_matlab_dataset(handle, handle["gt"][field]) for field in handle["gt"].keys()}
    is_target = np.asarray(gt["isTarget"], dtype=bool).reshape(-1)
    target_idx = np.where(is_target)[0]

    pos = np.asarray(gt["pos"], dtype=np.float32)
    vel = np.asarray(gt["vel"], dtype=np.float32)

    target_pos = pos[target_idx, :, :]
    target_vel = vel[target_idx, :, :]

    stats["x"].update(target_pos[:, :, 0])
    stats["y"].update(target_pos[:, :, 1])
    stats["vx"].update(target_vel[:, :, 0])
    stats["vy"].update(target_vel[:, :, 1])

    return int(pos.shape[1]), int(target_idx.size)


def update_iq_stats(handle: h5py.File, stats: dict[str, RunningStats]) -> None:
    iq = read_matlab_dataset(handle, handle["iq"])
    stats["iq_real"].update(np.real(iq))
    stats["iq_imag"].update(np.imag(iq))
    stats["iq_abs"].update(np.abs(iq))
    stats["iq_power"].update(np.square(np.abs(iq)))
    scene_rms = np.sqrt(np.mean(np.square(np.abs(iq))) + 1e-12)
    stats["iq_scene_rms"].update(np.array([scene_rms], dtype=np.float64))


def update_rd_stats(handle: h5py.File, stats: dict[str, RunningStats]) -> None:
    if "rdMap" not in handle:
        return
    rd_map = read_matlab_dataset(handle, handle["rdMap"])
    stats["rd"].update(rd_map)


def format_stats(name: str, stat: RunningStats) -> str:
    return (
        f"{name:12s} count={stat.count:10d} "
        f"min={stat.min:12.6g} max={stat.max:12.6g} "
        f"mean={stat.mean:12.6g} std={stat.std:12.6g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for radar_detect FMCW scenes."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="fmcw_traj_dataset",
        help="Directory containing scene_*.mat files.",
    )
    parser.add_argument("--tin", type=int, default=16, help="Input window length.")
    parser.add_argument("--tout", type=int, default=8, help="Prediction window length.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride.")
    parser.add_argument(
        "--skip-input",
        action="store_true",
        help="Only compute label statistics; skip IQ/RD input statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    scene_files = collect_scene_files(dataset_dir)
    index_csv = dataset_dir / "index.csv"

    label_stats = {name: RunningStats() for name in ("x", "y", "vx", "vy")}
    input_stats = {
        name: RunningStats()
        for name in ("iq_real", "iq_imag", "iq_abs", "iq_power", "iq_scene_rms", "rd")
    }

    total_windows = 0
    total_target_slots = 0
    nframes_seen: list[int] = []

    for scene_file in scene_files:
        with h5py.File(scene_file, "r") as handle:
            nframes, n_targets = update_label_stats(handle, label_stats)
            nframes_seen.append(nframes)
            windows = max(0, (nframes - args.tin - args.tout) // args.stride + 1)
            total_windows += windows
            total_target_slots += windows * n_targets

            if not args.skip_input:
                update_iq_stats(handle, input_stats)
                update_rd_stats(handle, input_stats)

    target_count_dist, interferer_count_dist = read_index_counts(index_csv)

    print(f"dataset dir           : {dataset_dir}")
    print(f"scene count           : {len(scene_files)}")
    print(f"frame counts          : {sorted(set(nframes_seen))}")
    print(f"Tin/Tout/stride       : {args.tin}/{args.tout}/{args.stride}")
    print(f"sliding windows       : {total_windows}")
    print(f"target-window labels  : {total_target_slots}")
    print(f"target count dist     : {target_count_dist}")
    print(f"interferer count dist : {interferer_count_dist}")

    print("\nLabel stats over real target frames:")
    for name in ("x", "y", "vx", "vy"):
        print(format_stats(name, label_stats[name]))

    print("\nRecommended label standardization:")
    for name in ("x", "y", "vx", "vy"):
        stat = label_stats[name]
        print(f"{name}_norm = ({name} - {stat.mean:.9g}) / {stat.std:.9g}")

    if args.skip_input:
        return

    print("\nInput stats:")
    for name in ("iq_real", "iq_imag", "iq_abs", "iq_power", "iq_scene_rms", "rd"):
        print(format_stats(name, input_stats[name]))

    print("\nRecommended input normalization:")
    print(
        "IQ global standardization: "
        f"real_norm = (real - {input_stats['iq_real'].mean:.9g}) / "
        f"{input_stats['iq_real'].std:.9g}, "
        f"imag_norm = (imag - {input_stats['iq_imag'].mean:.9g}) / "
        f"{input_stats['iq_imag'].std:.9g}"
    )
    print(
        "IQ per-sample energy normalization: "
        "iq_norm = iq / sqrt(mean(abs(iq)^2) + eps)"
    )
    print(
        "RD standardization: "
        f"rd_norm = (rdMap - {input_stats['rd'].mean:.9g}) / "
        f"{input_stats['rd'].std:.9g}"
    )


if __name__ == "__main__":
    main()
