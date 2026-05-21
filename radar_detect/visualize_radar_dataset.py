#!/usr/bin/env python3
"""Python visualization helpers for radar_detect FMCW trajectory scenes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'h5py'. Install with: pip install h5py numpy matplotlib"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'matplotlib'. Install with: pip install matplotlib"
    ) from exc


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


def as_1d_array(data, dtype=float) -> np.ndarray:
    return np.asarray(data, dtype=dtype).reshape(-1)


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
            if not ref:
                values.append(None)
            else:
                values.append(read_matlab_dataset(handle, handle[ref]))
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

    # MATLAB v7.3 scalar values are often stored as 1x1 arrays.
    if data.size == 1:
        return data.item()

    return data


def load_mat73_scene(scene_path: str | Path) -> dict:
    scene_path = Path(scene_path)
    with h5py.File(scene_path, "r") as handle:
        scene = {}
        for group_name in ("gt", "meta", "p"):
            group = {}
            for field in handle[group_name].keys():
                group[field] = read_matlab_dataset(handle, handle[group_name][field])
            scene[group_name] = group

        scene["iq"] = read_matlab_dataset(handle, handle["iq"])

    scene["scene_path"] = str(scene_path)
    return scene


def object_display_name(gt: dict, index: int) -> str:
    names = gt["name"]
    if not isinstance(names, list):
        names = [names]
    name = str(names[index])

    is_target = np.asarray(gt["isTarget"]).reshape(-1)
    prefix = "T" if bool(is_target[index]) else "I"
    return f"{prefix}{index + 1}:{name}"


def object_series(data, index: int) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 1:
        return arr
    return arr[index]


def save_scene_kinematics(scene: dict, out_dir: str | Path) -> Path:
    gt = scene["gt"]
    p = scene["p"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_objects = int(gt["numObjects"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    time_axis = np.arange(int(p["Nframes"])) / float(p["frameRate"])

    for idx in range(num_objects):
        axes[0].plot(
            time_axis,
            object_series(gt["range"], idx),
            linewidth=1.6,
            label=object_display_name(gt, idx),
        )
        axes[1].plot(
            time_axis,
            np.rad2deg(object_series(gt["azimuth"], idx)),
            linewidth=1.6,
            label=object_display_name(gt, idx),
        )
        axes[2].plot(
            time_axis,
            object_series(gt["radialVel"], idx),
            linewidth=1.6,
            label=object_display_name(gt, idx),
        )

    axes[0].set_ylabel("Range / m")
    axes[1].set_ylabel("Azimuth / deg")
    axes[2].set_ylabel("Radial velocity / m/s")
    axes[2].set_xlabel("Time / s")
    axes[0].set_title("Object kinematics over time")

    for ax in axes:
        ax.grid(True)
    axes[0].legend(loc="best", fontsize=8)

    curve_path = out_dir / f"{Path(scene['scene_path']).stem}_kinematics.png"
    fig.savefig(curve_path, dpi=180)
    plt.close(fig)
    return curve_path


def save_scene_iq(scene: dict, out_dir: str | Path, iq_frame=0, rx_index=0) -> Path:
    iq = scene["iq"]
    frame_idx = int(iq_frame)
    iq_slice = np.asarray(iq[:, :, rx_index, frame_idx])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    amp = 20.0 * np.log10(np.abs(iq_slice) + 1e-6)
    phase = np.angle(iq_slice)

    im0 = axes[0].imshow(amp, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(f"IQ amplitude | rx={rx_index + 1} frame={frame_idx + 1}")
    axes[0].set_xlabel("Chirp index")
    axes[0].set_ylabel("Fast-time sample")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(phase, aspect="auto", origin="lower", cmap="twilight")
    axes[1].set_title(f"IQ phase | rx={rx_index + 1} frame={frame_idx + 1}")
    axes[1].set_xlabel("Chirp index")
    axes[1].set_ylabel("Fast-time sample")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    iq_path = out_dir / (
        f"{Path(scene['scene_path']).stem}_iq_rx{rx_index + 1}_frame{frame_idx + 1}.png"
    )
    fig.savefig(iq_path, dpi=180)
    plt.close(fig)
    return iq_path


def summarize_index(index_csv: str | Path) -> None:
    index_csv = Path(index_csv)
    rows = []
    with index_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    print(f"index file: {index_csv}")
    print(f"num scenes: {len(rows)}")

    for field in ("n_targets", "n_interferers", "relation"):
        counter = {}
        for row in rows:
            value = row[field]
            counter[value] = counter.get(value, 0) + 1
        print(f"\n{field} distribution:")
        for key in sorted(counter, key=lambda x: (str(x))):
            print(f"  {key}: {counter[key]}")


def collect_scene_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    scene_files = sorted(dataset_dir.glob("scene_*.mat"))
    if not scene_files:
        raise FileNotFoundError(f"No scene_*.mat files found in: {dataset_dir}")
    return scene_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch visualize radar_detect scenes with Python.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="fmcw_traj_dataset",
        help="Directory containing scene_*.mat files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="python_visualization",
        help="Directory used to save generated figures directly without subfolders.",
    )
    parser.add_argument(
        "--iq-frame",
        type=int,
        default=1,
        help="1-based frame index used for IQ figure generation.",
    )
    parser.add_argument(
        "--rx-index",
        type=int,
        default=1,
        help="1-based RX channel index used for IQ figure generation.",
    )
    parser.add_argument(
        "--index-csv",
        type=str,
        default="fmcw_traj_dataset/index.csv",
        help="Path to index.csv for summary printing.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print index.csv statistics without loading a scene.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.summary_only:
        summarize_index(args.index_csv)
        return

    iq_frame = max(0, args.iq_frame - 1)
    rx_index = max(0, args.rx_index - 1)
    scene_files = collect_scene_files(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset dir : {args.dataset_dir}")
    print(f"output dir  : {out_dir}")
    print(f"scene count : {len(scene_files)}")
    print(f"iq frame    : {iq_frame + 1}")
    print(f"rx channel  : {rx_index + 1}")

    for scene_file in scene_files:
        scene = load_mat73_scene(scene_file)
        kinematics_path = save_scene_kinematics(scene, out_dir)
        iq_path = save_scene_iq(scene, out_dir, iq_frame=iq_frame, rx_index=rx_index)
        print(f"processed {scene_file.name}")
        print(f"  {kinematics_path.name}")
        print(f"  {iq_path.name}")


if __name__ == "__main__":
    main()
