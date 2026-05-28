#!/usr/bin/env python3
"""Preprocess MATLAB scenes into per-scene tensor cache files.

The cache stores full-scene normalized IQ tensors, not duplicated sliding
windows. Training still creates windows scene-first, but avoids MATLAB/HDF5
parsing and complex real/imag conversion during each epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .dataset import scene_to_tensor_record
    from .mat_utils import collect_scene_files
except ImportError:  # pragma: no cover
    from dataset import scene_to_tensor_record
    from mat_utils import collect_scene_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create per-scene tensor cache from .mat scenes.")
    parser.add_argument("--dataset-dir", type=str, default="radar_detect/fmcw_traj_dataset_2000scenes")
    parser.add_argument("--cache-dir", type=str, default="radar_detect/fmcw_tensor_cache_2000scenes")
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--iq-scale", type=float, default=76.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of scenes to preprocess.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_files = collect_scene_files(args.dataset_dir)
    if args.limit > 0:
        scene_files = scene_files[: args.limit]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_bytes = 0
    for index, scene_file in enumerate(scene_files, start=1):
        out_path = cache_dir / f"{scene_file.stem}.pt"
        if out_path.exists() and not args.overwrite:
            size = out_path.stat().st_size
            total_bytes += size
            records.append({"scene": scene_file.name, "cache": out_path.name, "bytes": size, "skipped": True})
            print(f"[{index}/{len(scene_files)}] skip {out_path.name}", flush=True)
            continue

        record = scene_to_tensor_record(
            scene_file,
            max_targets=args.max_targets,
            iq_scale=args.iq_scale,
        )
        torch.save(record, out_path)
        size = out_path.stat().st_size
        total_bytes += size
        records.append(
            {
                "scene": scene_file.name,
                "cache": out_path.name,
                "bytes": size,
                "nframes": int(record["x_scene"].shape[1]),
                "target_count": int(record["target_mask"].sum().item()),
                "skipped": False,
            }
        )
        print(
            f"[{index}/{len(scene_files)}] saved {out_path.name} "
            f"({size / (1024 ** 2):.2f} MiB)",
            flush=True,
        )

    metadata = {
        "type": "fmcw_scene_tensor_cache",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(Path(args.dataset_dir)),
        "cache_dir": str(cache_dir),
        "scene_count": len(scene_files),
        "max_targets": args.max_targets,
        "iq_scale": args.iq_scale,
        "total_bytes": total_bytes,
        "total_gib": total_bytes / (1024 ** 3),
        "records": records,
    }
    metadata_path = cache_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metadata: {metadata_path}")
    print(f"total cache size: {metadata['total_gib']:.2f} GiB")


if __name__ == "__main__":
    main()

