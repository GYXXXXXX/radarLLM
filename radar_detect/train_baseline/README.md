# Raw-IQ FMCW Baseline

This directory contains a PyTorch baseline for real-target classification and
future `[x, y, vx, vy]` regression from raw FMCW-MIMO IQ windows.

## Install

```bash
python3 -m pip install -r radar_detect/train_baseline/requirements.txt
```

## Compile Check

```bash
python3 -m compileall radar_detect/train_baseline
```

## Train

```bash
python3 radar_detect/train_baseline/train.py \
  --dataset-dir radar_detect/fmcw_traj_dataset \
  --epochs 50 \
  --device cpu \
  --batch-size 8 \
  --print-sample
```

Large raw-IQ scenes are cached with a bounded per-process LRU cache by default.
Use `--max-cache-scenes 32` to control the cap, `--max-cache-scenes 0` or
`--no-cache-scenes` to disable scene caching, and `--prefetch-factor 1` when
using `--num-workers > 0` to reduce DataLoader shared-memory pressure on
Windows.

Training and validation print running progress every 10 batches by default.
Use `--log-every N` to change the interval or `--log-every 0` to disable it.
For training, scene order is shuffled while windows from the same scene stay
adjacent, which avoids repeatedly reloading large `.mat` files. Use
`--window-shuffle` only if you need the older per-window random order.

Each run writes JSONL metrics to:

```text
radar_detect/train_baseline/runs/run_YYYYmmdd_HHMMSS/metrics.jsonl
```

The JSONL file contains one `config` record, one `epoch` record per epoch, and a
final `done` record. Checkpoints are saved as `best.pt` and `last.pt`.

`--device auto` uses CUDA when available and otherwise falls back to CPU. Use
`--device mps` only after confirming finite losses on your local PyTorch/macOS
stack.

## Visualize Predictions

```bash
python radar_detect/train_baseline/visualize_predictions.py \
  --checkpoint radar_detect/train_baseline/runs/run_20260523_152831/best.pt \
  --dataset-dir radar_detect/fmcw_traj_dataset_2000scenes \
  --scene-index 1600 \
  --window-index 0
```

The script writes a PNG under `prediction_plots/` next to the checkpoint. It
plots future ground-truth and predicted `[x, y]` trajectories plus future speed
for each real target slot.

To stitch every sliding-window prediction in one scene into a scene-level
trajectory, average overlapping future-frame predictions with:

```bash
python radar_detect/train_baseline/visualize_predictions.py \
  --checkpoint radar_detect/train_baseline/runs/run_20260523_152831/last.pt \
  --dataset-dir radar_detect/fmcw_traj_dataset_2000scenes \
  --scene-index 1600 \
  --stitch-scene
```

For the default `Tin=16, Tout=8`, stitched predictions cover frames 16-47.
Frames 0-15 are shown as ground truth only because they do not have a full
past-input window before them.
