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

Each run writes JSONL metrics to:

```text
radar_detect/train_baseline/runs/run_YYYYmmdd_HHMMSS/metrics.jsonl
```

The JSONL file contains one `config` record, one `epoch` record per epoch, and a
final `done` record. Checkpoints are saved as `best.pt` and `last.pt`.

`--device auto` uses CUDA when available and otherwise falls back to CPU. Use
`--device mps` only after confirming finite losses on your local PyTorch/macOS
stack.
