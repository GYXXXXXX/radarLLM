# Scene-First Patch ST-Transformer

This is a separate experiment directory. It does not replace or edit
`radar_detect/train_baseline`.

## What Changed

- Input is still raw IQ windows: `[B, 16, Tin, 32, 128]`.
- The model uses light patch embedding over chirp/fast-time, spatial Transformer
  per frame, temporal Transformer across `Tin`, and fixed slot queries.
- The output is objectness plus trajectory:

```text
objectness_logits: [B, 4]
state_pred:        [B, 4, Tout, 4]
```

- The loss is:

```text
loss = lambda_obj * BCEWithLogits(objectness, target_mask)
     + lambda_state * masked SmoothL1(state_pred, state_label)
```

- Checkpoints are saved as:

```text
best.pt
best_by_loss_state.pt
best_by_ade.pt
last.pt
```

## Train Directly From `.mat`

This uses scene-first loading: one dataset item is one scene, and the scene is
expanded into all sliding windows inside the DataLoader collate step.

Smoke test from cache:

```bash
python radar_detect/train_st_transformer/train.py \
  --dataset-dir radar_detect/fmcw_traj_dataset_2000scenes \
  --data-mode cached_scene \
  --cache-dir radar_detect/fmcw_tensor_cache_2000scenes \
  --epochs 1 \
  --scene-batch-size 2 \
  --windows-per-scene 2 \
  --val-windows-per-scene 4 \
  --max-train-scenes 8 \
  --max-val-scenes 2 \
  --device cuda \
  --print-sample
```

Training first samples `N` scenes, then samples `M` windows from each scene.
The effective batch size is:

```text
effective windows = scene_batch_size * windows_per_scene
```

With the smoke-test command above, that is `2 * 2 = 4` windows. Validation uses
all windows by default; use `--val-windows-per-scene M` for quick checks.

Full training example:

```bash
python radar_detect/train_st_transformer/train.py \
  --dataset-dir radar_detect/fmcw_traj_dataset_2000scenes \
  --data-mode mat_scene \
  --epochs 30 \
  --scene-batch-size 16 \
  --windows-per-scene 4 \
  --device cuda \
  --print-sample
```

For the default `Tin=16, Tout=8`, each scene contributes 25 windows. A
`--scene-batch-size 16 --windows-per-scene 4` therefore produces an effective
64-window training batch while reading only 16 scenes.

## Preprocess Tensor Cache

The original `.mat` files are already MATLAB v7.3 HDF5 files. Converting them
to another HDF5 file usually does not remove the core cost: parsing MATLAB
objects, fixing axis order, and splitting complex IQ into real/imag channels.

The recommended cache is therefore one `.pt` tensor file per scene:

```text
scene_000001.pt:
  x_scene     [16, 48, 32, 128] normalized real/imag IQ
  state_full  [4, 48, 4] raw [x,y,vx,vy]
  target_mask [4]
  class_label [4] retained for analysis only
```

It does not save all 25 windows explicitly, because that would duplicate
overlapping frames and can inflate the cache by roughly 8x.

Create the cache:

```bash
python radar_detect/train_st_transformer/preprocess_cache.py \
  --dataset-dir radar_detect/fmcw_traj_dataset_2000scenes \
  --cache-dir radar_detect/fmcw_tensor_cache_2000scenes
```

Train from cache:

```bash
python radar_detect/train_st_transformer/train.py \
  --dataset-dir radar_detect/fmcw_traj_dataset_2000scenes \
  --data-mode cached_scene \
  --cache-dir radar_detect/fmcw_tensor_cache_2000scenes \
  --epochs 30 \
  --scene-batch-size 16 \
  --windows-per-scene 4 \
  --device cuda
```

## Visualize

Single window:

```bash
python radar_detect/train_st_transformer/visualize_predictions.py \
  --checkpoint radar_detect/train_st_transformer/runs/run_YYYYmmdd_HHMMSS/best_by_ade.pt \
  --dataset-dir radar_detect/fmcw_traj_dataset_2000scenes \
  --scene-index 1600 \
  --window-index 0
```

Stitched scene:

```bash
python radar_detect/train_st_transformer/visualize_predictions.py \
  --checkpoint radar_detect/train_st_transformer/runs/run_YYYYmmdd_HHMMSS/best_by_ade.pt \
  --dataset-dir radar_detect/fmcw_traj_dataset_2000scenes \
  --scene-index 1600 \
  --stitch-scene
```

If using cached scenes, add:

```text
--data-mode cached_scene --cache-dir radar_detect/fmcw_tensor_cache_2000scenes
```
