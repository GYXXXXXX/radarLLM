# Detect Intention

This small project generates clean FMCW-MIMO radar scenes with target
intention labels and recommended response labels.

Run in MATLAB:

```matlab
cd('E:\VSCodeProject\radarLLM\detect_intention')
generate_intention_fmcw_dataset
visualize_intention_dataset
```

Quick smoke test:

```matlab
generate_intention_fmcw_dataset(5)
```

Compact dataset, much smaller per scene:

```matlab
generate_intention_fmcw_dataset(5000, [], 'compact')
```

Generated data are saved under:

```text
detect_intention/intention_dataset/
```

The compact profile saves to:

```text
detect_intention/intention_dataset_compact/
```

Each `scene_XXXXXX.mat` contains:

- `iq`: clean target-only FMCW-MIMO IQ data.
- `rdMap`: Range-Doppler maps.
- `gt`: trajectory truth plus `intentId`, `intentName`, `threatLevel`, and
  `recommendedActionName`.
- `meta`: scene-level labels and a short text summary for LLM-style intent
  reasoning.
- `p`: radar and dataset parameters.

Current intent classes:

| ID | Intent | Meaning | Response |
| --- | --- | --- | --- |
| 1 | `benign_transit` | Stable low-risk passage | `monitor` |
| 2 | `approach` | Closing toward radar or guarded water | `increase_tracking_rate` |
| 3 | `retreat` | Moving away from guarded water | `monitor` |
| 4 | `loiter_patrol` | Repeated local motion near one area | `classify_and_shadow` |
| 5 | `intercept` | Heading toward the protected point | `alert_and_allocate_tracker` |

Compared with `radar_detect/generate_fmcw_traj_dataset.m`, this generator:

- sets `nInterferers = 0`;
- does not synthesize false reflectors or wideband jammers;
- disables thermal noise by default for clean labels;
- adds intention and response fields to `gt`, `meta`, and `index.csv`.

Available generator profiles:

| Profile | Tensor size | Use case |
| --- | --- | --- |
| `full` | `Nfast=128, Nchirp=32, Nrx=8, Nframes=48` | Highest raw-IQ fidelity |
| `compact` | `Nfast=64, Nchirp=16, Nrx=4, Nframes=32` | Recommended for larger training sets |
| `tiny` | `Nfast=64, Nchirp=16, Nrx=2, Nframes=24` | Quick debugging only |

## Train intention prediction

The training code is in:

```text
detect_intention/train_intention/
```

Suggested order:

```powershell
conda run -n radar python detect_intention/train_intention/train.py --epochs 30 --scene-batch-size 1
conda run -n radar python detect_intention/train_intention/evaluate.py --checkpoint detect_intention/train_intention/runs/run_YYYYMMDD_HHMMSS/best.pt
conda run -n radar python detect_intention/train_intention/llm_decision.py --predictions detect_intention/train_intention/runs/run_YYYYMMDD_HHMMSS/predictions.jsonl
```

Training compact data:

```powershell
conda run -n radar python detect_intention/train_intention/train.py `
  --scene-batch-size 128 `
  --epochs 30 `
  --device cuda
```

The training script automatically reads `Nfast`, `Nchirp`, `Nrx`, and
`Nframes` from the first MAT scene. The default dataset is
`detect_intention/intention_dataset_compact`.
