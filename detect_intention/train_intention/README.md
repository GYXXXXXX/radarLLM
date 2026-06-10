# Train Intention

This module trains a Transformer to predict each target's intention from radar
monitoring sequences.

## Task

Default input:

- radar-derived target track sequence `[x, y, vx, vy]`;
- the script infers the window length from the dataset.

Raw IQ end-to-end training is still available with `--input-mode radar`, but
intention recognition is usually more stable with the default
`--input-mode track` because intent mainly depends on motion trend.

Outputs per target slot:

- `state_pred`: future `16` frames of `[x, y, vx, vy]`;
- `intentId`: intention classification;
- `threatLevel`: threat classification.

## Train

Smoke test:

```powershell
conda run -n radar python detect_intention/train_intention/train.py `
  --max-train-scenes 8 `
  --max-val-scenes 2 `
  --epochs 1 `
  --scene-batch-size 1 `
  --log-every 1 `
  --print-sample
```

Full run:

```powershell
conda run -n radar python detect_intention/train_intention/train.py `
  --epochs 30 `
  --scene-batch-size 64
```

This default command uses `--input-mode track`.
The default dataset is `detect_intention/intention_dataset_compact`.
Training optimizes only:

- `loss_intent`
- `loss_threat`
- `loss_state`

Early stopping is enabled by default:

- `--early-stop-patience 8`
- `--early-stop-metric loss_intent`

Watch these validation metrics in `metrics.jsonl` or the console:

- `val.intent_accuracy`: primary intention classification metric;
- `val.threat_accuracy`: threat-level classification metric;
- `val.ade_m` and `val.fde_m`: trajectory prediction error;
- `val.loss_intent`: early-stopping metric.

Raw IQ end-to-end run:

```powershell
conda run -n radar python detect_intention/train_intention/train.py `
  --input-mode radar `
  --epochs 30 `
  --scene-batch-size 64
```

The training script now infers `dataset_dir`, `tin`, `tout`, `nfast`,
`nchirp`, `nrx`, and patch sizes automatically from the first `scene_*.mat`.
If `detect_intention/intention_dataset_compact` exists, it is selected before
the full dataset.

To force a specific dataset:

```powershell
conda run -n radar python detect_intention/train_intention/train.py `
  --dataset-dir detect_intention/intention_dataset_compact `
  --epochs 30 `
  --scene-batch-size 128
```

Checkpoints and metrics are saved under:

```text
detect_intention/train_intention/runs/run_YYYYMMDD_HHMMSS/
```

## Evaluate

```powershell
conda run -n radar python detect_intention/train_intention/evaluate.py `
  --checkpoint detect_intention/train_intention/runs/run_YYYYMMDD_HHMMSS/best.pt `
  --split val
```

Evaluation writes:

- `eval_metrics.json`
- `predictions.jsonl`

## Decision Explanation

Offline rule-based explanation:

```powershell
conda run -n radar python detect_intention/train_intention/llm_decision.py `
  --predictions detect_intention/train_intention/runs/run_YYYYMMDD_HHMMSS/predictions.jsonl
```

OpenAI-compatible or local LLM endpoint:

```powershell
$env:LLM_API_URL="http://localhost:8000/v1"
$env:LLM_API_KEY="your-key-if-needed"
$env:LLM_MODEL="your-model-name"

conda run -n radar python detect_intention/train_intention/llm_decision.py `
  --provider openai_compatible `
  --predictions detect_intention/train_intention/runs/run_YYYYMMDD_HHMMSS/predictions.jsonl
```

The decision module consumes structured model predictions. The Transformer does
the numeric radar recognition; the LLM explains the response and can combine
intent, confidence, threat level, range trend, protected-point trend, and rules.
