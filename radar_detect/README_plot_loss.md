# Plot Loss Trends

This script plots loss trends from a `metrics.jsonl` file.

## Install

```powershell
pip install -r .\radar_detect\requirements.txt
```

## Run

```powershell
python .\radar_detect\plot_loss_trends.py `
  --metrics .\radar_detect\train_baseline\runs\run_20260522_180436\metrics.jsonl `
  --out .\radar_detect\train_baseline\runs\run_20260522_180436\loss_trends.png
```

Add `--show` to open an interactive window.
