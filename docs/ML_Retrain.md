# ML Nightly Retrain — Cron First

We rely on cron for nightly ML retraining and signal evaluation (no supervisor required).

## Install (02:10 UTC)
```bash
crontab -e
# Add:
10 2 * * * cd /ABS/PATH/TO/hedge-fund && /bin/bash scripts/ml_retrain_cron.sh >> models/cron.log 2>&1
```

## Manual Run (any time)
```bash
/bin/bash scripts/ml_retrain_now.sh
```

Artifacts written by both cron and manual runs:

- `models/registry.json` — model metadata per symbol
- `models/signal_eval.json` — ML vs RULE offline evaluation
- `models/last_train_report.json` — consolidated report consumed by the Dashboard ML tab

> Tip: If Binance is blocked on this host, temporarily set `ML_SIMULATOR=1` before running the scripts to produce synthetic artifacts. Remove it in production once connectivity is restored.
