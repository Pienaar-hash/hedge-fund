# Ops Commands (cheat sheet)

**Credentials**
```bash
read -s -p "Binance API key: " BINANCE_API_KEY; echo
read -s -p "Binance API secret: " BINANCE_API_SECRET; echo
BINANCE_API_KEY="$BINANCE_API_KEY" BINANCE_API_SECRET="$BINANCE_API_SECRET" bash scripts/write_keys_env.sh
set -a; source ./.env; set +a
```

**Auth doctor**
```bash
ENV=prod PYTHONPATH=. ./venv/bin/python scripts/binance_auth_doctor.py
```

**Warmup & logs**
```bash
EXECUTOR_MAX_SEC=45 bash scripts/exec_once_timeout.sh || true
bash scripts/quick_watch.sh
```

**Go-live**
```bash
bash scripts/go_live_now.sh
```

**Revert after event**
```bash
EVENT_GUARD=0 bash scripts/go_live_now.sh
```

**ML retrain**
```bash
crontab -e
10 2 * * * cd /ABS/PATH/TO/hedge-fund && /bin/bash scripts/ml_retrain_cron.sh >> models/cron.log 2>&1
/bin/bash scripts/ml_retrain_now.sh
/bin/bash scripts/ml_health.sh
```
