# Cloud Environment — Operations Guide

This document summarizes environment variables, log paths, services, and common ops tasks for the cloud host running Hedge.

## Services (Supervisor)
- Program `hedge-executor`: trading loop (`execution.executor_live`)
- Program `hedge-dashboard`: Streamlit dashboard (`dashboard/app.py`)

Supervisor quick commands:
```
sudo supervisorctl reread && sudo supervisorctl update
sudo supervisorctl status hedge-executor hedge-dashboard
sudo supervisorctl restart hedge-executor hedge-dashboard
```

## Logging
- Executor stdout: `/var/log/hedge/hedge-executor.out.log`
- Executor stderr: `/var/log/hedge/hedge-executor.err.log`
- Dashboard stdout: `/var/log/hedge/hedge-dashboard.out.log`
- Dashboard stderr: `/var/log/hedge/hedge-dashboard.err.log`

Tail helpers:
```
tail -n 200 /var/log/hedge/hedge-executor.out.log
tail -f /var/log/hedge/hedge-executor.out.log
```

## Environment Variables
Core:
- `ENV=prod|dev`: environment label
- `PYTHONPATH=/root/hedge-fund`
- `PYTHONUNBUFFERED=1`
- `BINANCE_TESTNET=1|0`: testnet vs mainnet
- `DRY_RUN=1|0`: skip real order placement when 1

Credentials:
- `FIREBASE_CREDS_PATH=/root/hedge-fund/config/firebase_creds.json`
- or `FIREBASE_CREDS_JSON` (raw or base64 JSON service account)
- `GOOGLE_APPLICATION_CREDENTIALS` also supported

Risk config:
- `RISK_LIMITS_CONFIG=/root/hedge-fund/config/risk_limits.json`

Telegram (optional):
- `TELEGRAM_ENABLED=1|0`
- `TELEGRAM_BOT_TOKEN=...`
- `TELEGRAM_CHAT_ID=...`

Dashboard:
- `DASHBOARD_REFRESH_SEC` (default 60)
- `EXECUTOR_LOG` (default `/var/log/hedge/hedge-executor.out.log`)

## File Layout
- Repo root: `/root/hedge-fund`
- Virtualenv: `/root/hedge-fund/venv`
- Config directory: `/root/hedge-fund/config`
- Risk limits: `/root/hedge-fund/config/risk_limits.json`

## Health Checks
```
supervisorctl status
python -c 'import os; print(os.getenv("ENV"), os.getenv("BINANCE_TESTNET"))'
PYTHONPATH=/root/hedge-fund /root/hedge-fund/venv/bin/python - <<'PY'
from execution.exchange_utils import _is_dual_side
print("dualSide:", _is_dual_side())
PY
```

## Dashboard
Start locally (venv):
```
streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0
```
Supervisor command is configured in `deploy/supervisor/hedge-dashboard.conf`.

### Expose dashboard with NGINX (TLS + Basic Auth)

- Copy the provided conf and enable the site:
  - Place `deploy/nginx/hedge-dashboard.conf` at `/etc/nginx/sites-available/hedge-dashboard.conf`.
  - Symlink it: `ln -s /etc/nginx/sites-available/hedge-dashboard.conf /etc/nginx/sites-enabled/`.
  - Create Basic Auth file: `sudo htpasswd -c /etc/nginx/.htpasswd hedge`.
  - Test and reload: `sudo nginx -t && sudo systemctl reload nginx`.

- TLS with certbot (notes):
  - `sudo certbot --nginx -d hedge.example.com`
  - Ensure firewall allows 80/443: `sudo ufw allow 80,443/tcp`.

- The server block proxies Streamlit on `127.0.0.1:8501`, sets security headers, gzip, and SSE‑friendly timeouts.

### Sizing math: notional vs margin

- Futures “notional” is gross exposure (`qty × price`). With leverage `L`, required margin ≈ `notional / L`.
- In this repo, `capital_per_trade` represents GROSS notional. The executor passes `gross/lev` to the exchange so on‑exchange qty matches your desired exposure.
- Risk checks compare gross notional against floors and portfolio caps. Example: a $10 trade at 20× uses ~0.5 USDT margin.

### DRY_RUN toggle (Supervisor‑managed services)

- Default is DRY_RUN=1. To flip to live later:
  - One‑liner (document only; do not commit):
    - `sudo sed -i 's/DRY_RUN=1/DRY_RUN=0/g' /etc/supervisor/conf.d/hedge-executor.conf && sudo supervisorctl reread && sudo supervisorctl update && sudo supervisorctl restart hedge-executor`

## Guardrails
- Never commit credentials; use `config/firebase_creds.json` or env-based JSON.
- Risk checks enforced via `execution/risk_limits.py` and `config/risk_limits.json`.
- Prefer DRY_RUN=1 for dry-run validation on mainnet settings.
