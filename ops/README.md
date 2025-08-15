# Hedge Fund Ops Guide — Phase 4

## Overview
This system runs **four Supervisor-managed processes** for live trading and reporting:

1. `hedge-signal` — Generates trade signals.
2. `hedge-executor` — Executes trades, updates Firestore state.
3. `hedge-sync` — Background sync to Firestore for NAV/positions.
4. `hedge-dashboard` — Streamlit investor dashboard.

All live state (Leaderboard, NAV, Positions) is stored in **Firestore** and read by the dashboard — **no local file reads**.

---

## Environment Variables

| Variable              | Purpose                                        | Example Value |
|-----------------------|------------------------------------------------|---------------|
| `ENV`                 | Environment label (`dev`, `prod`)              | `prod`        |
| `PYTHONPATH`          | Python module path root                        | `.`           |
| `FIREBASE_CREDS_PATH` | Path to Firestore credentials JSON             | `/root/hedge-fund/config/firebase_creds.json` |
| `BINANCE_TESTNET`     | Use Binance testnet (1) or live (0)             | `1`           |
| `TELEGRAM_ENABLED`    | Enable Telegram alerts (`1` or `0`)             | `1`           |
| `EXECUTOR_LABEL`      | Node label for logs / monitoring                | `hetzner-fsn1`|
| `BOT_TOKEN`           | Telegram bot token                              | *(secret)*    |
| `CHAT_ID`             | Telegram chat/group ID                          | *(secret)*    |

---

## Process Control

### Restart all
```bash
sudo supervisorctl restart all

sudo supervisorctl restart hedge-executor

sudo supervisorctl status

tail -f /var/log/hedge/hedge-executor.out.log
tail -f /var/log/hedge/hedge-dashboard.err.log

sudo supervisorctl status

http://167.235.205.126:8501

cd /root/hedge-fund
git fetch --all
git checkout <tag-or-commit>

sudo supervisorctl restart all
