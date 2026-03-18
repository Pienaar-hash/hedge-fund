# OPERATIONS.md — GPT Hedge v5.6 Runbook

## Core Commands

| Task | Command |
|------|----------|
| **Long-run executor** | `ENV=prod PYTHONPATH=. python -m execution.executor_live` |
| **One-shot intent + sync** | `ENV=prod PYTHONPATH=. ONE_SHOT=1 python -m execution.executor_live` |
| **Dashboard (Streamlit)** | `streamlit run dashboard/app.py --server.port=8501` |
| **Router doctor (audit)** | `ENV=prod PYTHONPATH=. python scripts/doctor.py --router` |
| **Backfill missing fills / PnL** | `python3 scripts/backfill_fills_pnl.py --apply` |
| **Telegram mini-report (dry-run)** | `ENV=prod PYTHONPATH=. python -m execution.telegram_report --dry-run` |

---

## Supervisor Process Map
| Process | Purpose |
|----------|----------|
| `executor` | Core trading loop (ACK/FILL split) |
| `sync_state` | Publishes NAV + Firestore updates |
| `dashboard` | Streamlit front-end |
| `doctor` | Router health + telemetry |
| `leaderboard_sync` | Optional investor feed |

**Restart after patches:**
```bash
sudo supervisorctl restart hedge:executor hedge:sync_state hedge:dashboard hedge:doctor
