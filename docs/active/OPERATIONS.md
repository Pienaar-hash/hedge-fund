# OPERATIONS.md — GPT Hedge v7.9 Runbook

## Core Commands

| Task | Command |
|------|----------|
| **Long-run executor** | `ENV=prod PYTHONPATH=. python -m execution.executor_live` |
| **One-shot intent + sync** | `ENV=prod PYTHONPATH=. ONE_SHOT=1 python -m execution.executor_live` |
| **Dashboard (Streamlit)** | `streamlit run dashboard/app.py --server.port=8501` |
| **Runtime sanity** | `ENV=prod PYTHONPATH=. python scripts/runtime_sanity_check_v7_6.py` |
| **Execution debug** | `ENV=prod PYTHONPATH=. python scripts/exec_debug.py` |
| **State smoke** | `ENV=prod PYTHONPATH=. python scripts/smoke_test.py` |
| **Telegram mini-report (dry-run)** | `ENV=prod PYTHONPATH=. python -m execution.telegram_report --dry-run` |

---

## Supervisor Process Map
| Process | Purpose |
|----------|----------|
| `executor` | Core trading loop |
| `sync_state` | Publishes NAV + Firestore updates |
| `dashboard` | Streamlit front-end |

**Restart after patches:**
```bash
sudo supervisorctl restart hedge:executor hedge:sync_state hedge:dashboard
