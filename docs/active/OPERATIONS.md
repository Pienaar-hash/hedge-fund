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
```

## Episode-ledger rebuild observation

`logs/state/episode_ledger_checkpoint.json` is an implementation checkpoint, not a trading control. Do not edit or delete it to improve timing. Inspect the append-only `logs/execution/episode_ledger_rebuild.jsonl` stream instead:

```bash
tail -n 20 logs/execution/episode_ledger_rebuild.jsonl
python3 -m json.tool logs/state/episode_ledger_checkpoint.json
```

Expected steady-state rows use `mode: incremental`, with `bytes_read: 0` on no-op cycles or only the appended JSONL range on new fills. `full_fallback` is a correctness safeguard for changed, truncated, replaced, or missing sources; corrupt or incompatible checkpoint state; hash divergence; or late input. Its `fallback_reason` is mandatory. A `forced_full` row is produced only by the explicit operator API. Telemetry is fail-open and must never be used to change execution configuration or trading controls.
