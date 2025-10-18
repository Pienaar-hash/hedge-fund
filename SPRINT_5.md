# SPRINT 5 – Execution Observability Refresh

## Goal
Improve execution visibility across the trading stack by standardising JSONL logging, enabling lightweight replay/analysis tools, and surfacing realtime health KPIs in the dashboard.

## Components Instrumented
- `execution/log_utils.py`: unified, thread-safe JSONL logger with atomic writes, rotation, and archive support.
- `execution/executor_live.py`: emits attempts, vetoes, orders, position snapshots, and heartbeats.
- `execution/order_router.py`: attaches latency metadata and ack/error breadcrumbs.
- `execution/risk_limits.py`: centralised risk veto logging with structured details.
- `execution/sync_daemon.py`: periodic heartbeat & latency reporting.
- `execution/leaderboard_sync.py`: publishes per-strategy attempted/executed counters and fill rates.
- `dashboard/main.py` & `dashboard/nav_helpers.py`: new **Execution** page showing KPIs, veto table, and heartbeat banner.
- `scripts/replay_logs.py`: reconstructs order flows; `scripts/smoke_exec_logging.py`: hermetic smoke.

## Log File Glossary
| Path | Description |
| ---- | ----------- |
| `logs/execution/orders_attempted.jsonl` | Enriched order attempt payloads (strategy, signal timestamp, NAV snapshot). |
| `logs/execution/risk_vetoes.jsonl` | Unified veto surface with reasons, thresholds, and context. |
| `logs/execution/orders_executed.jsonl` | Router/exchange acknowledgements capturing client IDs, latency, and fills. |
| `logs/execution/position_state.jsonl` | Post-fill position snapshots (qty, entry, PnL). |
| `logs/execution/sync_heartbeats.jsonl` | Daemon/service heartbeats with lag metrics. |
| `logs/execution/*/` | Smoke or replay artefacts (per-run isolated folders). |

## Rotation & Archive Policy
- Each logger rotates at ~10 MB (`max_bytes=10_000_000`) keeping 5 backups.
- Oldest segment is gzipped to `logs/archive/<filename>-<timestamp>.gz`.
- Rotation is atomic (temp file swap) for crash-safety.

## Replay & Analysis
- `scripts/replay_logs.py` reconstructs attempt → veto/order chains, computes coverage and latency quantiles, and can export JSON for downstream analysis.
- `dashboard/main.py` consumes `exec_stats` (Firestore/local) to render KPIs and top veto reasons.
- Latency summaries can be cached via replay JSON output for dashboard display.

## Dashboard View
Open the Streamlit dashboard (default command: `streamlit run dashboard/app.py`) and navigate to the **Execution** tab to view:
- Heartbeat banner (green/amber/red).
- Attempted / Executed / Veto counts, Fill rate, Latency p50/p90.
- Top veto reasons (24 h window).

## Quick Commands
```bash
# Smoke test the logging pipeline (writes to isolated folder under logs/execution/)
python3 scripts/smoke_exec_logging.py

# Replay recent execution logs (export chains to /tmp/replay.json)
python3 scripts/replay_logs.py --since "$(date -u +%FT%TZ)" --json /tmp/replay.json

# Launch dashboard and inspect the Execution page
streamlit run dashboard/app.py
```
