## v7 Telemetry Contract

### Publishers
- `execution/state_publish.py`: writes consolidated state JSON files under `logs/state/` (positions, nav, risk, router).
- `execution/sync_state.py`: mirrors/cleans state files for dashboard consumption.
- `execution/events.py`: JSONL event writer for orders/fills/health.
- `execution/router_metrics.py`: aggregates router metrics/slippage/rejects.
- `execution/trade_logs.py`, `pnl_tracker.py`: trade/PnL logs.
- Risk vetoes: `_emit_veto` in `risk_limits.py` → `logs/execution/risk_vetoes.jsonl`.

### Schemas (key fields)
- `logs/state/nav.json`: nav_total, nav_age_s, sources_ok, equity breakdown.
- `logs/state/positions.json`: positions list with symbol, qty, entry/mark, leverage, unrealized PnL.
- `logs/state/risk.json`: recent veto reasons, thresholds, nav freshness flags, drawdown stats.
- `logs/state/router.json`: recent router decisions, post-only rejects, taker fallbacks, slippage_bps, offsets used.
- `logs/state/intel.json` (if emitted): intel scores/offset suggestions.
- `logs/execution/risk_vetoes.jsonl`: per veto event with symbol, veto_reason, thresholds/observations, nav_total.
- `events` JSONL (orders/fills): symbol, side, price, qty, order_id, status, timestamps, is_maker, slippage_bps, rejections.

### Update cadence
- `state_publish.py` typically runs each executor loop; `sync_state.py` keeps dashboard copies in sync.
- Event streams are append-only per order/fill occurrence.

### Consumers
- Dashboard panels (`nav_helpers`, `pipeline_panel`, `router_health`, `intel_panel`) read `logs/state/*` and risk/router event logs.
- External monitoring can tail JSONL files for risk/router/error alerts.

### Invariants
- Fields expressed in fractions where applicable (caps, pct).
- gross_usd/qty are unlevered intents; leverage metadata preserved.
- nav_total from `nav_health_snapshot` used for risk/telemetry; stale nav triggers warnings/veto.
