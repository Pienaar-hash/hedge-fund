## v7 Dashboard Spec (current code)

### Surfaces
- `dashboard/app.py`: Streamlit app wiring panels.
- Panels: `nav_helpers.py`, `pipeline_panel.py`, `intel_panel.py`, `router_health.py`, `router_policy.py`, `live_helpers.py`, `dashboard_utils.py`.

### Data sources
- Reads JSON/JSONL from `logs/state/*` produced by `state_publish.py`/`sync_state.py`.
- Risk/telemetry: veto logs (`logs/execution/risk_vetoes.jsonl`), router metrics (state/router JSON), intel scores.

### Layout (as coded)
- NAV/AUM cards: nav_total, equity breakdown, freshness flags.
- Pipeline panel: attempts/emitted/submitted counts, per-symbol intents, signal metadata.
- Intel panel: symbol scores/offsets, feedback allocator signals.
- Router health: maker/taker mix, rejects, slippage, post-only behaviour; policy view from `router_policy.py`.
- Live helpers: recent positions/orders, health checks.
- Dashboard utils: caching/formatting helpers.

### KPIs & charts
- Donut/summary for AUM across assets, simple tables for positions/orders.
- Metrics: router slippage_bps, fill ratios, veto reasons counts, nav_age_s.
- ATR/regime/Sharpe placeholders referenced via intel panels; actual values pulled from state/intel when available.

### Reading contract
- Expects `logs/state/nav.json`, `positions.json`, `router.json`, `pipeline.json` (if emitted), intel state files; tolerates missing files with fallbacks.
- Uses `nav_helpers` to format NAV and PnL; `pipeline_panel` to render intents/attempts.

### v7 improvements (implied hooks)
- Additional KPIs can be added by extending `state_publish.py` outputs and reading in `dashboard_utils`.
- Router tuning surfaces via `router_policy.py` and state router metrics.
