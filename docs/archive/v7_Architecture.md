## v7 Architecture (current repo)

### High-level flow
- Screener (`execution/signal_screener.py`) builds intents from strategy_config, unlevered sizing, emits gross_usd/qty + leverage metadata.
- Executor (`execution/executor_live.py`) consumes intents, routes through risk (`risk_limits.py` / `risk_engine_v6.py`), then router (`order_router.py`) to place orders via `exchange_utils`.
- NAV + exposure derived from `nav.py` (exchange snapshots), persisted via `state_publish.py` and mirrored by `sync_state.py`.
- Telemetry/log surfaces: JSON/JSONL under `logs/state/*`, veto logs in `logs/execution/risk_vetoes.jsonl`, order/metrics via `events.py`, router metrics via `router_metrics.py`.
- Dashboard reads state files (`nav_helpers.py`, `pipeline_panel.py`, `router_health.py`, `intel_panel.py`) to render live cards.

### Runtime loop summary
- Intent generation: resolve universe/tier → fetch klines/prices → compute signals (RSI/zscore) → size unlevered gross_usd via per_trade_nav_pct/min_notional → attach leverage metadata → emit decisions/logs.
- Risk: `check_order` applies nav freshness, min_notional, per-symbol caps, per-trade caps (trade_equity_nav_pct/max_trade_nav_pct), portfolio cap, concurrency, tier caps, drawdown/daily loss, symbol guards.
- Routing: `route_intent` → maker-first limit with adaptive offsets; post-only rejects limited; fallback to taker when spread/offset bounds hit; chunking via min child notional; monitors fills and refreshes.
- State/telemetry: `state_publish` writes consolidated runtime state; `sync_state` mirrors for dashboard; `events.py` writes JSONL events; `pnl_tracker`, `router_metrics` feed metrics.

### State flow
- NAV snapshots: `logs/cache/nav_confirmed.json` (nav_total, sources_ok, age_s) via `nav.py`.
- Risk veto log: `logs/execution/risk_vetoes.jsonl`.
- State files: portfolio/positions/risk/health written by `state_publish.py` into `logs/state/`.
- Router telemetry: order events/metrics from `events.py`, `router_metrics.py`.
- Dashboard ingestion: helpers read `logs/state` JSON + veto/metrics files.

### Active JSON/JSONL surfaces
- `logs/cache/nav_confirmed.json` (nav_total, sources_ok, ts).
- `logs/execution/risk_vetoes.jsonl` (risk veto events).
- `logs/state/*.json` from `state_publish` (positions, nav, risk summary, router state).
- Event streams from `execution/events.py` (orders/fills/health).

### Config surfaces
- `config/strategy_config.json` (universe, per_trade_nav_pct, leverage metadata, signal params).
- `config/risk_limits.json` (fractional caps, per-symbol limits, testnet overrides).
- `config/runtime.yaml` (router/runtime tunables), `config/pairs_universe.json`, `config/symbol_tiers.json`, `config/strategy_registry.json`.

### NAV/AUM sources
- `nav.py` pulls futures balances/positions, computes trading NAV; includes spot treasury when configured.
- `PortfolioSnapshot` aggregates exposure/gross; `nav_health_snapshot` tracks age/sources_ok.

### Daemons/services
- `scripts/pipeline_shadow_heartbeat.py`, `scripts/pipeline_compare_service.py` (shadow/compare).
- `scripts/strategy_probe.py` (manual probe), `scripts/run_executor_once.sh` (one-shot), `bin/run-executor.sh`, `bin/run-dashboard.sh`, `bin/run-sync.sh`.

### Risk tuning hooks (v7)
- Fractional caps normalized via `risk_loader.normalize_percentage`.
- Per-trade sizing is unlevered in screener; leverage only metadata.
- Caps enforced solely in `risk_limits.py`/`risk_engine_v6.py`; router/executor are pass-through for gross_usd/qty.
