## v7 Risk Model (current repo)

### Sources
- `execution/risk_limits.py`, `execution/risk_engine_v6.py`
- Helpers: `risk_loader.py`, `universe_resolver.py`, `drawdown_tracker.py`, `nav.py`

### Inputs
- Intent fields: symbol, side, requested_notional (gross_usd from screener, unlevered), price, nav, open_qty, lev (metadata), tier_name, current gross/tier gross, open_positions_count.
- Config: `config/risk_limits.json` (fractional caps), per-symbol limits, testnet overrides; runtime env DRY_RUN/BINANCE_TESTNET.
- NAV snapshot: `nav_health_snapshot` (nav_total, age_s, sources_ok, fresh).
- Positions: live positions via `exchange_utils.get_live_positions`/state.live_positions.

### Normalization
- `normalize_percentage` treats >1 as percent (/100) else fraction.
- Config normalized on load (global + per-symbol). Drawdown shares/percents normalized to fractions.
- NAV uses `nav_total` when provided; leverage not applied to requested_notional.

### Checks (order of evaluation)
1. Restricted symbols guard, non-prod dry-run guard.
2. Universe membership via `universe_by_symbol`.
3. NAV freshness (threshold from config or default); veto nav_stale with fail-closed behaviour.
4. Already in trade guard (live position amount >0).
5. Whitelist guard (optional).
6. Drawdown/daily loss guardrails via `drawdown_tracker`.
7. Min notional (global/per-symbol) and max_order_notional (symbol) blocks.
8. Per-symbol notional cap (`max_nav_pct` or symbol_notional_share_cap_pct) applied to nav_total.
9. Max open qty (symbol), side block (block_sides).
10. Leverage cap (symbol/global) on provided lev metadata.
11. Burst limit (orders per window) when configured.
12. Per-trade caps:
    - `trade_equity_nav_pct` → reason `trade_gt_equity_cap`
    - `max_trade_nav_pct` → reason `max_trade_nav_pct`
    Observations include trade_nav_obs; thresholds carry normalized fractions.
13. Portfolio gross cap (`max_portfolio_gross_nav_pct`/`max_gross_nav_pct`) → reason `portfolio_cap`.
14. Tier caps (tiers.per_symbol_nav_pct applied to nav_total, uses tier_name/current_tier_gross_notional) → reason `tier_cap`.
15. Max concurrent positions (global or per-symbol override) → reason `max_concurrent`.
16. Symbol drawdown cap via `symbol_dd_guard` (fractional dd vs cap) → disable symbol for cooldown.

### Outputs
- `(veto: bool, details: dict)` where reasons contains first triggered reason; thresholds/observations include caps and nav data; nav_total and asset breakdown attached when available.
- Veto events logged via `_emit_veto` to `logs/execution/risk_vetoes.jsonl`.

### Risk engine wrapper
- `risk_engine_v6.py` orchestrates account checks, positions sync, applies `risk_limits.check_order` per intent, accumulates veto reasons, respects DRY_RUN/testnet flags, publishes telemetry via state writers.

### Testnet overrides
- From `risk_limits.json.testnet_overrides` when BINANCE_TESTNET=1: adjusts drawdown/loss thresholds, sets `_meta.testnet_overrides_active`.

### Veto schema (canonical reasons)
- `restricted_symbol`, `nav_stale`, `min_notional`, `symbol_cap`, `trade_gt_equity_cap`, `max_trade_nav_pct`, `portfolio_cap`, `tier_cap`, `max_concurrent`, `leverage_exceeded`, `burst_limit`, `not_whitelisted`, `side_blocked`, plus drawdown/daily_loss limits.
