## Pass B — Config → NAV → Risk → State audit

Changes
- Centralised risk config loader `execution.risk_limits.load_risk_config()`; applies BINANCE_TESTNET overrides once, normalises globals/per_symbol, exposes `testnet_overrides_active`.
- `risk_engine_v6` now defaults to the central loader to stay in sync with overrides.
- NAV module hardened: asset breakdown surfaced in `compute_trading_nav` detail, cache/freshness reporting via `nav.nav_health_snapshot`, consistent freshness semantics reused by risk gates.
- Risk gating aligns to NAV health: `check_order` uses `nav_health_snapshot` for nav_stale logic, surfaces nav health + thresholds (including testnet override flags) in diagnostics.
- Operator helper `scripts/nav_debug.py` prints NAV detail, nav health, drawdown thresholds/observations, and a dummy RiskEngineV6 check.

State expectations
- `state_publish` continues to emit `logs/state/nav.json`, `risk_snapshot.json`, `symbol_scores_v6.json`, `synced_state.json` with canonical fields (items/nav/engine_version/v6_flags/updated_at). No contract changes; reserved/legacy fields are documented inline in code comments.

Follow-ups
- Consider threading nav_health into dashboard helpers for richer stale-source messaging.
- Align state readers to consume the new `cache_detail`/`nav_health` fields for operator debugging.
- Add intent-id alignment in pipeline_v6_compare to reduce symbol-only matching drift.
