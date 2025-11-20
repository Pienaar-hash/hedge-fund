# v6 Risk Engine Contract

## Configuration inputs
- `config/risk_limits.json:1-74` holds global caps (`daily_loss_limit_pct`, `max_nav_drawdown_pct`, `max_trade_nav_pct`, etc.), per-symbol overrides (min notional, max order notional, leverage, nav % caps), and non-prod guards.
- `config/pairs_universe.json:1-80` defines tiers, enablement flags, and additional caps (`caps.max_nav_pct`, `max_concurrent_positions`).
- Runtime gating uses `execution/risk_limits._RISK_CFG` (`execution/executor_live.py:506-515`) loaded once at executor start and cached for risk + pipeline.

## Runtime state
- `RiskState` and `RiskGate` come from `execution/risk_limits.py`. `_load_registry_entries()` + `RiskAutotuner` restore/persist counters in `logs/cache/risk_state.json` (`execution/risk_autotune.py:1-210`).
- NAV freshness is enforced before every trade via `enforce_nav_freshness_or_veto()` (`execution/risk_limits.py:100-160`), which reads `logs/cache/nav_confirmed.json` and `logs/nav_log.json`.

## RiskEngineV6
- Defined in `execution/risk_engine_v6.py:1-114`.
- `OrderIntent`: typed container for symbol, side, qty, notional, leverage, nav, open positions, etc.
- `RiskDecision`: typed result with `.allowed`, `.clamped_qty`, `.reasons`, `.hit_caps`, `.diagnostics`.
- `RiskEngineV6.check_order()` delegates to `risk_limits.check_order()` (same file, lines 55-95) but always returns structured diagnostics and ensures nav/tier/current gross metadata is passed along.
- `RiskEngineV6.build_risk_snapshot()` enumerates the entire universe via `execution.utils.execution_health.compute_execution_health()` so state files reflect router + volatility context, not just raw caps.

## Autotune + feedback
- `execution/risk_autotune.py:1-210` monitors doctor confidence, order metrics, and realized PnL to adjust `RiskGate` parameters (burst limits, trade caps). It also persists both the changes and the live state for continuity across restarts.
- `execution/intel/feedback_allocator_v6.py:260-402` consumes expectancy, scores, router policy, nav, and risk snapshots to suggest nav/trade/concurrent caps per symbol. These suggestions are currently advisory, but the structure matches the contract expected by potential future auto-apply logic.

## Telemetry + audits
- Risk decisions are logged to `logs/execution/risk_vetoes.jsonl` through `execution/risk_limits.LOG_VETOES` any time `check_order()` vetoes.
- Portfolio risk snapshots go to `logs/state/risk_snapshot.json` (see `execution/executor_live.py:280-309`). The schema is described in `v6_state_contract.md` and is consumed by the dashboard and allocator.
- Tests: `tests/test_risk_engine_v6.py`, `tests/test_risk_limits.py`, `tests/test_risk_autotune.py`, and `tests/test_feedback_allocator_v6.py:1-120` validate order intent translation, veto logic, autotune heuristics, and allocator scaling.

## Flags + runtime gating
- `execution/v6_flags.py:18-59` exposes `RISK_ENGINE_V6_ENABLED`. When false, the executor still uses legacy `RiskState` for gating, but telemetry/risk snapshots continue to refresh (ensuring state parity ahead of flag flips).
- `_maybe_write_v6_runtime_probe()` (`execution/executor_live.py:237-251`) records the live flag combination so audit logs can prove when v6 risk was active.

This contract replaces legacy v5 documentation—the only risk sources are the modules and files enumerated above.
