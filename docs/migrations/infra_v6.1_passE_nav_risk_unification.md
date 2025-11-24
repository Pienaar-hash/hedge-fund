# infra_v6.1 PASS E — NAV, Risk, and State Unification

## Scope

PASS E aligns NAV computation, risk gates, and state baselines so testnet and prod share the same equity view:

- Enhanced NAV engine with explicit asset inclusion, mark-price conversion, and freshness metadata.
- Risk gates consume the same NAV source and ignore stale peak states.
- Sync/state writers fall back to confirmed NAV so dashboards/testnet runs stay representative.

## Target modules

- `execution/nav.py`
- `execution/risk_limits.py`
- `execution/sync_state.py`
- `execution/executor_live.py`
- `execution/utils/execution_health.py`
- `config/runtime.yaml`
- Tests: `tests/test_nav_risk_unification.py`

## Changes shipped

- **NAV engine**: optional `nav.include_assets` + `use_mark_price` config (runtime.yaml) to value USDT/USDC 1:1 and convert BTC/majors via ticker price. NAV snapshots now expose `assets`, `mark_prices`, `nav_mode`, `freshness` (balances/marks/cache age), and `ts`; runtime overrides merge into the strategy config for writers and are mirrored into state files.
- **Peak-state healing**: runtime `risk.reset_peak_on_testnet` and `risk.peak_stale_seconds` guard the drawdown baseline. Testnet boots auto-reset peak to current NAV. Stale peak files are regenerated from the latest nav_log tail (fallback to confirmed NAV) and written back for risk + dashboard.
- **Risk gates**: drawdown/daily-loss caps use unified NAV health + regenerated peak: drawdown pct = (peak_nav - nav)/peak_nav; daily loss pct = (daily_peak - nav)/daily_peak. NAV health (`fresh`, `sources_ok`) is honored before applying drawdown vetoes; stale peak states emit warnings instead of blocking. Diagnostics now include `nav_health`, `peak_state`, `drawdown`, `daily_loss`, and `assets`.
- **State baseline**: sync_state uses confirmed NAV when nav_log is cold/empty, seeds a synthetic row, and writes the healed peak; executor writes enhanced nav snapshot into `logs/state/nav.json` and `logs/state/synced_state.json` so dashboards see the enriched structure.
- **Tests**: coverage for mark-price NAV conversion, testnet peak reset, stale peak regeneration, unified drawdown math, and risk veto behavior across PASS E paths.

## Before → after (diagrams)

- **Before:** NAV detail limited to futures wallet; peak_state could go stale on testnet, leading to spurious drawdown vetoes; state files lacked asset/mark metadata.
- **After:** NAV snapshot carries assets/marks/freshness; testnet boots rewrite peak_state to current NAV; stale peak_state heals from nav_log; state files carry the enriched snapshot and risk gates use the same baseline.

### NAV flow (after)
1. runtime.yaml `nav` block merges into strategy config.
2. `compute_trading_nav` normalizes balances, converts configured assets via mark price, emits `{nav, assets, mark_prices, freshness, ts}`.
3. Executor/state writers persist the enriched snapshot to `logs/state/nav.json` and `logs/state/synced_state.json`.
4. Risk/peak heuristics read the same snapshot (`get_confirmed_nav.detail`) for diagnostics.

### Peak-state logic
- **Reset (testnet):** if `risk.reset_peak_on_testnet` and BINANCE_TESTNET=1 → `peak_nav = current_nav`, `daily_peak = current_nav`.
- **Stale:** if `peak_ts` older than `risk.peak_stale_seconds` → rebuild from nav_log tail; fallback to confirmed NAV when too few samples; persist healed peak_state.
- **Fresh:** retain peak/daily values; compute drawdown/daily loss from `peak_nav`/`daily_peak`.

### Risk gating examples
- NAV stale + fail_closed → veto `nav_stale`; fail_open → warn.
- Peak stale but NAV fresh → warn `drawdown_stale`, skip drawdown veto.
- Peak 1,000 / NAV 800 / limit 10% → drawdown 20% → veto `nav_drawdown_limit`.
- Daily peak 900 / NAV 810 / day limit 5% → daily loss ~10% → veto `day_loss_limit`.

## Tests and acceptance

- `tests/test_nav_risk_unification.py`: enhanced NAV asset inclusion + mark pricing; testnet peak reset; stale peak regen; unified drawdown math/veto.
- `tests/test_executor_state_files.py`: executor writes nav state with enriched snapshot.
- Acceptance: dashboard/state files carry enhanced nav snapshot; drawdown limits only fire on healthy NAV + peak; testnet runs start with fresh peak; stale peak heals automatically.

## Acceptance criteria

- NAV detail includes configured assets, mark prices, and freshness flags; source reports `nav_mode=enhanced` when config is active.
- Drawdown/daily-loss risk vetoes only fire on fresh peak/NAV data; stale flags log warnings instead of blocking.
- Sync/state writes still occur (with sensible totals) when nav_log is empty but a confirmed NAV exists.
