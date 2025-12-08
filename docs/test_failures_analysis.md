# Test Failures Analysis Report

**Date:** December 7, 2025  
**Branch:** v7-risk-tuning

## Summary

This report analyzes 35+ test failures from the hedge-fund project, categorizing them by root cause and providing recommendations.

---

## Category 1: STALE_MOCKS - Tests with Outdated Mock Paths or Return Values

These tests have stale expectations that no longer match the current implementation. **Recommend: Fix tests to align with current API.**

### 1.1 `tests/test_config_parsing.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_strategy_config_exists_and_parses` | Test expects `capital_per_trade` in strategy params, but config now uses `per_trade_nav_pct` | **FIX**: Update assertion to check for `per_trade_nav_pct` instead |
| `test_caps_consistency_between_files` | Config files have diverged: `pairs_universe.json` and `risk_limits.json` have different `max_nav_pct` values for same symbols | **FIX**: Either sync config files or update test to be more tolerant |

### 1.2 `tests/test_dashboard_metrics.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_signal_attempts_summary_parses_latest` | `signal_attempts_summary()` API changed - now returns line counts instead of parsed metrics | **FIX**: Update test expectations to match new format: `"attempt lines=X · emitted lines=Y"` |
| `test_signal_attempts_summary_missing` | Same API change - now returns `"X screener log lines."` instead of `"Signals: N/A"` | **FIX**: Update expected output |
| `test_nav_snapshot_prefers_state` | `get_nav_snapshot()` now reads from `series` array and uses `equity` fallback differently | **FIX**: Update test mock payload structure to match new parsing logic |

### 1.3 `tests/test_drawdown_normalization.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_already_fractional_stays_fractional` | `_normalize_observed_pct()` now ALWAYS divides by 100 (per spec change) - no more conditional logic | **DELETE**: This test is testing obsolete behavior; current design always assumes percent-style input |

### 1.4 `tests/test_order_router_routing_modes.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_broken_quality_forces_taker` | Reason string changed from `policy_quality_not_good` to `policy_quality_broken` | **FIX**: Update assertion to check for `policy_quality_broken` |
| `test_missing_orderbook_routes_taker_with_reason` | Reason string changed from `missing_maker_price` to `maker_submit_failed` | **FIX**: Update assertion or investigate if this is correct behavior |
| `test_router_health_snapshot_schema` | Test references `_build_router_health_snapshot` which may have changed signature | **FIX**: Investigate executor_live implementation |

### 1.5 `tests/test_passE_caps.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_check_order_respects_symbol_cap` | `check_order()` returns `False` (vetoed) but test expects `True` - likely a mock issue or logic change | **FIX**: Debug to understand why order is being vetoed; may need additional mocks |
| `test_check_order_returns_reasons_on_veto` | NAV=0 triggers different veto behavior | **FIX**: Update mocks to properly test veto reasons |

### 1.6 `tests/test_router_policy.py` (4 failures)

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_classify_router_quality_tiers_execution_intelligence` | v6.5 added bootstrap mode: returns "ok" when `sample_count` < 20 (MIN_SAMPLES_FOR_QUALITY). Test doesn't provide `sample_count`. | **FIX**: Add `sample_count: 50` to test dicts |
| `test_router_policy_disables_maker_when_broken_execution_intelligence` | Same bootstrap mode issue - mocked metrics don't include `sample_count` so it defaults to bootstrap ("ok") | **FIX**: Add `sample_count` to mock return values |
| `test_router_policy_prefers_maker_when_good_execution_intelligence` | Same issue | **FIX**: Add `sample_count` |
| `test_classify_router_regime_variants` | Thresholds changed in v6.3/v6.5 - `fallback_rate: 0.65` no longer triggers "fallback_heavy" | **FIX**: Update threshold test values or update expected results |

### 1.7 `tests/test_pipeline_v6_compare_runtime.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_sizing_diff_stats` | `compare_pipeline_v6()` no longer returns `sizing_diff_stats` in summary | **FIX**: Update test to match new return schema or restore feature |

---

## Category 2: API_CHANGED - Tests Where Underlying API Changed

These tests fail because the implementation API has changed significantly. **Recommend: Fix or delete based on whether the tested functionality still exists.**

### 2.1 `tests/test_exchange_dry_run.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_market_reduce_only_keeps_quantity_and_reduce_flag` | `send_order()` now strips `reduceOnly` flag from MARKET orders (per Binance API changes) | **FIX**: Update test - MARKET orders should NOT include `reduceOnly`; implementation is correct |

### 2.2 `tests/test_execution_hardening_router.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_post_only_fallback_execution_hardening` | `submit_limit()` no longer retries automatically on rejection; fallback is controlled differently | **FIX**: Update test to match new `submit_limit()` behavior |

### 2.3 `tests/test_nav_age.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_nav_age_selects_newest` | `_NAV_SNAPSHOT_PATHS` attribute may not exist; `get_nav_freshness_snapshot()` returns `(None, ...)` | **FIX**: Update test to use current NAV freshness mechanism |

### 2.4 `tests/test_nav_fail_closed.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_nav_fresh_ok` | `is_nav_fresh()` API may have changed or module reload behavior differs | **FIX**: Verify `is_nav_fresh()` signature and fix mock paths |

### 2.5 `tests/test_nav_modes.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_compute_nav_pair_defaults` | `compute_nav_pair()` function signature or return value changed | **FIX**: Update test to match current `nav.py` API |

### 2.6 `tests/test_nav_risk_unification.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_enhanced_nav_mark_price_conversion` | Mock for `get_balances()` returns list format but implementation expects dict; also `get_price()` mock is bypassed by real network calls | **FIX**: Update mock format and ensure no network calls are made |

### 2.7 `tests/test_infra_v5_5.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_router_health_aggregation` | `load_router_health()` signature changed - no longer accepts `signal_path`/`order_path` kwargs | **FIX**: Update to new API: `load_router_health(window=N, snapshot=dict)` |

### 2.8 `tests/test_router_health_events.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_load_router_health_from_order_events` | `ORDER_EVENTS_PATH` attribute doesn't exist on `router_health` module | **FIX**: Remove monkeypatch or find correct attribute path |

### 2.9 `tests/test_router_health_v2.py` (2 failures)

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_router_health_confidence_and_sharpe` | Same `ORDER_EVENTS_PATH` issue | **FIX**: See above |
| `test_router_health_uses_state_snapshot` | May work if `ORDER_EVENTS_PATH` issue is fixed | **FIX**: Verify after fixing module attribute |

### 2.10 `tests/test_passE_testnet_overrides.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_testnet_overrides_activate` | Testnet override value for `max_nav_drawdown_pct` may have changed from 95.0 | **FIX**: Check current testnet override values in `risk_loader.py` |

### 2.11 `tests/test_treasury_nav.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_nav_summary_trading_only` | `compute_nav_summary()` return schema changed - no longer has separate `treasury_nav` field | **FIX**: Update test expectations to match new schema |

---

## Category 3: DEPRECATED - Tests for Removed/Obsolete Functionality

These tests are testing functionality that has been intentionally removed or deprecated. **Recommend: Delete.**

### 3.1 `tests/test_execution_hardening_signals.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_asset_universe_usdc_only_execution_hardening` | `is_in_asset_universe()` now allows USDC, USDT, AND FDUSD suffixes (not USDC-only) | **DELETE**: Test was for USDC-only mode which is no longer the policy |

---

## Category 4: ENV_DEPENDENT - Tests Failing Due to Module/Environment Issues

These tests fail due to missing module attributes or import-time side effects. **Recommend: Fix test isolation.**

### 4.1 `tests/test_executor_reduce_only_position_side.py` (3 errors)

| Test | Issue | Recommendation |
|------|-------|----------------|
| All 3 tests | `executor_live._RISK_GATE` attribute doesn't exist; `_PORTFOLIO_SNAPSHOT` may also be missing | **FIX**: Update fixture to mock correct module-level attributes or refactor test approach |

### 4.2 `tests/test_screener_sizing_no_leverage.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_screener_sizing_ignores_leverage` | Returns 2 intents instead of 1 - vol_target strategy is also generating signals | **FIX**: Mock `_load_strategy_list()` to return only the test strategy, or filter intents in assertion |

### 4.3 `tests/test_signal_pipeline.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_trend_filter_blocks_counter_trend_sell` | Vol_target strategy generates a BUY signal even when zscore-based SELL is blocked | **FIX**: Need to also mock/disable vol_target strategy or adjust test expectations |

### 4.4 `tests/test_symbol_filters.py`

| Test | Issue | Recommendation |
|------|-------|----------------|
| `test_screener_respects_binance_floors` | `generate_signals_from_config()` is loading real config, not using mocked `fake_open` | **FIX**: Properly mock the config loading mechanism |

### 4.5 `tests/test_telegram_v7.py` (2 failures)

| Test | Issue | Recommendation |
|------|-------|----------------|
| 2 tests | **RESOLVED**: These tests are now passing | **NO ACTION NEEDED** |

---

## Consolidated Recommendations

### Priority 1: Quick Fixes (Low Risk)
1. `test_config_parsing.py` - Update key names (`capital_per_trade` → `per_trade_nav_pct`)
2. `test_dashboard_metrics.py` - Update expected output strings
3. `test_order_router_routing_modes.py` - Update reason strings in assertions
4. `test_exchange_dry_run.py` - Accept that MARKET orders don't include `reduceOnly`

### Priority 2: Delete Obsolete Tests
1. `test_drawdown_normalization.py::test_already_fractional_stays_fractional` - Behavior changed by design
2. `test_execution_hardening_signals.py::test_asset_universe_usdc_only_execution_hardening` - Policy changed

### Priority 3: Fix Stale Mocks (Medium Effort)
1. `test_infra_v5_5.py` - Update `load_router_health()` call signature
2. `test_router_health_*.py` - Remove `ORDER_EVENTS_PATH` monkeypatch
3. `test_nav_*.py` - Update NAV module mocks to match current API
4. `test_executor_reduce_only_position_side.py` - Fix fixture to use existing module attributes

### Priority 4: Investigate & Decide
1. `test_passE_caps.py` - Debug why orders are being vetoed
2. `test_signal_pipeline.py` - Decide if vol_target strategy interference is acceptable
3. `test_telegram_v7.py` - Identify root cause

---

## Summary Statistics

| Category | Count | Action |
|----------|-------|--------|
| STALE_MOCKS | 17 | Fix tests |
| API_CHANGED | 12 | Fix or review |
| DEPRECATED | 2 | Delete tests |
| ENV_DEPENDENT | 4 | Fix isolation |
| RESOLVED | 2 | Already passing |
| **Total** | **37** | |
