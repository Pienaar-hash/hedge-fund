# Dashboard Data Freshness Audit Report

**Date:** 2025-11-27  
**Auditor:** Automated Audit  
**Scope:** Trade count reduction, symbol health staleness, V7 Risk KPIs staleness

---

## Executive Summary

**All reported staleness concerns are FALSE POSITIVES.** The dashboard data is current and functioning correctly. The observed trade count reduction from 50+ to 48 is **expected behavior** due to the 48-hour rolling window configuration.

---

## Issue 1: Trade Count Reduced from 50+ to 48

### Root Cause: Rolling Window Expiration

The trade counting mechanism uses a **48-hour rolling window** (`lookback_hours: 48.0`), meaning trades automatically "age out" of the count after 48 hours.

**Evidence from `expectancy_v6.json`:**
```json
{
  "sample_count": 48,
  "lookback_hours": 48.0,
  "updated_ts": "2025-11-27T07:49:05.971790+00:00"
}
```

### Trades Near the 48h Boundary (as of 08:22 UTC)

| Timestamp (UTC) | Hours Ago | Symbol | Status |
|-----------------|-----------|--------|--------|
| 2025-11-25T04:42:00 | 51.7h | SOLUSDT | **Aged out** |
| 2025-11-25T04:47:29 | 51.6h | ETHUSDT | **Aged out** |
| 2025-11-25T04:47:33 | 51.6h | ETHUSDT | **Aged out** |
| 2025-11-25T06:27:49 | 49.9h | ETHUSDT | **Aged out** |
| 2025-11-25T06:27:53 | 49.9h | ETHUSDT | **Aged out** |
| 2025-11-25T06:34:10 | 49.8h | SOLUSDT | **Aged out** |
| 2025-11-25T06:51:59 | 49.9h | BTCUSDT | **Aged out** |
| 2025-11-25T12:04:29 | 44.3h | BTCUSDT | Still in window |

**Conclusion:** Overnight, multiple trades from Nov 25 crossed the 48-hour threshold and were removed from the rolling count. This is **normal behavior**, not a bug.

### Trade Distribution by Symbol (Current 48h Window)

| Symbol | Count |
|--------|-------|
| SOLUSDT | 29 |
| ETHUSDT | 5 |
| BTCUSDT | 4 |
| LTCUSDT | 2 |
| LINKUSDT | 2 |
| SUIUSDT | 2 |
| WIFUSDT | 2 |
| DOGEUSDT | 2 |
| **Total** | **48** |

---

## Issue 2: Symbol Health Appears Stale

### Status: ✅ NOT STALE

**Evidence from `router_health.json`:**
```json
{
  "updated_ts": 1764229416  // Recent timestamp
}
```

### Current Symbol Health Status

| Symbol | Status | Fallback Rate | Avg Latency | Notes |
|--------|--------|---------------|-------------|-------|
| BTCUSDT | **degraded** | 1.0 | 71.41ms | Full fallback mode |
| ETHUSDT | **degraded** | 1.0 | 71.63ms | Full fallback mode |
| SOLUSDT | **degraded** | 1.0 | 38.55ms | Full fallback mode |
| LTCUSDT | ok | 0.0 | 29.84ms | Normal |
| LINKUSDT | ok | 0.0 | 35.67ms | Normal |
| SUIUSDT | ok | 0.0 | 30.66ms | Normal |
| WIFUSDT | ok | 0.0 | 52.48ms | Normal |
| DOGEUSDT | ok | 0.0 | 29.02ms | Normal |

**Health Summary:** 5 symbols OK, 3 symbols degraded  
**Router Quality:** `poor` (due to high fallback ratio on major pairs)

---

## Issue 3: V7 Risk KPIs Appear Stale

### Status: ✅ NOT STALE

**Evidence from `kpis_v7.json`:**
```json
{
  "ts": 1764229421  // Recent timestamp
}
```

### Current V7 Risk KPIs

| Metric | Value | Status |
|--------|-------|--------|
| ATR Regime | `normal` | ✅ |
| Drawdown State | `defensive` | ⚠️ Currently in defensive mode |
| Fee/PnL Ratio | 0.573 | ⚠️ Fees consuming 57% of PnL |
| Maker Fill Ratio | 0.0 | ⚠️ No maker fills |
| Fallback Ratio | 1.0 | ⚠️ 100% fallback routing |
| Router Quality | `poor` | ⚠️ Degraded routing quality |

---

## Issue 4: NAV State File

### Status: ✅ POPULATED AND CURRENT

The `nav_state.json` file contains active time-series data with 378+ entries spanning from 2025-11-26 21:32 to 2025-11-27 08:22 UTC.

**Current NAV Metrics:**
- **Total Equity:** $4,183.28
- **Peak Equity:** $4,231.56
- **Realized PnL Today:** -$29.37
- **Unrealized PnL:** +$6.96
- **Drawdown:** 1.14%
- **Last Update:** 2025-11-27T08:22:22 UTC

---

## System Health Summary

| Component | Status | Last Update | Notes |
|-----------|--------|-------------|-------|
| Trade Count | ✅ Working | 07:49 UTC | Rolling 48h window operating correctly |
| Symbol Health | ✅ Fresh | Recent | 3 major pairs degraded (fallback mode) |
| V7 KPIs | ✅ Fresh | Recent | System in defensive mode |
| NAV State | ✅ Fresh | 08:22 UTC | Continuous updates every ~1.3 min |
| Order Execution | ✅ Active | Recent | 808 orders logged, recent activity visible |

---

## Recommendations

1. **No action required for trade count** – The reduction is expected behavior from the 48-hour rolling window. Trade count will increase as new trades are executed.

2. **Monitor router quality** – Three major pairs (BTC, ETH, SOL) are in fallback mode with 100% fallback rate. This may indicate:
   - Primary exchange connectivity issues
   - Liquidity routing problems
   - Configuration that needs review

3. **Review defensive mode** – System is in `dd_state: defensive`. Consider whether market conditions warrant this or if it can be relaxed.

4. **Fee optimization** – Fee/PnL ratio of 57% suggests fees are consuming a significant portion of profits. Consider:
   - Increasing trade size to amortize fixed fees
   - Improving maker fill ratio (currently 0%)
   - Reviewing execution timing

---

## Technical Details

### State File Paths
```
/root/hedge-fund/logs/state/
├── expectancy_v6.json    # Trade statistics (48h rolling)
├── kpis_v7.json          # V7 Risk KPIs
├── nav_state.json        # NAV time series
├── router_health.json    # Symbol health metrics
├── positions_v7.json     # Open positions
└── orders_executed.jsonl # Order execution log (808 entries)
```

### Relevant Code
- **Trade counting:** `dashboard/state_v7.py` → `_calculate_alltime_futures_stats()`
- **KPI rendering:** `dashboard/kpi_panel.py` → `render_kpi_overview()`
- **Router health:** `dashboard/router_health.py`

---

**End of Audit Report**
