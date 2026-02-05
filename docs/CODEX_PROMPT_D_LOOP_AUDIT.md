# Codex Prompt D — Loop Cadence & Sentinel SLA Audit

**Date:** 2026-02-05  
**Branch:** `phase-a3-observability`  
**Status:** Completed

---

## Executive Summary

The executor loop runs at **~90-150 seconds per iteration** instead of the expected 60 seconds.  
The primary bottleneck is **API call latency** in the signal screener, which makes **2 API calls per symbol** (klines + price) across ~11+ strategy symbols plus vol_target symbols.

**Root Cause:** Sequential, blocking REST calls to Binance API during signal generation.

---

## 1. Loop Structure Analysis

### Main Loop (`main()` at line 4854)
```
while True:
    _maybe_compute_sentinel_x()  # 1 klines call (300 bars)
    _loop_once(i)                # Main work
    _maybe_emit_heartbeat()      # Local only
    _maybe_run_internal_screener() # Rate-limited (SCREENER_INTERVAL=300s)
    _maybe_write_v6_runtime_probe()
    time.sleep(60)               # SLEEP=60s
```

### `_loop_once()` Structure
```
_sync_dry_run()
_refresh_risk_config()
_account_snapshot()           # 2 API calls: get_balances() + get_positions()
get_positions()               # 1 API call
exit_scanner                  # 0-N API calls (price per position if missing)
generate_intents()            # HEAVIEST: 2 calls × N symbols
```

---

## 2. API Call Inventory Per Loop

| Component | Function | API Calls | Data Size |
|-----------|----------|-----------|-----------|
| **Sentinel-X** | `get_klines("BTCUSDT", "15m", limit=300)` | 1 | 300 bars |
| **Account Snapshot** | `get_balances()` | 1 | - |
| **Account Snapshot** | `get_positions()` | 1 | - |
| **Position Check** | `get_positions()` (duplicate) | 1 | - |
| **Signal Screener** | `get_klines(sym, tf, limit=750)` per symbol | N | 750 bars each |
| **Signal Screener** | `get_price(sym)` per symbol | N | - |
| **Vol Target** | `get_klines(sym, tf, limit=150)` per vol_target symbol | M | 150 bars each |
| **Vol Target** | `get_price(sym)` per vol_target symbol | M | - |

**Current Universe:**
- Strategy symbols: ~11 (from `strategy_config.json`)
- Vol_target symbols: 3 (BTCUSDT, ETHUSDT, SOLUSDT)
- Total estimated API calls: **4 + 2×11 + 2×3 = 32+ calls per loop**

---

## 3. Latency Breakdown (Estimated)

| Section | Duration | Notes |
|---------|----------|-------|
| Sentinel compute | ~2-5s | 300-bar klines fetch |
| Account snapshot | ~1-2s | 2 API calls |
| Position check | ~0.5-1s | 1 API call (redundant) |
| Exit scanner | ~0-1s | Only if positions lack markPrice |
| **Signal generation** | **60-90s** | Main bottleneck |
| State emission | ~1-2s | JSON writes, Firestore sync |

**Total work time:** ~70-100s  
**Plus sleep:** 60s  
**Effective loop cadence:** ~130-160s

---

## 4. Root Causes

### 4.1 Sequential API Calls
The signal screener iterates over symbols **sequentially**, making 2 REST calls per symbol:
```python
for scfg in strategies:
    kl = get_klines(sym, tf, limit=750)  # Blocking REST call
    price = get_price(sym)                # Blocking REST call
```

With ~11 strategies × 2 calls @ ~3s/call average = **66+ seconds** just for API I/O.

### 4.2 Large Kline Requests
- Strategy symbols request **750 bars** (for vol regime computation)
- Vol_target requests **150 bars** (for ATR)
- Sentinel requests **300 bars**

### 4.3 No Caching
Every loop iteration fetches fresh klines for all symbols, even though:
- 15m candles only update every 15 minutes
- Many signals are immediately vetoed by doctrine

### 4.4 Duplicate Position Fetches
`get_positions()` is called twice per loop:
1. In `_account_snapshot()` 
2. Explicitly in `_loop_once()` for `baseline_positions`

---

## 5. Instrumentation Added

### New Module: `execution/loop_timing.py`
- **Activation:** `LOOP_TIMING_DEBUG=1`
- **Output:** `logs/execution/loop_timing.jsonl`
- **Threshold:** `LOOP_TIMING_THRESHOLD=5` (log warnings for >5s sections)

### Timing Hooks in `executor_live.py`
- `sentinel_compute` — Sentinel-X computation
- `loop_setup` — Config sync and refresh
- `account_snapshot` — Balance/position fetch
- `get_positions` — Baseline position fetch
- `exit_scanner` — Exit scanning with API call tracking
- `generate_intents` — Signal generation (main bottleneck)
- `heartbeat_emit`, `internal_screener`, `runtime_probe`

### API Call Tracking in `signal_screener.py`
- `record_api_call()` on each `get_klines()` and `get_price()` call

---

## 6. Recommendations

### Immediate (Low Risk)
1. **Reduce Kline Limits**
   - Strategy symbols: 750 → 200 (vol regime only needs ~50 for ratio)
   - Vol_target: 150 → 50 (ATR only needs ~20 lookback)
   - Estimated savings: ~30% API data transfer

2. **Dedupe Position Fetch**
   - Remove duplicate `get_positions()` in `_loop_once()`
   - Pass positions from `_account_snapshot()` result
   - Savings: 1 API call

3. **Increase Sentinel Threshold** ✓ Already Done
   - Changed from 300s to 420s to account for real loop cadence

### Medium-Term (Medium Risk)
4. **Add Kline Caching**
   ```python
   @lru_cache(ttl=60)  # or simple dict with timestamp
   def get_klines_cached(sym, tf, limit):
       ...
   ```
   - Cache TTL: 60s for 15m candles (4 candles per hour)
   - Estimated savings: ~50% API calls after warm-up

5. **Batch Price Fetches**
   - Binance supports `/fapi/v1/ticker/price` without symbol (returns all)
   - Single call instead of N calls
   - Savings: N-1 API calls

### Long-Term (Higher Risk)
6. **WebSocket for Prices**
   - Use `@markPrice` stream instead of REST polling
   - Eliminates all `get_price()` calls
   - Requires connection management

7. **Parallel API Calls**
   ```python
   async def fetch_all_klines(symbols):
       tasks = [get_klines_async(s) for s in symbols]
       return await asyncio.gather(*tasks)
   ```
   - Requires async refactor of screener
   - Potential 5-10x speedup on signal generation

8. **Early Doctrine Veto**
   - Check doctrine gate BEFORE fetching klines
   - If regime is CRISIS/CHOPPY, skip signal generation entirely
   - Current: Signals generated → vetoed later

---

## 7. Usage

### Enable Timing Diagnostics
```bash
export LOOP_TIMING_DEBUG=1
export LOOP_TIMING_THRESHOLD=5  # Optional, default 5s
sudo supervisorctl restart hedge:hedge-executor
```

### View Timing Logs
```bash
tail -f logs/execution/loop_timing.jsonl | jq '.sections | to_entries | sort_by(.value.duration_s) | reverse'
```

### Aggregate Analysis
```python
import json
timings = [json.loads(l) for l in open('logs/execution/loop_timing.jsonl')]
avg_work = sum(t['work_s'] for t in timings) / len(timings)
print(f"Average work time: {avg_work:.1f}s")
```

---

## 8. Files Changed

| File | Change |
|------|--------|
| `execution/loop_timing.py` | **New** — Timing instrumentation module |
| `execution/executor_live.py` | Added timing hooks to main loop and `_loop_once()` |
| `execution/signal_screener.py` | Added `record_api_call()` tracking |

---

## 9. Test Commands

```bash
# Syntax check
python3 -m py_compile execution/loop_timing.py execution/signal_screener.py execution/executor_live.py

# Unit test timing module
PYTHONPATH=. python3 -c "
from execution.loop_timing import is_enabled, start_loop, end_loop, timed_section
import os
os.environ['LOOP_TIMING_DEBUG'] = '1'
from importlib import reload
from execution import loop_timing
reload(loop_timing)
loop_timing.start_loop(0)
with loop_timing.timed_section('test', api_calls=1): pass
print(loop_timing.end_loop())
"

# Full test suite
PYTHONPATH=. pytest -q
```

---

## 10. Conclusion

The loop cadence issue is a **design bottleneck**, not a bug. The system is doing exactly what it's designed to do — fetching fresh market data for all symbols every iteration. However, the cumulative API latency far exceeds the 60s sleep interval.

**Key Insight:** With current architecture, realistic loop cadence is **120-180s** with full symbol coverage. The Sentinel staleness fix (420s threshold) correctly accounts for this reality.

**Priority Recommendations:**
1. ✅ Sentinel threshold increase (Done)
2. ⏳ Enable timing diagnostics to measure actual distribution
3. 🔜 Implement kline caching (most bang for buck)
4. 🔜 Batch price fetches (single API call optimization)
