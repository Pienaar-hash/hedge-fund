# Shadow Soak State Schema Documentation

## Overview

The shadow soak observer produces two main output streams:

1. **Event Stream** (`logs/research/shadow_soak_events.jsonl`) - Append-only log of individual observations
2. **State Snapshot** (`logs/state/shadow_soak_state.json`) - Current aggregated metrics and verdict

## Event Stream Schema

**File**: `logs/research/shadow_soak_events.jsonl`

**Format**: JSON Lines (one event per line, append-only)

### Event Object Structure

```json
{
  "ts": "2026-05-12T12:57:45.090853Z",
  "event_type": "live_order_match",
  "run_id": "v8_phase4_fps_v2_cert_003",
  "symbol": "BTCUSDT",
  "live_side": "BUY",
  "shadow_side": "BUY",
  "live_qty": 1.0,
  "shadow_qty": 1.0,
  "live_price": 65000.0,
  "shadow_price": 65000.0,
  "live_order_ts": "2026-05-12T12:00:00Z",
  "shadow_signal_ts": "2026-05-12T12:00:00Z",
  "symbol_match": true,
  "direction_match": true,
  "quantity_bucket_match": true,
  "timestamp_delta_s": 0.5,
  "slippage_bps_actual": 2.0,
  "slippage_bps_model": 5.0,
  "slippage_error_bps": -3.0,
  "catastrophic_mismatch": false,
  "reason": "symbol_match=true, direction_match=true, qty_match=true"
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `ts` | ISO8601 string | When event was recorded (UTC) |
| `event_type` | string | Type: `live_order_match`, `symbol_mismatch`, `direction_mismatch`, `quantity_mismatch`, `timestamp_drift`, `slippage_outlier`, `edge_case_skip`, `metrics_checkpoint` |
| `run_id` | string | Certification run ID being observed (e.g., `v8_phase4_fps_v2_cert_003`) |
| `symbol` | string | Trading symbol (e.g., `BTCUSDT`) |
| `live_side` | string ∣ null | Live order side: `BUY`, `SELL`, or `null` |
| `shadow_side` | string ∣ null | Shadow signal side: `BUY`, `SELL`, or `null` |
| `live_qty` | float ∣ null | Filled quantity from live order |
| `shadow_qty` | float ∣ null | Expected quantity from shadow signal |
| `live_price` | float ∣ null | Actual fill price |
| `shadow_price` | float ∣ null | Model-expected price |
| `live_order_ts` | ISO8601 string ∣ null | When live order was submitted |
| `shadow_signal_ts` | ISO8601 string ∣ null | When shadow signal was generated |
| `symbol_match` | boolean | Whether live symbol matches shadow symbol |
| `direction_match` | boolean ∣ null | Whether side matches (null if symbol didn't match) |
| `quantity_bucket_match` | boolean | Whether qty within 5% tolerance |
| `timestamp_delta_s` | float ∣ null | Seconds between signal and live order (absolute value) |
| `slippage_bps_actual` | float ∣ null | Actual slippage in basis points (BUY: (fill - model) / model * 10000) |
| `slippage_bps_model` | float | Model assumption in basis points (always 5.0) |
| `slippage_error_bps` | float ∣ null | Deviation from model (actual - model) |
| `catastrophic_mismatch` | boolean | True if direction or symbol mismatch (triggers PAUSED status) |
| `reason` | string | Human-readable explanation of observation |

### Event Type Reference

- **`live_order_match`**: Successful match between live order and shadow signal (symbol match verified)
- **`symbol_mismatch`**: Live symbol ≠ shadow symbol (catastrophic, triggers PAUSED)
- **`direction_mismatch`**: Live side ≠ shadow side after symbol match (catastrophic, triggers PAUSED)
- **`quantity_mismatch`**: Quantity deviation > 5% tolerance
- **`timestamp_drift`**: Signal and order separated by > 60s (one executor cycle)
- **`slippage_outlier`**: Actual slippage > model + 10bps
- **`edge_case_skip`**: Live order with no matching shadow signal (e.g., gap, halt)
- **`metrics_checkpoint`**: Periodic aggregation of metrics (recorded every 1000 events)

### Example Events

**Perfect Match**:
```json
{
  "ts": "2026-05-12T13:00:00Z",
  "event_type": "live_order_match",
  "run_id": "v8_phase4_fps_v2_cert_003",
  "symbol": "BTCUSDT",
  "live_side": "BUY",
  "shadow_side": "BUY",
  "live_qty": 1.0,
  "shadow_qty": 1.0,
  "live_price": 65100.0,
  "shadow_price": 65000.0,
  "live_order_ts": "2026-05-12T12:59:59Z",
  "shadow_signal_ts": "2026-05-12T12:59:59.500Z",
  "symbol_match": true,
  "direction_match": true,
  "quantity_bucket_match": true,
  "timestamp_delta_s": 0.5,
  "slippage_bps_actual": 15.4,
  "slippage_bps_model": 5.0,
  "slippage_error_bps": 10.4,
  "catastrophic_mismatch": false,
  "reason": "within tolerance"
}
```

**Direction Mismatch**:
```json
{
  "ts": "2026-05-12T13:00:30Z",
  "event_type": "direction_mismatch",
  "run_id": "v8_phase4_fps_v2_cert_003",
  "symbol": "ETHUSDT",
  "live_side": "BUY",
  "shadow_side": "SELL",
  "live_qty": 10.0,
  "shadow_qty": 10.0,
  "live_price": 3500.0,
  "shadow_price": 3500.0,
  "live_order_ts": "2026-05-12T13:00:00Z",
  "shadow_signal_ts": "2026-05-12T13:00:00.100Z",
  "symbol_match": true,
  "direction_match": false,
  "quantity_bucket_match": true,
  "timestamp_delta_s": 0.1,
  "slippage_bps_actual": 0.0,
  "slippage_bps_model": 5.0,
  "slippage_error_bps": -5.0,
  "catastrophic_mismatch": true,
  "reason": "direction mismatch: BUY vs SELL"
}
```

---

## State Snapshot Schema

**File**: `logs/state/shadow_soak_state.json`

**Format**: Single JSON object (snapshot, overwritten on each update)

### State Object Structure

```json
{
  "run_id": "v8_phase4_fps_v2_cert_003",
  "started_at": "2026-05-12T12:57:45.090822Z",
  "updated_at": "2026-05-12T13:15:30.123456Z",
  "status": "running",
  "sample_size": 456,
  "symbol_match_rate": 0.987,
  "direction_match_rate": 0.995,
  "quantity_bucket_match_rate": 0.992,
  "timestamp_alignment_p95_s": 2.3,
  "slippage_model_error_r": 0.842,
  "median_abs_slippage_error_bps": 1.8,
  "p95_abs_slippage_error_bps": 8.5,
  "fill_latency_p99_s": 1.2,
  "catastrophic_mismatch_count": 0,
  "consecutive_failed_checks": 0,
  "verdict": "pass",
  "failed_criteria": []
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Certification run ID being observed |
| `started_at` | ISO8601 string | When shadow soak started |
| `updated_at` | ISO8601 string | When state was last updated |
| `status` | string | Status: `running`, `paused`, `complete`, `failed` |
| `sample_size` | integer | Total events processed (with symbol match) |
| `symbol_match_rate` | float 0-1 | % of events with symbol match |
| `direction_match_rate` | float 0-1 ∣ null | % of symbol-matched events with direction match (null if n < 1) |
| `quantity_bucket_match_rate` | float 0-1 ∣ null | % of events with qty within 5% bucket (null if n < 1) |
| `timestamp_alignment_p95_s` | float ∣ null | 95th percentile of \|live_ts - shadow_ts\| (null if n < 20) |
| `slippage_model_error_r` | float 0-1 ∣ null | Correlation measure for actual vs model slippage (null if n < 10) |
| `median_abs_slippage_error_bps` | float ∣ null | Median of \|actual - model\| in basis points (null if n < 10) |
| `p95_abs_slippage_error_bps` | float ∣ null | 95th percentile of \|actual - model\| (null if n < 20) |
| `fill_latency_p99_s` | float ∣ null | 99th percentile of order round-trip time in seconds (null if n < 100) |
| `catastrophic_mismatch_count` | integer | Total direction + symbol mismatches |
| `consecutive_failed_checks` | integer | How many consecutive checks failed (increments on FAIL verdict) |
| `verdict` | string | `pending`, `pass`, `conditional`, `fail` |
| `failed_criteria` | array[string] | List of criteria that failed (empty if `pass`) |

### Status Reference

- **`running`**: Shadow soak actively collecting events
- **`paused`**: Paused due to catastrophic mismatch or correlation drop; investigation required
- **`complete`**: 14-day soak period finished; final verdict issued
- **`failed`**: Unrecoverable error (missing/corrupt logs, exception)

### Verdict Reference

**`pass`**: All 10 criteria met (gate to Phase 6 approved)

Criteria:
1. `sample_size >= 100`
2. `symbol_match_rate >= 0.95`
3. `direction_match_rate >= 0.95` (or null)
4. `quantity_bucket_match_rate >= 0.95` (or null)
5. `timestamp_alignment_p95_s <= 60.0` (or null)
6. `slippage_model_error_r >= 0.80` (or null)
7. `median_abs_slippage_error_bps <= 3.0` (or null)
8. `p95_abs_slippage_error_bps <= 10.0` (or null)
9. `fill_latency_p99_s < 2.0` (or null)
10. `catastrophic_mismatch_count == 0`

**`conditional`**: 4/5 criteria met (recommend 7-day extension soak)

**`fail`**: <4/5 criteria met, or catastrophic mismatch detected (recommend strategy redesign)

**`pending`**: Insufficient data or no gate decision made yet

### Example State Snapshots

**RUNNING - Early Stage**:
```json
{
  "run_id": "v8_phase4_fps_v2_cert_003",
  "started_at": "2026-05-12T12:57:45Z",
  "updated_at": "2026-05-12T13:15:30Z",
  "status": "running",
  "sample_size": 45,
  "symbol_match_rate": 0.982,
  "direction_match_rate": 0.989,
  "quantity_bucket_match_rate": 0.987,
  "timestamp_alignment_p95_s": null,
  "slippage_model_error_r": null,
  "median_abs_slippage_error_bps": null,
  "p95_abs_slippage_error_bps": null,
  "fill_latency_p99_s": null,
  "catastrophic_mismatch_count": 0,
  "consecutive_failed_checks": 0,
  "verdict": "pending",
  "failed_criteria": ["sample_size >= 100"]
}
```

**PAUSED - Catastrophic Mismatch**:
```json
{
  "run_id": "v8_phase4_fps_v2_cert_003",
  "started_at": "2026-05-12T12:57:45Z",
  "updated_at": "2026-05-12T15:22:10Z",
  "status": "paused",
  "sample_size": 280,
  "symbol_match_rate": 0.975,
  "direction_match_rate": 0.980,
  "quantity_bucket_match_rate": 0.985,
  "timestamp_alignment_p95_s": 1.8,
  "slippage_model_error_r": 0.825,
  "median_abs_slippage_error_bps": 2.1,
  "p95_abs_slippage_error_bps": 7.8,
  "fill_latency_p99_s": 1.1,
  "catastrophic_mismatch_count": 1,
  "consecutive_failed_checks": 0,
  "verdict": "fail",
  "failed_criteria": ["catastrophic_mismatch_count == 0"]
}
```

**RUNNING - 14-Day PASS**:
```json
{
  "run_id": "v8_phase4_fps_v2_cert_003",
  "started_at": "2026-05-12T12:57:45Z",
  "updated_at": "2026-05-26T12:57:45Z",
  "status": "complete",
  "sample_size": 47230,
  "symbol_match_rate": 0.9876,
  "direction_match_rate": 0.9952,
  "quantity_bucket_match_rate": 0.9918,
  "timestamp_alignment_p95_s": 2.1,
  "slippage_model_error_r": 0.843,
  "median_abs_slippage_error_bps": 1.5,
  "p95_abs_slippage_error_bps": 8.2,
  "fill_latency_p99_s": 1.8,
  "catastrophic_mismatch_count": 0,
  "consecutive_failed_checks": 0,
  "verdict": "pass",
  "failed_criteria": []
}
```

---

## Integration with Dashboard

The dashboard reads `logs/state/shadow_soak_state.json` on each refresh to display:

- **Verdict Status Panel**: Current verdict and 5/10 criteria passing
- **Correlation Metrics**: symbol_match_rate, direction_match_rate over time
- **Slippage Analysis**: Scatter plot of actual vs model (r coefficient)
- **Latency Distribution**: Fill latency P99 over 14-day window
- **Mismatch Log**: Recent catastrophic_mismatch events with reasons

---

## Workflow Transitions

```
[Research soak starts]
      ↓
  RUNNING, verdict=pending, sample_size < 100
      ↓
  [Events accumulate]
      ↓
  RUNNING, verdict=conditional/fail, sample_size >= 100
      ↓
  [Decision at 14-day mark]
      ↓
  ┌─────────────────────────────────────────────┐
  │ All 10 criteria pass?                       │
  ├─────────────────────────────────────────────┤
  │ YES → COMPLETE, verdict=pass                │
  │       (recommend Phase 6 approval)          │
  │                                             │
  │ 4-5 criteria → COMPLETE, verdict=conditional│
  │       (recommend 7-day extension)           │
  │                                             │
  │ Catastrophic mismatch → PAUSED, verdict=fail│
  │       (recommend redesign)                  │
  │                                             │
  │ <4 criteria → COMPLETE, verdict=fail        │
  │       (recommend redesign)                  │
  └─────────────────────────────────────────────┘
```

---

## Monitoring Commands

```bash
# View latest state
cat logs/state/shadow_soak_state.json | jq '.'

# Watch correlation trend
tail -100 logs/research/shadow_soak_events.jsonl | jq '{symbol_match: .symbol_match, direction_match: .direction_match}'

# Count catastrophic mismatches
grep 'catastrophic_mismatch": true' logs/research/shadow_soak_events.jsonl | wc -l

# Check verdict changes
grep '"verdict"' logs/state/shadow_soak_state.json

# Slippage distribution
grep 'live_order_match' logs/research/shadow_soak_events.jsonl | jq '.slippage_bps_actual' | sort -n | uniq -c
```

---

**Last Updated**: 2026-05-12  
**Phase**: V8 Phase 5 (Shadow Soak)  
**Status**: Ready for 14-day observation period
