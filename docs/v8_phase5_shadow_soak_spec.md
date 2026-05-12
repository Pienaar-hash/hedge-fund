# V8 Phase 5: Shadow Soak Specification

## Overview
Phase 5 operationalizes the certified replay strategy (TREND_PULLBACK_V2_REPLAY_CANDIDATE) as a **shadow layer** running parallel to live trading without altering execution. The shadow soak collects real-time order flow and market microstructure against the certified strategy logic to validate reproducibility and slippage assumptions.

## Scope & Objectives

### Primary Goals
1. **Reproducibility Validation**: Confirm strategy logic produces identical signals on live market data vs. replayed data
2. **Slippage Calibration**: Measure actual fills vs. model assumptions (5 bps slippage, 10 bps fees)
3. **Latency Profiling**: Track order round-trip time (submit → ack → fill)
4. **Signal Quality**: Detect edge cases (gaps, halts, liquidity spikes) not present in replay seed

### Non-Goals
- No position-taking or live order placement
- No doctrine kernel mutation
- No risk limit changes
- No dashboard alerts or notifications

## Architecture

```
┌─ Live Executor ──────────────────┐
│  (doctrine_kernel.py)            │
│  - Entry/exit gating (unchanged) │
│  - Order routing                 │
└─────────────────────────────────┘
         ↓ orders
┌─ Shadow Soak Layer (NEW) ────────┐
│  research/shadow_soak_v8.py       │
│  - Mirrors signal logic           │
│  - Records hypothetical trades    │
│  - Compares fills vs. model       │
│  - Emits shadow_soak_events.jsonl │
└─────────────────────────────────┘
         ↓ shadow events
┌─ State Publisher ─────────────────┐
│  logs/state/shadow_soak_state.json │
│  - Running correlation stats      │
│  - Slippage distribution          │
│  - Edge case log                  │
└─────────────────────────────────┘
```

## Implementation Details

### 1. Shadow Soak Engine (`research/shadow_soak_v8.py`)

**Inputs:**
- Live order stream (from executor audit log)
- Live market data feed (15m OHLCV, real-time tick data)
- Certified strategy config (from Phase 4 artifacts)

**Processing Loop:**
```python
def run_shadow_soak():
    """
    1. Ingest latest bars from exchange API (non-blocking)
    2. Compute strategy signal using live OHLCV
    3. Compare signal vs. actual orders (if overlapping)
    4. Record shadow trade (hypothetical entry/exit + model assumptions)
    5. Measure fill vs. model slippage
    6. Append event to shadow_soak_events.jsonl
    """
```

**Output Event Schema:**
```json
{
  "timestamp": "2026-05-15T14:30:00Z",
  "shadow_signal": "LONG_ENTRY",
  "symbol": "BTCUSDT",
  "signal_price": 65125.50,
  "model_slippage": 5,
  "signal_reason": "close > prev_close AND low <= prev_close*0.999",
  "live_order_match": {
    "exists": true,
    "order_id": "19328741928",
    "actual_entry": 65128.25,
    "actual_slippage": 2.75,
    "fill_latency_ms": 142
  },
  "discrepancy": null,
  "correlation": 1.0,
  "phase": "shadow_soak_v8"
}
```

### 2. Correlation Metrics

**Recorded Every 1000 Shadow Events:**
- **Signal Overlap**: % of shadow signals matching actual executor orders
- **Slippage Correlation**: Actual vs. model (target: r > 0.8)
- **Direction Accuracy**: Entry/exit signal directional match (target: >95%)
- **Latency P50/P95/P99**: Order round-trip time distribution

**Failure Triggers:**
- Signal correlation drops below 0.75 → log warning to telemetry
- Slippage mean > model + 10 bps → investigate liquidity degradation
- Unexplained discrepancies > 5 → escalate to human review

### 3. State Publishing (`logs/state/shadow_soak_state.json`)

**Schema:**
```json
{
  "updated_at": "2026-05-15T14:30:00Z",
  "phase": "shadow_soak_v8",
  "events_processed": 47230,
  "signal_correlation": 0.987,
  "slippage_model_r": 0.843,
  "slippage_mean_bps": 5.2,
  "slippage_stddev_bps": 2.1,
  "fill_latency_p50_ms": 128,
  "fill_latency_p95_ms": 312,
  "fill_latency_p99_ms": 780,
  "edge_cases_detected": [
    {
      "timestamp": "2026-05-14T08:15:00Z",
      "symbol": "ETHUSDT",
      "reason": "gap > 2%",
      "action": "skipped signal"
    }
  ],
  "verdict_status": "healthy"
}
```

### 4. Edge Case Handling

**Gaps (>1% overnight/weekend):**
- Skip shadow signal on gap open
- Record gap event with magnitude
- Resume shadow on gap-close

**Halts:**
- Pause shadow soak during trading halt
- Resume post-halt after market stabilization (1 minute)

**Extreme Volatility (>5% intrabar):**
- Log as outlier
- Apply model 10 bps additional slippage buffer
- Continue shadow soak

**Low Liquidity:**
- Track NotionalUSD < min_notional_usd threshold
- Record as "illiquidity skip"
- Resume when liquidity recovers

## Success Criteria

### Gate to Phase 6 (Live Activation)

**All must be true after 14 days shadow soak:**

1. ✅ **Signal Correlation ≥ 0.95**  
   - At least 95% of shadow signals have corresponding live orders (or intentional vetoes)

2. ✅ **Slippage Calibration R ≥ 0.80**  
  - Actual slippage vs. model 5 bps assumption correlates r ≥ 0.80
  - Median absolute slippage error must be <= 3 bps
  - P95 absolute slippage error must be <= 10 bps
  - Wider observed values may be recorded, but they do not qualify the gate

3. ✅ **No Catastrophic Mismatches**  
   - Zero instances of:
     - Direction mismatch (signal LONG, actual SHORT)
     - Quantity deviation > 10%
     - Symbol confusion (signal BTCUSDT, order ETHUSDT)

4. ✅ **Fill Latency P99 < 2s**  
   - 99th percentile round-trip time under 2 seconds

5. ✅ **Edge Case Handling Robust**  
   - Gaps, halts, volatility spikes handled without crashes
   - Shadow soak resumes cleanly post-event

### Abort Criteria

**Any of these trigger shadow soak pause pending investigation:**

- Signal correlation drops below 0.75 (2 consecutive checks)
- Correlation remains below 0.95 at the Phase 5 gate review point
- Slippage r < 0.60 (indicates model assumption invalid)
- Direction mismatch detected
- Unexplained message sequencing (out-of-order acks)
- Doctrine kernel logs refusal of shadow-signaled orders without documented veto

## Dependencies & Constraints

### Runtime Config
- **Runtime.yaml**: shadow_soak_enabled flag (boolean), shadow_soak_batch_interval (seconds)
- **Risk Limits**: Unchanged (no new exposure)
- **Doctrine Kernel**: Read-only (no mutations)
- **Position Cache**: Shadowed, not executed

### Code Changes Required
1. Add `research/shadow_soak_v8.py` module
2. Add shadow soak flag to `execution/v6_flags.py`
3. Update state manifest (`v7_manifest.json`) to include `shadow_soak_state.json`
4. Add test `tests/unit/test_shadow_soak_v8.py`

### Compliance
- ✅ No doctrine mutation
- ✅ No conviction authority change
- ✅ No live orders placed by shadow layer
- ✅ Append-only logging
- ✅ Read-only market data consumption

## Transition Path

### Pre-Launch (Phase 5 Days 1–3)
1. Deploy `research/shadow_soak_v8.py` in observation mode
2. Verify live data ingestion without errors
3. Validate signal computation on real bars

### Soak Period (Phase 5 Days 4–14)
1. Run shadow soak in background (non-blocking)
2. Collect metrics every 1000 events
3. Publish shadow_soak_state.json to dashboard
5. Investigate any correlation < 0.95

### Gate Decision (End of Day 14)
- **PASS**: All 5 success criteria met → recommend Phase 6 (live activation proposal)
- **CONDITIONAL**: 4/5 criteria met → recommend 7-day extension soak
- **FAIL**: <4/5 criteria met → recommend strategy redesign or model recalibration

## Monitoring & Alerts

### Dashboard Panel
- Real-time shadow_soak_state.json display
- Rolling correlation over 24h / 7d / 14d windows
- Slippage scatter plot: actual vs. model

### Telemetry
- Shadow soak events published to `logs/execution/shadow_soak_events.jsonl` (append-only)
- No Telegram alerts (observation-only layer)

### Developer Debugging
```bash
# View latest 100 shadow events
tail -100 logs/execution/shadow_soak_events.jsonl | jq '{symbol, shadow_signal, correlation}'

# Check correlation drift
python -c "
  import json
  events = [json.loads(l) for l in open('logs/execution/shadow_soak_events.jsonl')]
  recent = events[-1000:]
  corr = sum(1 for e in recent if e.get('correlation') > 0.95) / len(recent)
  print(f'Correlation (last 1000): {corr:.2%}')
"
```

## Acceptance Test

```bash
# Pre-launch validation
PYTHONPATH=. pytest -q tests/unit/test_shadow_soak_v8.py

# Smoke test: 1-hour soak on testnet
export BINANCE_TESTNET=1 DRY_RUN=1 SHADOW_SOAK_ENABLED=1
python -c "
  from research.shadow_soak_v8 import run_shadow_soak_sync
  result = run_shadow_soak_sync(duration_sec=3600)
  assert result['events_processed'] > 100
  assert result['signal_correlation'] > 0.80
  print('✓ Shadow soak smoke test PASSED')
"
```

## Rollback Plan

**If shadow soak encounters fatal errors during Phase 5:**
1. Set `shadow_soak_enabled = false` in runtime.yaml
2. Stop `research/shadow_soak_v8.py` process
3. Archive `logs/research/shadow_soak_events.jsonl` to `logs/research/archive/`
4. Write `logs/state/shadow_soak_state.json` with `status=PAUSED` and `reason=operator_disabled`
5. Executor continues unchanged
6. Post-mortem: analyze logs, update strategy, redeploy in soak mode

**No executor changes required** (shadow layer is independent).

## References

- **Certified Replay Artifacts**: `data/replay_certifications/v8_phase4_fps_v2_cert_003/`
- **Strategy Logic**: `research/backtest_engine_v8.py::_strategy_signal()`
- **Doctrine Authority**: `execution/doctrine_kernel.py` (unchanged)
- **Risk Limits**: `config/risk_limits.json` (unchanged)

---
**Status**: Specification (Pre-Implementation)  
**Last Updated**: 2026-05-12  
**Phase**: V8 Phase 5 (Shadow Soak)  
**Next Gate**: Approval for Phase 5 implementation + deployment
