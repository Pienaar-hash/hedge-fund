# Fee Gate Edge Diagnostics — Workstream 4

**Date:** 2026-02-24  
**Status:** Active  
**Severity:** P0 — 100% entry veto rate due to key mismatch

---

## Summary

The fee gate (`execution/fee_gate.py`, v7.9-E2) universally vetoes all entries
because the `expected_edge_pct` it reads is **always 0.0**.  This is a
**key mismatch** — the fee gate reads from a location that no upstream component
ever writes to.

```
edge $0.0000 < required $0.0577   ← every intent, every cycle
```

## Root Cause: Key Path Mismatch

### Where edge IS written

| Source | Location in intent | Value |
|--------|--------------------|-------|
| `signal_generator.py` L342 | `intent["expected_edge"]` | `confidence - 0.5` (top-level) |
| `signal_screener.py` L1306 | `intent["hybrid_components"]["expectancy"]` | hybrid expectancy factor (0.0–1.0) |
| `signal_screener.py` L1413 | `intent["metadata"]["hybrid_score"]` | blended hybrid score |
| `signal_screener.py` L1418 | `intent["metadata"]["conviction_score"]` | conviction engine score |

### Where fee gate READS

```python
# executor_live.py L3896–3900
_fg_meta = intent.get("metadata") or {}
expected_edge_pct = float(
    _fg_meta.get("expectancy", 0)         # ← NEVER SET
    or _fg_meta.get("expected_edge_pct", 0) # ← NEVER SET
    or 0
)
```

**Neither `metadata.expectancy` nor `metadata.expected_edge_pct` is populated
by any code path.** The related data exists in:

- `intent["expected_edge"]` — top-level, from `signal_generator.py`
- `intent["hybrid_components"]["expectancy"]` — nested under `hybrid_components`, from screener
- `intent["hybrid_score"]` — blended score from screener

## Data Flow Diagram

```
┌─────────────────────────────┐
│  signal_generator.py        │
│  L342: intent["expected_edge"] = confidence - 0.5  ← 0.0 to 1.0
└─────────────┬───────────────┘
              ↓
┌─────────────────────────────────────────┐
│  signal_screener.py                     │
│  L1304: intent["hybrid_score"] = 0.xxx  │
│  L1306: intent["hybrid_components"]     │
│         = {"trend": 0.x, "carry": 0.x,  │
│            "expectancy": 0.x,           │
│            "router": 0.x}              │
│  L1413: intent["metadata"]["hybrid_score"]    │
│  L1418: intent["metadata"]["conviction_score"]│
│  (NO write to metadata.expectancy)     │
└─────────────┬───────────────────────────┘
              ↓
┌──────────────────────────────────────────────┐
│  executor_live.py (fee gate at L3890)        │
│  READS: intent["metadata"]["expectancy"]     │  ← 0 (key doesn't exist)
│  READS: intent["metadata"]["expected_edge_pct"]│ ← 0 (key doesn't exist)
│  expected_edge_pct = 0.0                     │
│  expected_edge_usd = notional × 0.0 = $0.00 │
│  required_edge = notional × 0.0012 = $0.05+ │
│  RESULT: $0.00 < $0.05+ → UNIVERSAL VETO     │
└──────────────────────────────────────────────┘
```

## Fix: Wire Edge Into Fee Gate

### Primary fix (executor_live.py L3896–3900)

Read edge from the correct locations, with a **priority chain**:

1. `intent["expected_edge"]` — signal_generator's `confidence - 0.5` (primary, top-level)
2. `intent["hybrid_components"]["expectancy"]` — hybrid scoring's expectancy factor (fallback)
3. `intent["metadata"]["expectancy"]` — original path (preserve backward compat)
4. `intent["metadata"]["expected_edge_pct"]` — original path (preserve backward compat)

### Secondary fix: emit FEE_GATE_VETO_DETAIL structured event

Every fee gate veto should emit a structured event to
`logs/execution/fee_gate_events.jsonl` with:

```json
{
  "event": "FEE_GATE_VETO_DETAIL",
  "ts": 1740000000.0,
  "symbol": "BTCUSDT",
  "side": "BUY",
  "notional_usd": 48.08,
  "expected_edge_pct": 0.065,
  "expected_edge_usd": 3.12,
  "required_edge_usd": 0.0577,
  "round_trip_fee_usd": 0.0385,
  "fee_buffer_mult": 1.5,
  "taker_fee_rate": 0.0004,
  "gate_status": "veto",
  "shortfall_usd": -3.06,
  "edge_source": "expected_edge",
  "edge_components": {
    "expected_edge": 0.065,
    "hybrid_expectancy": 0.42,
    "hybrid_score": 0.55,
    "conviction_score": 0.56
  }
}
```

This turns the fee gate from opaque ("$0 < $0.06 → veto") into a
diagnostic truth layer showing exactly where edge comes from.

## Fee Math Reference

```
taker_fee_rate  = 0.0004  (0.04%)          # from runtime.yaml
round_trip_fee  = notional × 0.0004 × 2    # entry + exit, both legs
fee_buffer_mult = 1.5                       # safety margin
required_edge   = round_trip_fee × 1.5     # = notional × 0.0012

Minimum edge percentage to pass: 0.12% (regardless of notional)
```

| Notional | RT Fee | Required Edge | Min edge % |
|----------|--------|---------------|------------|
| $22      | $0.0176| $0.0264       | 0.12%      |
| $48      | $0.0384| $0.0577       | 0.12%      |
| $100     | $0.08  | $0.12         | 0.12%      |
| $500     | $0.40  | $0.60         | 0.12%      |

## Edge Sources Available

| Source | Field | Range | Quality |
|--------|-------|-------|---------|
| signal_generator confidence | `intent["expected_edge"]` = confidence - 0.5 | -0.5 to 1.0 | Low — raw TA confidence, not true expected move |
| hybrid expectancy | `intent["hybrid_components"]["expectancy"]` | 0.0–1.0 | Medium — rolling win rate / PnL ratio |
| hybrid_score | `intent["hybrid_score"]` | -1.0–1.0 | Medium — blended multi-factor score |
| conviction_score | `intent["conviction_score"]` | 0.0–1.0 | Medium — conviction-weighted composite |

**Note:** None of these are a true expected-move estimate in price terms.
They are all scoring proxies. The `expected_edge` from signal_generator
(= `confidence - 0.5`) is the best proxy available: a signal with
confidence 0.65 produces `expected_edge = 0.15` (15%), which is the
signal's claimed edge over a coin-flip baseline.

## Recommendation

1. **Wire `intent["expected_edge"]` into the fee gate** as the primary source
2. **Emit `FEE_GATE_VETO_DETAIL` structured events** with full decomposition
3. **Add tests** for "edge present → gate evaluates" vs "edge missing → gate vetoes"
4. **Future:** Build a proper expected-move estimator (ATR-based or vol-scaled)
   that computes edge in price-move terms rather than confidence-score terms
