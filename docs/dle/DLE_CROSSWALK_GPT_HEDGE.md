# DLE Crosswalk: GPT-Hedge Current State → DLE Target State

> **Status:** MAPPING DOCUMENT  
> **Date:** 2026-01-28  
> **Purpose:** Define migration path from current logs to DLE ledgers

---

## Overview

This document maps current GPT-Hedge log structures to DLE target schemas. It answers:
- What exists today?
- What becomes what?
- What's new?
- What fields need backfilling?

---

## 1. doctrine_events.jsonl → Decision Ledger + Telemetry

### Current Structure

**File:** `logs/doctrine_events.jsonl`

```json
{
  "ts": "2025-12-13T10:59:46.050360+00:00",
  "type": "ENTRY_VETO",
  "symbol": "BTCUSDT",
  "verdict": "VETO_NO_REGIME",
  "intent": {
    "timestamp": "2025-12-13T10:59:40+00:00",
    "symbol": "BTCUSDT",
    "signal": "SELL",
    "price": 90540.2,
    "leverage": 4,
    "qty": 0.001,
    "nav_used": 10194.21,
    "per_trade_nav_pct": 0.015,
    "tier": "CORE",
    "trend": "NEUTRAL",
    "attempt_id": "sig_402132d10b",
    ...
  }
}
```

### Current Event Types

| Type | Meaning | DLE Category |
|------|---------|--------------|
| `ENTRY_VETO` | Entry blocked | → Telemetry (denial observation) |
| `ENTRY_PERMIT` | Entry allowed | → Decision + Telemetry |
| `EXIT_VETO` | Exit blocked | → Telemetry |
| `EXIT_PERMIT` | Exit allowed | → Decision + Telemetry |
| `REGIME_CHECK` | Regime evaluation | → Telemetry only |

### Migration Mapping

#### → `logs/dle/decision_ledger.jsonl` (NEW)

Extract authority statements only:

```json
{
  "decision_id": "auto_d_{hash}",
  "created_ts": "{ts}",
  "intent": "Doctrine permits {signal} on {symbol}",
  "scope": {
    "symbols": ["{symbol}"],
    "directions": ["{signal == BUY ? LONG : SHORT}"],
    "max_notional_usd": "{intent.gross_usd * 2}"
  },
  "permitted_actions": ["OPEN_LONG" | "OPEN_SHORT"],
  "forbidden_actions": [],
  "constraints": {
    "required_regime": ["{current_regime}"],
    "min_confidence": 0.5
  },
  "expiration": {
    "mode": "SINGLE_USE",
    "max_uses": 1
  },
  "phase_id": "{current_phase}",
  "risk_acknowledgement": {
    "max_loss_pct": "{intent.per_trade_nav_pct}",
    "fail_closed": true
  },
  "authority_source": "DOCTRINE"
}
```

**Key insight:** Current `ENTRY_PERMIT` events become Decisions. The `intent` object contains all the constraint data needed.

#### → `logs/doctrine_events.jsonl` (REMAINS — pure telemetry)

Keep all events but reclassify:

```json
{
  "ts": "...",
  "type": "VETO_OBSERVATION",
  "event_type_legacy": "ENTRY_VETO",
  "symbol": "BTCUSDT",
  "verdict": "VETO_NO_REGIME",
  "decision_id": null,
  "telemetry": {
    "regime": "CHOPPY",
    "confidence": 0.45,
    "nav_usd": 10194.21,
    "intent_summary": {...}
  }
}
```

---

## 2. orders_executed.jsonl → Order + Permit Binding

### Current Structure

**File:** `logs/execution/orders_executed.jsonl`

Three event types observed:

#### Order Submission
```json
{
  "event_type": "order_submit",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "order_type": "LIMIT",
  "qty": 0.001,
  "price": 90540.2,
  "position_side": "LONG",
  "reduce_only": false,
  "ts": "...",
  "attempt_id": "sig_402132d10b"
}
```

#### Order Acknowledged
```json
{
  "event_type": "order_ack",
  "symbol": "BTCUSDT",
  "orderId": "123456789",
  "clientOrderId": "...",
  "status": "NEW",
  "intent_id": "...",
  "attempt_id": "...",
  "ts_ack": "..."
}
```

#### Order Filled
```json
{
  "event_type": "order_filled",
  "symbol": "BTCUSDT",
  "orderId": "123456789",
  "status": "FILLED",
  "avgPrice": 90542.5,
  "executedQty": "0.001",
  "fee_total": 0.0543,
  "feeAsset": "USDT",
  "ts_fill_first": "...",
  "ts_fill_last": "...",
  "intent_id": "...",
  "attempt_id": "...",
  "metadata": {...}
}
```

### Migration Mapping

**No file change needed** — orders_executed.jsonl remains as execution evidence.

**Add fields in DLE mode:**

```json
{
  "event_type": "order_submit",
  ...
  "permit_id": "p_{uuid}",           // NEW: Links to permit
  "decision_id": "d_{uuid}"          // NEW: Links to decision
}
```

**Validation rule:** In DLE mode, order_submit MUST have permit_id. Executor rejects orders without valid permit.

---

## 3. episode_ledger.json → DLE Episode Ledger

### Current Structure

**File:** `logs/state/episode_ledger.json`

```json
{
  "last_rebuild_ts": "2026-01-28T10:00:00+00:00",
  "episode_count": 21,
  "episodes": [
    {
      "episode_id": "EP_0021",
      "symbol": "ETHUSDT",
      "side": "LONG",
      "entry_ts": "2026-01-20T22:04:49+00:00",
      "exit_ts": "2026-01-21T00:36:27+00:00",
      "duration_hours": 2.53,
      "entry_fills": 4,
      "exit_fills": 1,
      "entry_notional": 784.24,
      "exit_notional": 8.83,
      "total_qty": 0.262,
      "avg_entry_price": 2993.291,
      "avg_exit_price": 2942.9,
      "gross_pnl": -0.1512,
      "fees": 0.213,
      "net_pnl": -0.3642,
      "regime_at_entry": "unknown",
      "regime_at_exit": "unknown",
      "exit_reason": "regime_flip",
      "strategy": "vol_target"
    }
  ],
  "stats": {...}
}
```

### Migration Mapping

**Evolve to DLE Episode format:**

```json
{
  "episode_id": "EP_0021",                    // KEEP (backward compat)
  "symbol": "ETHUSDT",                        // KEEP
  "direction": "LONG",                        // RENAME from "side"
  "state": "CLOSED",                          // NEW (enum)
  "created_ts": "2026-01-20T22:04:49+00:00",  // RENAME from entry_ts
  "closed_ts": "2026-01-21T00:36:27+00:00",   // RENAME from exit_ts
  "phase_id": "CYCLE_002",                    // NEW (backfill: infer from date)
  
  "authority": {                              // NEW BLOCK
    "decision_id": "PRE_DLE",                 // Backfill marker
    "permit_id": null,                        // Not available pre-DLE
    "requester": {
      "type": "HYDRA_HEAD",
      "id": "unknown"                         // Backfill from strategy
    },
    "rationale": "Pre-DLE episode"
  },
  
  "entry": {                                  // RESTRUCTURE
    "filled_price_avg": 2993.291,             // from avg_entry_price
    "filled_notional_usd": 784.24,            // from entry_notional
    "filled_size": 0.262,                     // from total_qty
    "order_ids": [],                          // Backfill from orders_executed
    "entry_ts": "2026-01-20T22:04:49+00:00"
  },
  
  "position": {                               // NEW BLOCK
    "size": 0.262,
    "entry_price": 2993.291
  },
  
  "exit": {                                   // RESTRUCTURE
    "exit_reason": "REGIME_CHANGE",           // NORMALIZE from regime_flip
    "exit_price_avg": 2942.9,                 // from avg_exit_price
    "exit_order_ids": [],                     // Backfill from orders_executed
    "exit_ts": "2026-01-21T00:36:27+00:00"
  },
  
  "outcome": {                                // RESTRUCTURE
    "gross_pnl_usd": -0.1512,                 // from gross_pnl
    "fees_usd": 0.213,                        // from fees
    "net_pnl_usd": -0.3642,                   // from net_pnl
    "hold_duration_s": 9098,                  // from duration_hours * 3600
    "win": false                              // NEW: net_pnl > 0
  },
  
  "snapshots": {                              // NEW BLOCK
    "regime_at_entry": "unknown",             // KEEP
    "regime_at_exit": "unknown"               // KEEP
  }
}
```

### Exit Reason Normalization

| Current | DLE |
|---------|-----|
| `regime_flip` | `REGIME_CHANGE` |
| `tp_hit` | `TAKE_PROFIT` |
| `sl_hit` | `STOP_LOSS` |
| `trailing_stop` | `TRAILING_STOP` |
| `manual` | `MANUAL_CLOSE` |
| `risk_veto` | `RISK_VETO` |
| `thesis_invalid` | `THESIS_INVALIDATED` |

---

## 4. positions_state.json → Snapshot Anchoring

### Current Structure

**File:** `logs/state/positions_state.json`

```json
{
  "positions": [],
  "updated_at": "2026-01-28T14:00:00",
  "updated_ts": 1738076400.123
}
```

### DLE Usage

positions_state.json remains unchanged but provides **snapshot anchoring** for:

1. **Permit scope_snapshot:**
   ```json
   "scope_snapshot": {
     "portfolio_hash": "sha256({positions_state.json})",
     ...
   }
   ```

2. **Episode snapshots:**
   ```json
   "snapshots": {
     "entry_snapshot_hash": "sha256({positions_state at entry})",
     "exit_snapshot_hash": "sha256({positions_state at exit})"
   }
   ```

### Hash Computation

```python
import hashlib
import json

def compute_positions_hash(positions_state: dict) -> str:
    """Compute deterministic hash of position state."""
    # Sort keys for determinism
    canonical = json.dumps(
        positions_state["positions"],
        sort_keys=True,
        separators=(",", ":")
    )
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
```

---

## 5. New DLE Ledger Files

These files do not exist today and must be created:

| File | Purpose | Initial State |
|------|---------|---------------|
| `logs/dle/decision_ledger.jsonl` | Authority statements | Empty until Phase 2 |
| `logs/dle/permits_issued.jsonl` | Execution tokens | Empty until Phase 3 |
| `logs/dle/denials.jsonl` | Refused requests | Empty until Phase 2 |
| `logs/dle/execution_requests.jsonl` | Gate input log | Empty until Phase 2 |

### Directory Structure

```
logs/
├── dle/                           # NEW
│   ├── decision_ledger.jsonl      # Authority
│   ├── permits_issued.jsonl       # Tokens
│   ├── denials.jsonl              # Refusals
│   └── execution_requests.jsonl   # Requests
├── execution/
│   └── orders_executed.jsonl      # EXISTING (add permit_id field)
├── state/
│   ├── episode_ledger.json        # EXISTING (evolve schema)
│   └── positions_state.json       # EXISTING (unchanged)
└── doctrine_events.jsonl          # EXISTING (becomes telemetry only)
```

---

## 6. Backfill Strategy

### Phase 1: Historical Episodes

For episodes created before DLE:

```json
{
  "authority": {
    "decision_id": "PRE_DLE",
    "permit_id": null,
    "rationale": "Pre-DLE episode (backfilled)"
  }
}
```

### Phase 2: Order→Episode Binding

Scan `orders_executed.jsonl` and match to episodes by:
- Symbol
- Timestamp range
- Side

Populate:
- `entry.order_ids`
- `exit.exit_order_ids`

### Phase 3: Regime Backfill

For episodes with `regime_at_entry: "unknown"`:
- Query Sentinel-X state at `entry_ts`
- If unavailable, keep as "unknown"

---

## 7. Validation Queries

After migration, these queries should pass:

### Every closed episode has authority
```sql
SELECT * FROM episodes 
WHERE state = 'CLOSED' 
AND authority.decision_id IS NULL
-- Should return 0 rows (excluding PRE_DLE)
```

### Every order has permit (DLE mode)
```sql
SELECT * FROM orders 
WHERE ts > DLE_ACTIVATION_DATE 
AND permit_id IS NULL
-- Should return 0 rows
```

### Every denial references a request
```sql
SELECT * FROM denials 
WHERE request_id IS NULL
-- Should return 0 rows
```

### Decision→Permit→Episode chain
```sql
SELECT e.episode_id, e.authority.decision_id, e.authority.permit_id
FROM episodes e
WHERE e.state = 'CLOSED'
AND e.authority.decision_id != 'PRE_DLE'
AND (
  NOT EXISTS (SELECT 1 FROM decisions d WHERE d.decision_id = e.authority.decision_id)
  OR NOT EXISTS (SELECT 1 FROM permits p WHERE p.permit_id = e.authority.permit_id)
)
-- Should return 0 rows
```

---

## 8. Implementation Order

| Step | Action | Breaking? | Files Affected |
|------|--------|-----------|----------------|
| 1 | Create `logs/dle/` directory | No | New directory |
| 2 | Add DLE fields to episode writer | No | episode_ledger.json |
| 3 | Shadow-log decisions from doctrine | No | decision_ledger.jsonl |
| 4 | Shadow-log denials | No | denials.jsonl |
| 5 | Add permit_id to order metadata | No | orders_executed.jsonl |
| 6 | Require permit for orders | **YES** | executor, order_router |

---

## Approval

This crosswalk defines the migration path from current GPT-Hedge logs to DLE-compliant ledgers.

Implementation begins in CYCLE_004 Phase A (shadow mode).
