# DLE Episode Schema v1

> **Status:** SPECIFICATION — not yet implemented  
> **Author:** GPT-Hedge governance audit  
> **Date:** 2026-01-28

## Purpose

An **Episode** is the complete lifecycle of a trading decision: from permission to execution to outcome. It is the **canonical audit artifact** for investors.

Episodes bind together: Decision → Permit → Orders → Fills → P&L → Exit.

---

## Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DLE Episode",
  "type": "object",
  "required": [
    "episode_id",
    "symbol",
    "direction",
    "state",
    "created_ts",
    "phase_id",
    "authority",
    "entry",
    "position"
  ],
  "properties": {
    "episode_id": {
      "type": "string",
      "description": "Unique episode identifier",
      "pattern": "^[a-f0-9-]{36}$"
    },
    "symbol": {
      "type": "string"
    },
    "direction": {
      "enum": ["LONG", "SHORT"]
    },
    "state": {
      "enum": ["PENDING", "OPEN", "CLOSING", "CLOSED", "DENIED"],
      "description": "Current episode state"
    },
    "created_ts": {
      "type": "string",
      "format": "date-time"
    },
    "closed_ts": {
      "type": "string",
      "format": "date-time"
    },
    "phase_id": {
      "type": "string",
      "description": "Phase this episode belongs to"
    },
    
    "authority": {
      "type": "object",
      "description": "What authorized this episode",
      "required": ["decision_id"],
      "properties": {
        "decision_id": {
          "type": "string",
          "description": "Decision that authorized entry"
        },
        "permit_id": {
          "type": "string",
          "description": "Permit issued for entry"
        },
        "requester": {
          "type": "object",
          "properties": {
            "type": { "type": "string" },
            "id": { "type": "string" }
          }
        },
        "rationale": {
          "type": "string",
          "description": "Why this trade was taken"
        }
      }
    },
    
    "denied": {
      "type": "object",
      "description": "If state=DENIED, why",
      "properties": {
        "deny_reason": { "type": "string" },
        "deny_details": { "type": "string" },
        "denied_ts": { "type": "string", "format": "date-time" }
      }
    },
    
    "entry": {
      "type": "object",
      "description": "Entry execution details",
      "properties": {
        "intent_price": { "type": "number" },
        "intent_size": { "type": "number" },
        "intent_notional_usd": { "type": "number" },
        "filled_price_avg": { "type": "number" },
        "filled_size": { "type": "number" },
        "filled_notional_usd": { "type": "number" },
        "slippage_bps": { "type": "number" },
        "order_ids": {
          "type": "array",
          "items": { "type": "string" }
        },
        "entry_ts": { "type": "string", "format": "date-time" }
      }
    },
    
    "position": {
      "type": "object",
      "description": "Current/final position state",
      "properties": {
        "size": { "type": "number" },
        "entry_price": { "type": "number" },
        "current_price": { "type": "number" },
        "unrealized_pnl_usd": { "type": "number" },
        "unrealized_pnl_pct": { "type": "number" }
      }
    },
    
    "risk_management": {
      "type": "object",
      "description": "TP/SL configuration",
      "properties": {
        "stop_loss_price": { "type": "number" },
        "stop_loss_pct": { "type": "number" },
        "take_profit_price": { "type": "number" },
        "take_profit_pct": { "type": "number" },
        "trailing_stop_pct": { "type": "number" }
      }
    },
    
    "exit": {
      "type": "object",
      "description": "Exit details (if closed)",
      "properties": {
        "exit_reason": {
          "enum": [
            "TAKE_PROFIT",
            "STOP_LOSS",
            "TRAILING_STOP",
            "THESIS_INVALIDATED",
            "REGIME_CHANGE",
            "MANUAL_CLOSE",
            "RISK_VETO",
            "PHASE_END",
            "DECISION_EXPIRED",
            "EMERGENCY_HALT"
          ]
        },
        "exit_trigger": {
          "type": "string",
          "description": "What triggered the exit"
        },
        "exit_decision_id": {
          "type": "string",
          "description": "Decision that authorized exit (if different from entry)"
        },
        "exit_permit_id": {
          "type": "string"
        },
        "exit_price_avg": { "type": "number" },
        "exit_size": { "type": "number" },
        "exit_slippage_bps": { "type": "number" },
        "exit_order_ids": {
          "type": "array",
          "items": { "type": "string" }
        },
        "exit_ts": { "type": "string", "format": "date-time" }
      }
    },
    
    "outcome": {
      "type": "object",
      "description": "Final P&L and metrics",
      "properties": {
        "realized_pnl_usd": { "type": "number" },
        "realized_pnl_pct": { "type": "number" },
        "fees_usd": { "type": "number" },
        "net_pnl_usd": { "type": "number" },
        "net_pnl_pct": { "type": "number" },
        "hold_duration_s": { "type": "integer" },
        "max_favorable_excursion_pct": { "type": "number" },
        "max_adverse_excursion_pct": { "type": "number" },
        "win": { "type": "boolean" }
      }
    },
    
    "snapshots": {
      "type": "object",
      "description": "State snapshots for replay",
      "properties": {
        "entry_snapshot_hash": { "type": "string" },
        "exit_snapshot_hash": { "type": "string" },
        "regime_at_entry": { "type": "string" },
        "regime_at_exit": { "type": "string" },
        "nav_at_entry": { "type": "number" },
        "nav_at_exit": { "type": "number" }
      }
    },
    
    "audit_trail": {
      "type": "array",
      "description": "Chronological list of all events in this episode",
      "items": {
        "type": "object",
        "properties": {
          "ts": { "type": "string", "format": "date-time" },
          "event": { "type": "string" },
          "details": { "type": "object" }
        }
      }
    },
    
    "metadata": {
      "type": "object"
    }
  },
  "additionalProperties": false
}
```

---

## Episode States

```
PENDING → OPEN → CLOSING → CLOSED
    │
    └──→ DENIED (if entry permit refused)
```

| State | Meaning |
|-------|---------|
| `PENDING` | Request made, awaiting permit |
| `OPEN` | Position active |
| `CLOSING` | Exit in progress |
| `CLOSED` | Episode complete |
| `DENIED` | Entry was refused |

---

## Ledger File

**Path:** `logs/dle/episode_ledger.jsonl`  
**Format:** Newline-delimited JSON (append-only)

Episodes are updated by appending new entries with same `episode_id` and updated fields.

---

## Example: Complete Episode

```json
{
  "episode_id": "ep_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "symbol": "BTCUSDT",
  "direction": "LONG",
  "state": "CLOSED",
  "created_ts": "2026-01-15T10:00:00Z",
  "closed_ts": "2026-01-15T14:30:00Z",
  "phase_id": "CYCLE_002",
  
  "authority": {
    "decision_id": "d7f3a2b1-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
    "permit_id": "p1a2b3c4-d5e6-f7a8-b9c0-d1e2f3a4b5c6",
    "requester": { "type": "HYDRA_HEAD", "id": "TREND" },
    "rationale": "TREND_UP regime with 72% confidence"
  },
  
  "entry": {
    "intent_price": 98500.00,
    "intent_notional_usd": 1500,
    "filled_price_avg": 98512.50,
    "filled_notional_usd": 1498.75,
    "slippage_bps": 1.27,
    "order_ids": ["ord_entry_001"],
    "entry_ts": "2026-01-15T10:00:15Z"
  },
  
  "position": {
    "size": 0.0152,
    "entry_price": 98512.50
  },
  
  "risk_management": {
    "stop_loss_price": 96542.25,
    "stop_loss_pct": 0.02,
    "take_profit_price": 101467.88,
    "take_profit_pct": 0.03
  },
  
  "exit": {
    "exit_reason": "TAKE_PROFIT",
    "exit_trigger": "Price reached TP level",
    "exit_price_avg": 101455.00,
    "exit_slippage_bps": 1.27,
    "exit_order_ids": ["ord_exit_001"],
    "exit_ts": "2026-01-15T14:30:00Z"
  },
  
  "outcome": {
    "realized_pnl_usd": 44.75,
    "realized_pnl_pct": 0.0299,
    "fees_usd": 2.40,
    "net_pnl_usd": 42.35,
    "net_pnl_pct": 0.0283,
    "hold_duration_s": 16185,
    "max_favorable_excursion_pct": 0.032,
    "max_adverse_excursion_pct": 0.008,
    "win": true
  },
  
  "snapshots": {
    "regime_at_entry": "TREND_UP",
    "regime_at_exit": "TREND_UP",
    "nav_at_entry": 10500.00,
    "nav_at_exit": 10542.35
  }
}
```

---

## Example: Denied Episode

```json
{
  "episode_id": "ep_denied_xyz",
  "symbol": "ETHUSDT",
  "direction": "SHORT",
  "state": "DENIED",
  "created_ts": "2026-01-28T14:00:00Z",
  "phase_id": "CYCLE_003",
  
  "authority": {
    "requester": { "type": "HYDRA_HEAD", "id": "MEAN_REVERT" },
    "rationale": "Mean reversion signal on ETH"
  },
  
  "denied": {
    "deny_reason": "DENY_REGIME_FORBIDDEN",
    "deny_details": "CHOPPY regime is in decision's forbidden_regime list",
    "denied_ts": "2026-01-28T14:00:01Z"
  }
}
```

---

## Migration from Current `episode_ledger.json`

Current episode ledger lacks:
- `decision_id` / `permit_id` binding
- Explicit `denied` episodes
- State snapshot hashes
- Audit trail

Migration path:
1. Add DLE fields to episode writer
2. Backfill `decision_id: "PRE_DLE"` for historical episodes
3. Start logging denied episodes (currently not tracked)

---

## Canonical Examples

### Valid: Minimal Open Episode

```json
{
  "episode_id": "ep_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "symbol": "BTCUSDT",
  "direction": "LONG",
  "state": "OPEN",
  "created_ts": "2026-01-28T14:00:00Z",
  "phase_id": "CYCLE_004",
  "authority": {
    "decision_id": "d7f3a2b1-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
    "permit_id": "p1a2b3c4-d5e6-f7a8-b9c0-d1e2f3a4b5c6"
  },
  "entry": {
    "filled_price_avg": 104250.50,
    "filled_notional_usd": 1500
  },
  "position": {
    "size": 0.0144,
    "entry_price": 104250.50
  }
}
```

### Valid: Closed Episode with Full Outcome

See "Example: Complete Episode" above for full closed episode with all fields.

### Invalid: Missing Authority Block

```json
{
  "episode_id": "ep_abc123",
  "symbol": "BTCUSDT",
  "direction": "LONG",
  "state": "OPEN",
  "created_ts": "2026-01-28T14:00:00Z",
  "phase_id": "CYCLE_004",
  "entry": { "filled_price_avg": 104250.50 },
  "position": { "size": 0.0144, "entry_price": 104250.50 }
}
```
**Rejection reason:** Missing required `authority` block. Cannot trace episode to decision.

### Invalid: State/Fields Mismatch

```json
{
  "episode_id": "ep_abc123",
  "symbol": "BTCUSDT",
  "direction": "LONG",
  "state": "CLOSED",
  "created_ts": "2026-01-28T14:00:00Z",
  "phase_id": "CYCLE_004",
  "authority": { "decision_id": "d_abc" },
  "entry": { "filled_price_avg": 104250.50 },
  "position": { "size": 0.0144, "entry_price": 104250.50 }
}
```
**Rejection reason:** State is `CLOSED` but missing `exit`, `outcome`, and `closed_ts`. Inconsistent state.

### Invalid: PRE_DLE Without Marker

```json
{
  "episode_id": "EP_0021",
  "symbol": "ETHUSDT",
  "direction": "LONG",
  "state": "CLOSED",
  "created_ts": "2026-01-20T22:04:49Z",
  "phase_id": "CYCLE_002",
  "authority": {
    "decision_id": null,
    "permit_id": null
  },
  "entry": { ... },
  "outcome": { ... }
}
```
**Rejection reason:** For pre-DLE episodes, `decision_id` should be `"PRE_DLE"` (explicit marker), not `null`.
