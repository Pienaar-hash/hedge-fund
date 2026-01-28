# DLE Execution Request Schema v1

> **Status:** SPECIFICATION — not yet implemented  
> **Author:** GPT-Hedge governance audit  
> **Date:** 2026-01-28

## Purpose

An **Execution Request** is the input to the DLE Gate. It represents "I want to do X" and asks the ledger for permission.

The gate returns either a **Permit** (permission granted) or a **Denial** (permission refused with reason).

---

## Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DLE Execution Request",
  "type": "object",
  "required": [
    "request_id",
    "requested_ts",
    "requester",
    "action",
    "context"
  ],
  "properties": {
    "request_id": {
      "type": "string",
      "description": "Unique request identifier",
      "pattern": "^[a-f0-9-]{36}$"
    },
    "requested_ts": {
      "type": "string",
      "format": "date-time",
      "description": "When this request was made"
    },
    "requester": {
      "type": "object",
      "required": ["type", "id"],
      "properties": {
        "type": {
          "enum": ["HYDRA_HEAD", "EXIT_SCANNER", "MANUAL", "RISK_ENGINE"],
          "description": "What component is requesting execution"
        },
        "id": {
          "type": "string",
          "description": "Identifier (e.g., head name, user id)"
        }
      }
    },
    "action": {
      "type": "object",
      "required": ["type", "symbol"],
      "properties": {
        "type": {
          "enum": [
            "OPEN_LONG",
            "OPEN_SHORT",
            "CLOSE_POSITION",
            "ADD_TO_POSITION",
            "REDUCE_POSITION",
            "SET_STOP_LOSS",
            "SET_TAKE_PROFIT",
            "CANCEL_ORDER",
            "HEDGE"
          ]
        },
        "symbol": {
          "type": "string"
        },
        "direction": {
          "enum": ["LONG", "SHORT", "FLAT"]
        },
        "size_intent": {
          "type": "object",
          "properties": {
            "notional_usd": { "type": "number" },
            "quantity": { "type": "number" },
            "nav_pct": { "type": "number" }
          }
        },
        "price_constraint": {
          "type": "object",
          "properties": {
            "limit_price": { "type": "number" },
            "market_order": { "type": "boolean" },
            "slippage_tolerance_pct": { "type": "number" }
          }
        },
        "urgency": {
          "enum": ["NORMAL", "URGENT", "IMMEDIATE"],
          "description": "Execution urgency (affects routing)"
        }
      }
    },
    "context": {
      "type": "object",
      "description": "Current state snapshot for decision matching",
      "required": ["regime", "nav_usd", "positions_hash"],
      "properties": {
        "regime": {
          "type": "string",
          "description": "Current Sentinel-X regime"
        },
        "regime_confidence": {
          "type": "number"
        },
        "nav_usd": {
          "type": "number"
        },
        "positions_hash": {
          "type": "string",
          "description": "Hash of current position state"
        },
        "risk_state": {
          "type": "object",
          "properties": {
            "portfolio_heat": { "type": "number" },
            "drawdown_pct": { "type": "number" },
            "correlation_exposure": { "type": "number" }
          }
        },
        "market_state": {
          "type": "object",
          "properties": {
            "symbol_price": { "type": "number" },
            "spread_bps": { "type": "number" },
            "volatility": { "type": "number" }
          }
        },
        "phase_id": {
          "type": "string",
          "description": "Current operational phase"
        }
      }
    },
    "rationale": {
      "type": "string",
      "description": "Why this action is being requested (for audit)",
      "maxLength": 512
    },
    "metadata": {
      "type": "object",
      "description": "Optional context"
    }
  },
  "additionalProperties": false
}
```

---

## Gate Response

The DLE Gate returns one of:

### Success: Permit Issued

```json
{
  "status": "PERMIT_ISSUED",
  "permit_id": "p1a2b3c4-...",
  "decision_id": "d7f3a2b1-...",
  "expires_ts": "2026-01-28T14:36:00Z"
}
```

### Failure: Denial

```json
{
  "status": "DENIED",
  "deny_reason": "DENY_NO_DECISION",
  "deny_details": "No active decision permits OPEN_SHORT for BTCUSDT",
  "request_id": "req_abc123"
}
```

See [DLE_DENY_REASONS.md](DLE_DENY_REASONS.md) for all denial codes.

---

## Example Request

```json
{
  "request_id": "req_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "requested_ts": "2026-01-28T14:35:00Z",
  "requester": {
    "type": "HYDRA_HEAD",
    "id": "TREND"
  },
  "action": {
    "type": "OPEN_LONG",
    "symbol": "BTCUSDT",
    "direction": "LONG",
    "size_intent": {
      "notional_usd": 1500,
      "nav_pct": 0.14
    },
    "urgency": "NORMAL"
  },
  "context": {
    "regime": "TREND_UP",
    "regime_confidence": 0.72,
    "nav_usd": 10751.23,
    "positions_hash": "sha256:abc123...",
    "risk_state": {
      "portfolio_heat": 0.0,
      "drawdown_pct": 0.02,
      "correlation_exposure": 0.0
    },
    "market_state": {
      "symbol_price": 104250.50,
      "spread_bps": 2.5,
      "volatility": 0.015
    },
    "phase_id": "CYCLE_004"
  },
  "rationale": "TREND head detected bullish momentum, regime supports long entry"
}
```

---

## Ledger File

**Path:** `logs/dle/execution_requests.jsonl`  
**Format:** Newline-delimited JSON (append-only)

Every request is logged regardless of outcome. Response is logged as a separate entry with matching `request_id`.

---

## Flow

```
Hydra/Exit Scanner/Risk Engine
         │
         ▼
   ExecutionRequest
         │
         ▼
    ┌─────────┐
    │DLE Gate │ ◄── Queries decision_ledger.jsonl
    └────┬────┘
         │
    ┌────┴────┐
    │         │
 PERMIT    DENIAL
    │         │
    ▼         ▼
 Execute   Log + Stop
```

---

## Canonical Examples

### Valid: Minimal Request

```json
{
  "request_id": "req_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "requested_ts": "2026-01-28T14:35:00Z",
  "requester": {
    "type": "HYDRA_HEAD",
    "id": "TREND"
  },
  "action": {
    "type": "OPEN_LONG",
    "symbol": "BTCUSDT"
  },
  "context": {
    "regime": "TREND_UP",
    "nav_usd": 10751.23,
    "positions_hash": "sha256:abc123def456"
  }
}
```

### Valid: Full Request with Size Intent

```json
{
  "request_id": "req_b2c3d4e5-f6a7-8901-bcde-f23456789012",
  "requested_ts": "2026-01-28T14:35:00Z",
  "requester": {
    "type": "HYDRA_HEAD",
    "id": "MEAN_REVERT"
  },
  "action": {
    "type": "OPEN_SHORT",
    "symbol": "ETHUSDT",
    "direction": "SHORT",
    "size_intent": {
      "notional_usd": 1200,
      "nav_pct": 0.11
    },
    "urgency": "NORMAL"
  },
  "context": {
    "regime": "MEAN_REVERT",
    "regime_confidence": 0.68,
    "nav_usd": 10751.23,
    "positions_hash": "sha256:abc123def456",
    "risk_state": {
      "portfolio_heat": 0.05,
      "drawdown_pct": 0.01
    },
    "phase_id": "CYCLE_004"
  },
  "rationale": "Mean reversion signal: ETH overbought RSI 78, expecting pullback"
}
```

### Invalid: Missing Required Fields

```json
{
  "request_id": "req_abc123",
  "action": {
    "type": "OPEN_LONG",
    "symbol": "BTCUSDT"
  }
}
```
**Rejection reason:** Missing `requested_ts`, `requester`, `context`. All are required.

### Invalid: Unknown Requester Type

```json
{
  "request_id": "req_abc123",
  "requested_ts": "2026-01-28T14:35:00Z",
  "requester": {
    "type": "UNKNOWN_COMPONENT",
    "id": "something"
  },
  "action": { "type": "OPEN_LONG", "symbol": "BTCUSDT" },
  "context": { "regime": "TREND_UP", "nav_usd": 10000, "positions_hash": "sha256:abc" }
}
```
**Rejection reason:** `requester.type` must be one of: `HYDRA_HEAD`, `EXIT_SCANNER`, `MANUAL`, `RISK_ENGINE`.

### Invalid: Action Type Mismatch

```json
{
  "request_id": "req_abc123",
  "requested_ts": "2026-01-28T14:35:00Z",
  "requester": { "type": "HYDRA_HEAD", "id": "TREND" },
  "action": {
    "type": "OPEN_LONG",
    "symbol": "BTCUSDT",
    "direction": "SHORT"
  },
  "context": { "regime": "TREND_UP", "nav_usd": 10000, "positions_hash": "sha256:abc" }
}
```
**Rejection reason:** `action.type` is `OPEN_LONG` but `direction` is `SHORT`. Inconsistent.
