# DLE Permit Schema v1

> **Status:** SPECIFICATION — not yet implemented  
> **Author:** GPT-Hedge governance audit  
> **Date:** 2026-01-28

## Purpose

A **Permit** is a single-use execution token issued by the DLE Gate. It proves that a specific action was authorized by a specific decision at a specific moment.

Permits are the **bridge between authority (Decision) and execution (Order)**.

---

## Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DLE Permit",
  "type": "object",
  "required": [
    "permit_id",
    "decision_id",
    "issued_ts",
    "action",
    "scope_snapshot",
    "expires_ts",
    "state"
  ],
  "properties": {
    "permit_id": {
      "type": "string",
      "description": "Unique permit identifier (UUID v4)",
      "pattern": "^[a-f0-9-]{36}$"
    },
    "decision_id": {
      "type": "string",
      "description": "The decision that authorized this permit"
    },
    "issued_ts": {
      "type": "string",
      "format": "date-time",
      "description": "When this permit was issued"
    },
    "action": {
      "type": "object",
      "required": ["type", "symbol", "direction"],
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
            "slippage_tolerance_pct": { "type": "number" }
          }
        }
      }
    },
    "scope_snapshot": {
      "type": "object",
      "description": "Frozen state at permit issuance (for replay verification)",
      "required": ["regime", "nav_usd", "portfolio_hash"],
      "properties": {
        "regime": {
          "type": "string",
          "description": "Active regime when permit issued"
        },
        "regime_confidence": {
          "type": "number"
        },
        "nav_usd": {
          "type": "number",
          "description": "NAV at issuance"
        },
        "portfolio_hash": {
          "type": "string",
          "description": "Hash of position state at issuance"
        },
        "risk_snapshot_hash": {
          "type": "string",
          "description": "Hash of risk state at issuance"
        },
        "market_price": {
          "type": "number",
          "description": "Symbol price at issuance"
        }
      }
    },
    "expires_ts": {
      "type": "string",
      "format": "date-time",
      "description": "Permit expires if not used by this time (short TTL)"
    },
    "state": {
      "enum": ["ISSUED", "CONSUMED", "EXPIRED", "REVOKED"],
      "description": "Current permit state"
    },
    "consumed_ts": {
      "type": "string",
      "format": "date-time",
      "description": "When permit was used (if CONSUMED)"
    },
    "consumed_by_order_id": {
      "type": "string",
      "description": "Order ID that consumed this permit"
    },
    "revoked_reason": {
      "type": "string",
      "description": "Why permit was revoked (if REVOKED)"
    },
    "metadata": {
      "type": "object",
      "description": "Optional audit context"
    }
  },
  "additionalProperties": false
}
```

---

## Invariants

1. **Single use** — A permit can only be consumed once
2. **Short TTL** — Permits expire quickly (default: 60 seconds)
3. **State frozen** — `scope_snapshot` captures state at issuance; execution must verify state hasn't drifted
4. **No permit = no order** — Executor cannot place orders without a valid permit
5. **Permit binds to order** — `consumed_by_order_id` creates audit trail

---

## Lifecycle

```
ISSUED → CONSUMED (order placed successfully)
       → EXPIRED  (TTL exceeded without use)
       → REVOKED  (decision invalidated before use)
```

---

## Ledger File

**Path:** `logs/dle/permits_issued.jsonl`  
**Format:** Newline-delimited JSON (append-only)

State transitions are logged as new entries with same `permit_id` and updated `state`.

---

## Example

```json
{
  "permit_id": "p1a2b3c4-d5e6-f7a8-b9c0-d1e2f3a4b5c6",
  "decision_id": "d7f3a2b1-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
  "issued_ts": "2026-01-28T14:35:00Z",
  "action": {
    "type": "OPEN_LONG",
    "symbol": "BTCUSDT",
    "direction": "LONG",
    "size_intent": {
      "notional_usd": 1500,
      "nav_pct": 0.14
    }
  },
  "scope_snapshot": {
    "regime": "TREND_UP",
    "regime_confidence": 0.72,
    "nav_usd": 10751.23,
    "portfolio_hash": "sha256:abc123...",
    "market_price": 104250.50
  },
  "expires_ts": "2026-01-28T14:36:00Z",
  "state": "ISSUED"
}
```

---

## Consumption Record

When permit is used:

```json
{
  "permit_id": "p1a2b3c4-d5e6-f7a8-b9c0-d1e2f3a4b5c6",
  "state": "CONSUMED",
  "consumed_ts": "2026-01-28T14:35:12Z",
  "consumed_by_order_id": "ord_789xyz"
}
```

---

## Verification at Execution

Before placing an order, executor must verify:

1. `permit.state == "ISSUED"`
2. `now < permit.expires_ts`
3. Current regime matches `scope_snapshot.regime`
4. NAV hasn't drifted >X% from `scope_snapshot.nav_usd`
5. Position state hash matches (no concurrent modifications)

If any check fails → **reject execution, log denial**.
