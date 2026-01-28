# DLE Decision Schema v1

> **Status:** SPECIFICATION — not yet implemented  
> **Author:** GPT-Hedge governance audit  
> **Date:** 2026-01-28

## Purpose

A **Decision** is the atomic unit of authority in the Decision Ledger Engine. It answers: *"Who permitted what, under what constraints, and when does that permission expire?"*

Decisions are **immutable once written**. They cannot be edited—only superseded by new decisions.

---

## Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DLE Decision",
  "type": "object",
  "required": [
    "decision_id",
    "created_ts",
    "intent",
    "scope",
    "permitted_actions",
    "forbidden_actions",
    "constraints",
    "expiration",
    "phase_id",
    "risk_acknowledgement",
    "authority_source"
  ],
  "properties": {
    "decision_id": {
      "type": "string",
      "description": "Unique immutable identifier (UUID v4 or deterministic hash)",
      "pattern": "^[a-f0-9-]{36}$"
    },
    "created_ts": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp when decision was created"
    },
    "intent": {
      "type": "string",
      "description": "Human-readable statement of what this decision authorizes",
      "maxLength": 256
    },
    "scope": {
      "type": "object",
      "description": "Boundaries of what this decision covers",
      "required": ["symbols", "directions", "max_notional_usd"],
      "properties": {
        "symbols": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Symbols this decision applies to (* = all)"
        },
        "directions": {
          "type": "array",
          "items": { "enum": ["LONG", "SHORT", "FLAT"] },
          "description": "Permitted position directions"
        },
        "max_notional_usd": {
          "type": "number",
          "minimum": 0,
          "description": "Maximum notional value this decision can authorize"
        },
        "strategy_heads": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Which Hydra heads can use this decision"
        }
      }
    },
    "permitted_actions": {
      "type": "array",
      "items": {
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
      "description": "Explicit list of actions this decision permits"
    },
    "forbidden_actions": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Explicit list of actions this decision forbids (overrides permitted)"
    },
    "constraints": {
      "type": "object",
      "description": "Execution constraints bound to this decision",
      "properties": {
        "min_confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Minimum regime confidence required"
        },
        "required_regime": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Regime(s) that must be active"
        },
        "forbidden_regime": {
          "type": "array",
          "items": { "type": "string" },
          "description": "Regime(s) that invalidate this decision"
        },
        "max_portfolio_heat": {
          "type": "number",
          "description": "Max portfolio heat allowed when using this decision"
        },
        "max_correlation_exposure": {
          "type": "number",
          "description": "Max correlated exposure allowed"
        }
      }
    },
    "expiration": {
      "type": "object",
      "required": ["mode"],
      "properties": {
        "mode": {
          "enum": ["TIME", "EVENT", "SINGLE_USE", "PHASE_END"],
          "description": "How this decision expires"
        },
        "expires_at": {
          "type": "string",
          "format": "date-time",
          "description": "For TIME mode: when decision expires"
        },
        "expires_on_event": {
          "type": "string",
          "description": "For EVENT mode: event type that invalidates"
        },
        "max_uses": {
          "type": "integer",
          "minimum": 1,
          "description": "For SINGLE_USE mode: how many permits can be issued"
        }
      }
    },
    "phase_id": {
      "type": "string",
      "description": "Phase this decision belongs to (e.g., CYCLE_003)",
      "pattern": "^[A-Z0-9_]+$"
    },
    "risk_acknowledgement": {
      "type": "object",
      "required": ["max_loss_pct", "fail_closed"],
      "properties": {
        "max_loss_pct": {
          "type": "number",
          "description": "Maximum acceptable loss as fraction of NAV"
        },
        "drawdown_halt_pct": {
          "type": "number",
          "description": "Drawdown level that invalidates this decision"
        },
        "regime_assumption": {
          "type": "string",
          "description": "Market regime this decision assumes (if any)"
        },
        "fail_closed": {
          "type": "boolean",
          "description": "If true, ambiguity = deny (must be true)"
        }
      }
    },
    "authority_source": {
      "type": "string",
      "description": "What granted this decision (DOCTRINE, MANUAL, ESCALATION)",
      "enum": ["DOCTRINE", "MANUAL_OVERRIDE", "ESCALATION_RESOLUTION"]
    },
    "supersedes": {
      "type": "string",
      "description": "decision_id this replaces (if any)"
    },
    "metadata": {
      "type": "object",
      "description": "Optional context for audit"
    }
  },
  "additionalProperties": false
}
```

---

## Invariants

1. **Immutable after creation** — Decisions cannot be edited, only superseded
2. **No implicit authority** — If no decision permits an action, it is forbidden
3. **Forbidden overrides permitted** — Explicit forbid always wins
4. **Phase-bound** — Decision invalid outside its phase
5. **Fail-closed** — `fail_closed` must always be `true`

---

## Ledger File

**Path:** `logs/dle/decision_ledger.jsonl`  
**Format:** Newline-delimited JSON (append-only)

---

## Example

```json
{
  "decision_id": "d7f3a2b1-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
  "created_ts": "2026-01-28T14:30:00Z",
  "intent": "Permit TREND head to open longs in BTC during TREND_UP regime",
  "scope": {
    "symbols": ["BTCUSDT"],
    "directions": ["LONG"],
    "max_notional_usd": 2000,
    "strategy_heads": ["TREND"]
  },
  "permitted_actions": ["OPEN_LONG", "SET_STOP_LOSS", "SET_TAKE_PROFIT"],
  "forbidden_actions": ["OPEN_SHORT", "HEDGE"],
  "constraints": {
    "min_confidence": 0.65,
    "required_regime": ["TREND_UP"],
    "forbidden_regime": ["CRISIS", "CHOPPY"],
    "max_portfolio_heat": 0.15
  },
  "expiration": {
    "mode": "EVENT",
    "expires_on_event": "REGIME_CHANGE"
  },
  "phase_id": "CYCLE_004",
  "risk_acknowledgement": {
    "max_loss_pct": 0.02,
    "drawdown_halt_pct": 0.05,
    "regime_assumption": "TREND_UP",
    "fail_closed": true
  },
  "authority_source": "DOCTRINE"
}
```

---

## Migration Notes

Current `doctrine_events.jsonl` contains mixed authority + telemetry. To migrate:

1. Extract authority statements → `decision_ledger.jsonl`
2. Keep observations/vetoes → `doctrine_events.jsonl` (now pure telemetry)
3. Backfill `decision_id` into existing episode records where possible

---

## Canonical Examples

### Valid: Minimal Decision

```json
{
  "decision_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "created_ts": "2026-01-28T14:30:00Z",
  "intent": "Permit TREND head to open longs in BTC",
  "scope": {
    "symbols": ["BTCUSDT"],
    "directions": ["LONG"],
    "max_notional_usd": 1500
  },
  "permitted_actions": ["OPEN_LONG"],
  "forbidden_actions": [],
  "constraints": {},
  "expiration": {
    "mode": "SINGLE_USE",
    "max_uses": 1
  },
  "phase_id": "CYCLE_004",
  "risk_acknowledgement": {
    "max_loss_pct": 0.02,
    "fail_closed": true
  },
  "authority_source": "DOCTRINE"
}
```

### Invalid: Missing Required Fields

```json
{
  "decision_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "created_ts": "2026-01-28T14:30:00Z",
  "intent": "Permit trading"
}
```
**Rejection reason:** Missing `scope`, `permitted_actions`, `forbidden_actions`, `constraints`, `expiration`, `phase_id`, `risk_acknowledgement`, `authority_source`.

### Invalid: fail_closed = false

```json
{
  "decision_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "created_ts": "2026-01-28T14:30:00Z",
  "intent": "Permit trading",
  "scope": { "symbols": ["*"], "directions": ["LONG", "SHORT"], "max_notional_usd": 10000 },
  "permitted_actions": ["OPEN_LONG", "OPEN_SHORT"],
  "forbidden_actions": [],
  "constraints": {},
  "expiration": { "mode": "TIME", "expires_at": "2026-02-01T00:00:00Z" },
  "phase_id": "CYCLE_004",
  "risk_acknowledgement": {
    "max_loss_pct": 0.05,
    "fail_closed": false
  },
  "authority_source": "DOCTRINE"
}
```
**Rejection reason:** `fail_closed` MUST be `true`. This is a constitutional invariant.

### Invalid: Timestamp Without Timezone

```json
{
  "decision_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "created_ts": "2026-01-28T14:30:00",
  "intent": "Permit trading",
  ...
}
```
**Rejection reason:** Timestamp missing timezone suffix. Must use ISO 8601 with `Z` or `+00:00`.
