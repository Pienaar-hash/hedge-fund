# DLE Deny Reasons v1

> **Status:** SPECIFICATION — partially implemented (C.1 uses `ENTRY_DENIED_*` subset: `NO_PERMIT`, `EXPIRED_PERMIT`, `MISMATCH_SYMBOL`, `MISMATCH_DIRECTION`, `INDEX_UNAVAILABLE`)  
> **Implementation:** `execution/enforcement_gate.py`, `v7_manifest.json → dle_entry_denials`  
> **Author:** GPT-Hedge governance audit  
> **Date:** 2026-01-28  
> **Updated:** 2026-02-18 (status only — spec text unchanged)

## Purpose

When the DLE Gate refuses to issue a permit, it must return a **specific, actionable denial reason**. This document defines the canonical set of denial codes.

---

## Denial Code Taxonomy

### Authority Denials

| Code | Meaning | Resolution |
|------|---------|------------|
| `DENY_NO_DECISION` | No active decision permits this action | Create a decision, or wait for conditions to match an existing one |
| `DENY_DECISION_EXPIRED` | A matching decision exists but has expired | Decision must be renewed |
| `DENY_DECISION_EXHAUSTED` | Decision's `max_uses` reached | New decision required |
| `DENY_FORBIDDEN_ACTION` | Action is explicitly forbidden by active decision | Cannot proceed; action is prohibited |
| `DENY_OUT_OF_SCOPE` | Action outside decision's scope (symbol, direction, size) | Reduce scope or get broader decision |

### Conflict Denials

| Code | Meaning | Resolution |
|------|---------|------------|
| `DENY_CONFLICT` | Multiple decisions conflict on this action | Escalation required; no heuristic merge |
| `DENY_SUPERSEDED` | Referenced decision has been superseded | Use current decision |

### Phase Denials

| Code | Meaning | Resolution |
|------|---------|------------|
| `DENY_PHASE_MISMATCH` | Request phase doesn't match decision phase | Wait for correct phase or get phase-appropriate decision |
| `DENY_PHASE_BOUNDARY` | Execution would cross phase boundary | Complete before phase ends or defer to next phase |

### Constraint Denials

| Code | Meaning | Resolution |
|------|---------|------------|
| `DENY_REGIME_MISMATCH` | Current regime not in decision's `required_regime` | Wait for regime change |
| `DENY_REGIME_FORBIDDEN` | Current regime is in decision's `forbidden_regime` | Cannot proceed in this regime |
| `DENY_CONFIDENCE_LOW` | Regime confidence below decision's `min_confidence` | Wait for higher confidence |
| `DENY_SIZE_EXCEEDS_SCOPE` | Requested size exceeds decision's `max_notional_usd` | Reduce size |
| `DENY_HEAT_EXCEEDED` | Portfolio heat exceeds decision's `max_portfolio_heat` | Reduce exposure first |
| `DENY_CORRELATION_EXCEEDED` | Correlation exposure exceeds limit | Reduce correlated positions |

### Risk Denials

| Code | Meaning | Resolution |
|------|---------|------------|
| `DENY_DRAWDOWN_HALT` | Drawdown exceeds `drawdown_halt_pct` | Decision invalidated; wait for recovery |
| `DENY_RISK_VETO` | Risk engine veto (secondary to DLE) | Address risk condition |
| `DENY_NAV_STALE` | NAV data too old for safe execution | Wait for fresh NAV |
| `DENY_STATE_DRIFT` | State has drifted since permit request | Re-request with fresh context |

### System Denials

| Code | Meaning | Resolution |
|------|---------|------------|
| `DENY_AMBIGUOUS` | Cannot determine permission (fail-closed) | Clarify request or decision |
| `DENY_INTERNAL_ERROR` | Gate encountered an error | Investigate logs; do not retry blindly |
| `DENY_RATE_LIMITED` | Too many requests in window | Back off |

### Permit Denials

| Code | Meaning | Resolution |
|------|---------|------------|
| `DENY_PERMIT_CONSUMED` | Permit has already been used | Request new permit |
| `DENY_PERMIT_EXPIRED` | Permit TTL exceeded before use | Request new permit |
| `DENY_PERMIT_REVOKED` | Permit was revoked before use | Request new permit |
| `DENY_NO_PERMIT` | Order attempted without permit | Must request permit first |

---

## Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "DLE Denial",
  "type": "object",
  "required": [
    "request_id",
    "denied_ts",
    "deny_reason",
    "deny_details"
  ],
  "properties": {
    "request_id": {
      "type": "string",
      "description": "The request that was denied"
    },
    "denied_ts": {
      "type": "string",
      "format": "date-time"
    },
    "deny_reason": {
      "enum": [
        "DENY_NO_DECISION",
        "DENY_DECISION_EXPIRED",
        "DENY_DECISION_EXHAUSTED",
        "DENY_FORBIDDEN_ACTION",
        "DENY_OUT_OF_SCOPE",
        "DENY_CONFLICT",
        "DENY_SUPERSEDED",
        "DENY_PHASE_MISMATCH",
        "DENY_PHASE_BOUNDARY",
        "DENY_REGIME_MISMATCH",
        "DENY_REGIME_FORBIDDEN",
        "DENY_CONFIDENCE_LOW",
        "DENY_SIZE_EXCEEDS_SCOPE",
        "DENY_HEAT_EXCEEDED",
        "DENY_CORRELATION_EXCEEDED",
        "DENY_DRAWDOWN_HALT",
        "DENY_RISK_VETO",
        "DENY_NAV_STALE",
        "DENY_STATE_DRIFT",
        "DENY_AMBIGUOUS",
        "DENY_INTERNAL_ERROR",
        "DENY_RATE_LIMITED",
        "DENY_PERMIT_CONSUMED",
        "DENY_PERMIT_EXPIRED",
        "DENY_PERMIT_REVOKED",
        "DENY_NO_PERMIT"
      ]
    },
    "deny_details": {
      "type": "string",
      "description": "Human-readable explanation",
      "maxLength": 512
    },
    "deny_context": {
      "type": "object",
      "description": "Relevant state at denial time",
      "properties": {
        "decision_id": { "type": "string" },
        "expected_value": { "type": "string" },
        "actual_value": { "type": "string" },
        "threshold": { "type": "number" }
      }
    },
    "recoverable": {
      "type": "boolean",
      "description": "Whether this denial might resolve on retry"
    },
    "retry_after_s": {
      "type": "integer",
      "description": "Suggested wait before retry (if recoverable)"
    }
  },
  "additionalProperties": false
}
```

---

## Example Denials

### No Decision

```json
{
  "request_id": "req_abc123",
  "denied_ts": "2026-01-28T14:35:00Z",
  "deny_reason": "DENY_NO_DECISION",
  "deny_details": "No active decision permits OPEN_SHORT for ETHUSDT in phase CYCLE_003",
  "recoverable": false
}
```

### Regime Mismatch

```json
{
  "request_id": "req_def456",
  "denied_ts": "2026-01-28T14:35:00Z",
  "deny_reason": "DENY_REGIME_MISMATCH",
  "deny_details": "Decision d7f3a2b1 requires regime TREND_UP but current regime is CHOPPY",
  "deny_context": {
    "decision_id": "d7f3a2b1-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
    "expected_value": "TREND_UP",
    "actual_value": "CHOPPY"
  },
  "recoverable": true,
  "retry_after_s": 300
}
```

### Conflict

```json
{
  "request_id": "req_ghi789",
  "denied_ts": "2026-01-28T14:35:00Z",
  "deny_reason": "DENY_CONFLICT",
  "deny_details": "Decisions d7f3a2b1 and e8g4b3c2 conflict on OPEN_LONG for BTCUSDT",
  "deny_context": {
    "decision_id": "d7f3a2b1,e8g4b3c2"
  },
  "recoverable": false
}
```

---

## Ledger Integration

All denials are logged to:

**Path:** `logs/dle/denials.jsonl`

This provides a complete audit trail of "what was refused and why."

---

## Mapping to Current GPT-Hedge Vetoes

| Current Veto | DLE Code |
|--------------|----------|
| `doctrine_veto` | `DENY_NO_DECISION` or `DENY_REGIME_MISMATCH` |
| `risk_veto` | `DENY_RISK_VETO` |
| `nav_stale` | `DENY_NAV_STALE` |
| `per_symbol_cap` | `DENY_SIZE_EXCEEDS_SCOPE` |
| `correlation_cap` | `DENY_CORRELATION_EXCEEDED` |
| `portfolio_dd` | `DENY_DRAWDOWN_HALT` |
| `min_notional` | `DENY_OUT_OF_SCOPE` |

---

## Canonical Examples

### Valid: Minimal Denial

```json
{
  "request_id": "req_a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "denied_ts": "2026-01-28T14:35:00Z",
  "deny_reason": "DENY_NO_DECISION",
  "deny_details": "No active decision permits OPEN_SHORT for ETHUSDT"
}
```

### Valid: Denial with Context

```json
{
  "request_id": "req_b2c3d4e5-f6a7-8901-bcde-f23456789012",
  "denied_ts": "2026-01-28T14:35:00Z",
  "deny_reason": "DENY_REGIME_MISMATCH",
  "deny_details": "Decision d7f3a2b1 requires TREND_UP but current regime is CHOPPY",
  "deny_context": {
    "decision_id": "d7f3a2b1-4c5e-6f7a-8b9c-0d1e2f3a4b5c",
    "expected_value": "TREND_UP",
    "actual_value": "CHOPPY"
  },
  "recoverable": true,
  "retry_after_s": 300
}
```

### Invalid: Missing Required Fields

```json
{
  "denied_ts": "2026-01-28T14:35:00Z",
  "deny_reason": "DENY_NO_DECISION"
}
```
**Rejection reason:** Missing `request_id` and `deny_details`. Cannot trace denial to request.

### Invalid: Unknown Deny Reason

```json
{
  "request_id": "req_abc123",
  "denied_ts": "2026-01-28T14:35:00Z",
  "deny_reason": "DENY_UNKNOWN_ERROR",
  "deny_details": "Something went wrong"
}
```
**Rejection reason:** `deny_reason` must be from canonical enum. Use `DENY_INTERNAL_ERROR` for unexpected failures.

This allows backward compatibility during migration.
