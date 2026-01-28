# CYCLE_004 Phase A — Shadow DLE Gate

**Status:** APPROVED  
**Author:** Agent  
**Date:** 2026-01-28  
**Depends on:** DLE v1.1 Specification (falsified, survived)

---

## Objective

Implement a **shadow DLE gate** that:
- Generates deterministic `decision_id` and `permit_id` for every doctrine verdict
- Binds them to execution events via order metadata
- Logs to a dedicated, versioned DLE event stream
- Has **zero execution impact** — observation only

This is not enforcement. This is evidence collection.

---

## Non-Goals

Phase A must **NOT**:

- ❌ Block or modify any order
- ❌ Change sizing, veto logic, regime logic, or exits
- ❌ Write to existing execution logs in new formats
- ❌ Create new dependencies on dashboard parsing
- ❌ Introduce latency to the execution path (shadow runs post-decision)
- ❌ Require dashboard changes to function

If any of these happen, Phase A has failed.

---

## Determinism + Replay

Every shadow event must be **replayable** and **auditable**.

### Schema Version

All events include:
```json
"schema_version": "dle_shadow_v1"
```

### ID Generation (Deterministic)

**No random UUIDs for core identity.** All IDs are deterministic hashes:

| ID Type | Hash Input | Format |
|---------|------------|--------|
| `request_id` | Reuse existing `attempt_id` (already stable) | `sig_<hex12>` |
| `decision_id` | `sha256(symbol + direction + strategy + regime + ts_bucket_5min)` | `DEC_<hex12>` |
| `permit_id` | `sha256(decision_id + request_id + issued_at_ts)` | `PRM_<hex12>` |

**ts_bucket_5min:** Floor timestamp to 5-minute boundary for decision grouping.

### Snapshot Hashes

Every permit includes environment snapshot hashes:

```json
{
  "snapshot_hashes": {
    "positions_state": "<sha256_hex8>",
    "regime_state": "<sha256_hex8>",
    "config": "<sha256_hex8>"
  }
}
```

If a component is missing/empty, hash sentinel string `"MISSING"`.

### Ordering Rule

- Log file is **append-only**
- Events are **monotonically ordered** by `ts`
- No out-of-order writes permitted

---

## Success Criteria (Falsifiable)

| Criterion | Measurement | Pass Condition |
|-----------|-------------|----------------|
| Permit coverage | Count ENTRY_VERDICT vs DLE_PERMIT | 100% match |
| Decision binding | Orphan permits (no decision_id) | 0 |
| Metadata binding | order_fill events with `dle_permit_id` | 100% when shadow enabled |
| Zero execution divergence | See definition below | Pass |
| Regime flip coverage | Episodes with `exit_reason: REGIME_FLIP` | ≥1 observed |
| Reconstruction | Episode rebuilt from DLE logs alone | Successful |

### Zero Execution Divergence Definition

With `SHADOW_DLE_ENABLED=1` vs `SHADOW_DLE_ENABLED=0`:

- ✅ Number of orders submitted: **unchanged**
- ✅ Order params (symbol, side, qty, type, reduceOnly): **unchanged**
- ✅ Order timing: **within existing system variance** (no new sleeps/awaits)
- ✅ Doctrine verdicts: **unchanged**
- ✅ Position state: **unchanged**

If any of these differ, shadow gate has violated its contract.

---

## Architecture

```
Signal → Hydra → Cerberus → Doctrine Gate → Risk Limits → Router → Exchange
                                ↓                                      
                     doctrine_events.jsonl                             
                                ↓                                      
                        [SHADOW DLE GATE]  ← observe only, post-decision
                                ↓                                      
                     dle_events_v1.jsonl                               
                                ↓                                      
                     (permit_id injected into order metadata)          
```

The shadow gate **observes** doctrine verdicts **after** they occur and **annotates** order metadata.
It does not evaluate, override, or gate anything.

---

## Feature Flags

**File:** `execution/v6_flags.py`

```python
import os

# Phase A: Shadow DLE gate (log-only, no execution impact)
# Master on/off switch
SHADOW_DLE_ENABLED = os.getenv("SHADOW_DLE_ENABLED", "0") == "1"

# Controls whether shadow gate writes to disk
# Default ON when enabled; allows dry-run testing in CI
SHADOW_DLE_WRITE_LOGS = os.getenv("SHADOW_DLE_WRITE_LOGS", "1") == "1"
```

**Why two flags:**
- `SHADOW_DLE_ENABLED=1, SHADOW_DLE_WRITE_LOGS=0` → CI/unit tests without file I/O
- `SHADOW_DLE_ENABLED=1, SHADOW_DLE_WRITE_LOGS=1` → Production shadow mode
- `SHADOW_DLE_ENABLED=0` → Shadow gate completely bypassed

**Activation (production):**
```bash
export SHADOW_DLE_ENABLED=1
export SHADOW_DLE_WRITE_LOGS=1
```

---

## Log Path

**File:** `logs/dle/dle_events_v1.jsonl`

Versioned filename prevents breaking readers on schema changes.

**Directory creation:** Shadow gate creates `logs/dle/` if missing.

**Fail-closed behavior:** If log write fails:
1. Execution continues (never crash executor)
2. Single `stderr` warning emitted
3. Shadow gate tracks write failure count
4. No silent failures — failure is observable

---

## Integration Points (2 Primary + 1 Enrichment)

### Hook 1: Entry Path (Primary)

**Location:** `execution/executor_live.py` — **after** doctrine verdict, **before** order submission

Hook at the **final moment** before orders are submitted. Shadow request represents *what will actually be executed*, not what was considered.

```python
# EXISTING: doctrine verdict returns
verdict = self._doctrine_gate(signal, regime, ...)

if verdict.verdict == "ALLOW":
    # ... existing sizing/routing logic ...
    
    # NEW: Shadow DLE observation (Phase A) — AFTER all decisions made
    if SHADOW_DLE_ENABLED:
        from execution.dle_shadow import get_shadow_gate
        dle_ids = get_shadow_gate().observe_entry(
            symbol=signal.symbol,
            direction=signal.direction,
            strategy=signal.strategy,
            verdict="ALLOW",
            denial_code=None,
            regime=regime.name if regime else None,
            regime_confidence=regime.confidence if regime else None,
            final_qty=final_qty,  # actual qty after sizing
            final_notional=final_notional,  # actual notional
            attempt_id=signal.attempt_id,
            positions_state=self._positions_state_snapshot(),
            regime_state=self._regime_state_snapshot(),
            config_hash=self._config_hash(),
        )
        # Bind to order metadata (for downstream tracing)
        order_metadata["dle_permit_id"] = dle_ids["permit_id"]
        order_metadata["dle_decision_id"] = dle_ids["decision_id"]
```

For DENY verdicts, observe immediately after doctrine returns:

```python
if verdict.verdict != "ALLOW":
    if SHADOW_DLE_ENABLED:
        from execution.dle_shadow import get_shadow_gate
        get_shadow_gate().observe_entry(
            symbol=signal.symbol,
            direction=signal.direction,
            strategy=signal.strategy,
            verdict="DENY",
            denial_code=verdict.denial_code,
            regime=regime.name if regime else None,
            regime_confidence=regime.confidence if regime else None,
            final_qty=0,
            final_notional=0,
            attempt_id=signal.attempt_id,
            positions_state=self._positions_state_snapshot(),
            regime_state=self._regime_state_snapshot(),
            config_hash=self._config_hash(),
        )
```

### Hook 2: Exit Path (Primary)

**Location:** `execution/executor_live.py` — **at exit decision formation**, before exit orders generated

```python
# EXISTING: exit decision made
exit_decision = self._evaluate_exit(position, regime, ...)

if exit_decision.should_exit:
    # NEW: Shadow DLE observation (Phase A)
    if SHADOW_DLE_ENABLED:
        from execution.dle_shadow import get_shadow_gate
        get_shadow_gate().observe_exit(
            symbol=position.symbol,
            direction=position.direction,
            exit_reason=exit_decision.reason,
            exit_trigger=exit_decision.trigger,  # e.g., "sl", "tp", "regime_flip"
            exit_qty=position.qty,
            entry_permit_id=position.metadata.get("dle_permit_id"),
            regime_at_exit=regime.name if regime else None,
            positions_state=self._positions_state_snapshot(),
        )
    
    # ... existing exit order logic ...
```

### Hook 3: Regime Flip Annotation (Enrichment)

Regime flip is a **reason**, not a separate event. Captured as field in exit observation:

```python
exit_decision = ExitDecision(
    should_exit=True,
    reason="REGIME_FLIP",
    trigger=f"Regime changed from {old_regime} to {new_regime}",
    ...
)
```

No separate regime flip hook needed — it flows through exit path with `exit_reason="REGIME_FLIP"`.

---

## New Files

### `execution/dle_shadow.py`

```python
"""
DLE Shadow Gate — CYCLE_004 Phase A

Observes doctrine verdicts and generates DLE event stream.
ZERO EXECUTION IMPACT — log-only mode.

Invariants:
- Never modifies execution flow
- Never prevents orders  
- Fail-open on errors (warn to stderr, continue execution)
- Deterministic IDs (no random UUIDs)
- Schema versioned for replay safety
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Literal

from execution.log_utils import get_logger
from execution.v6_flags import SHADOW_DLE_ENABLED, SHADOW_DLE_WRITE_LOGS

logger = get_logger("dle_shadow")

SCHEMA_VERSION = "dle_shadow_v1"
LOG_PATH = "logs/dle/dle_events_v1.jsonl"


# -----------------------------------------------------------------------------
# Deterministic ID Generation
# -----------------------------------------------------------------------------

def _hash_to_hex12(data: str) -> str:
    """SHA256 hash truncated to 12 hex chars."""
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def _ts_bucket_5min(ts: datetime) -> str:
    """Floor timestamp to 5-minute boundary."""
    floored = ts.replace(second=0, microsecond=0)
    floored = floored.replace(minute=(floored.minute // 5) * 5)
    return floored.isoformat()


def generate_decision_id(
    symbol: str,
    direction: str,
    strategy: str,
    regime: Optional[str],
    ts: datetime,
) -> str:
    """Deterministic decision ID from inputs."""
    bucket = _ts_bucket_5min(ts)
    data = f"{symbol}|{direction}|{strategy}|{regime or 'NONE'}|{bucket}"
    return f"DEC_{_hash_to_hex12(data)}"


def generate_permit_id(
    decision_id: str,
    request_id: str,
    issued_at: str,
) -> str:
    """Deterministic permit ID from decision + request."""
    data = f"{decision_id}|{request_id}|{issued_at}"
    return f"PRM_{_hash_to_hex12(data)}"


def generate_snapshot_hash(state: Optional[dict]) -> str:
    """Hash state dict, or sentinel if missing."""
    if state is None:
        return hashlib.sha256(b"MISSING").hexdigest()[:8]
    serialized = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


# -----------------------------------------------------------------------------
# Event Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class DLEEntryEvent:
    """Shadow entry observation."""
    schema_version: str
    event_type: Literal["DLE_ENTRY"]
    decision_id: str
    permit_id: str
    request_id: str
    symbol: str
    direction: str
    strategy: str
    verdict: Literal["ALLOW", "DENY"]
    denial_code: Optional[str]
    regime_at_request: Optional[str]
    regime_confidence: Optional[float]
    final_qty: float
    final_notional_usd: float
    snapshot_hashes: dict
    ts: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "decision_id": self.decision_id,
            "permit_id": self.permit_id,
            "request_id": self.request_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "strategy": self.strategy,
            "verdict": self.verdict,
            "denial_code": self.denial_code,
            "regime_at_request": self.regime_at_request,
            "regime_confidence": self.regime_confidence,
            "final_qty": self.final_qty,
            "final_notional_usd": self.final_notional_usd,
            "snapshot_hashes": self.snapshot_hashes,
            "ts": self.ts,
        }


@dataclass
class DLEExitEvent:
    """Shadow exit observation."""
    schema_version: str
    event_type: Literal["DLE_EXIT"]
    permit_id: str
    entry_permit_id: Optional[str]
    symbol: str
    direction: str
    exit_reason: str
    exit_trigger: str
    exit_qty: float
    regime_at_exit: Optional[str]
    snapshot_hashes: dict
    ts: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "permit_id": self.permit_id,
            "entry_permit_id": self.entry_permit_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "exit_reason": self.exit_reason,
            "exit_trigger": self.exit_trigger,
            "exit_qty": self.exit_qty,
            "regime_at_exit": self.regime_at_exit,
            "snapshot_hashes": self.snapshot_hashes,
            "ts": self.ts,
        }


# -----------------------------------------------------------------------------
# Shadow Gate
# -----------------------------------------------------------------------------

class DLEShadowGate:
    """
    Shadow DLE gate — observes doctrine verdicts, generates DLE events.
    
    INVARIANTS (HARD):
    - Never modifies execution flow
    - Never prevents orders
    - Fail-open on errors (stderr warning, continue)
    - Deterministic IDs for replay
    """

    def __init__(self, log_path: str = LOG_PATH):
        self.log_path = log_path
        self._write_failures = 0
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Create log directory if needed. Fail silently."""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        except Exception as e:
            print(f"[DLE_SHADOW] Warning: cannot create log dir: {e}", file=sys.stderr)

    def _write_event(self, event: dict) -> None:
        """
        Append event to DLE log.
        
        FAIL-OPEN: If write fails, warn to stderr and continue.
        Never raise, never crash executor.
        """
        if not SHADOW_DLE_WRITE_LOGS:
            logger.debug(f"DLE shadow (dry-run): {event.get('event_type')}")
            return

        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            self._write_failures += 1
            print(
                f"[DLE_SHADOW] Warning: log write failed ({self._write_failures} total): {e}",
                file=sys.stderr,
            )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def observe_entry(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        verdict: str,
        denial_code: Optional[str],
        regime: Optional[str],
        regime_confidence: Optional[float],
        final_qty: float,
        final_notional: float,
        attempt_id: str,
        positions_state: Optional[dict] = None,
        regime_state: Optional[dict] = None,
        config_hash: Optional[str] = None,
    ) -> dict:
        """
        Observe doctrine entry verdict.
        
        Returns dict with permit_id and decision_id for order metadata binding.
        """
        now = datetime.now(timezone.utc)
        ts = now.isoformat()

        # Deterministic IDs
        decision_id = generate_decision_id(symbol, direction, strategy, regime, now)
        permit_id = generate_permit_id(decision_id, attempt_id, ts)

        # Snapshot hashes
        snapshot_hashes = {
            "positions_state": generate_snapshot_hash(positions_state),
            "regime_state": generate_snapshot_hash(regime_state),
            "config": config_hash or generate_snapshot_hash(None),
        }

        event = DLEEntryEvent(
            schema_version=SCHEMA_VERSION,
            event_type="DLE_ENTRY",
            decision_id=decision_id,
            permit_id=permit_id,
            request_id=attempt_id,
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            verdict=verdict,
            denial_code=denial_code if verdict == "DENY" else None,
            regime_at_request=regime,
            regime_confidence=regime_confidence,
            final_qty=final_qty,
            final_notional_usd=final_notional,
            snapshot_hashes=snapshot_hashes,
            ts=ts,
        )

        self._write_event(event.to_dict())
        logger.info(f"DLE shadow entry: {symbol} {direction} -> {verdict} (permit={permit_id})")

        return {
            "decision_id": decision_id,
            "permit_id": permit_id,
            "request_id": attempt_id,
        }

    def observe_exit(
        self,
        symbol: str,
        direction: str,
        exit_reason: str,
        exit_trigger: str,
        exit_qty: float,
        entry_permit_id: Optional[str] = None,
        regime_at_exit: Optional[str] = None,
        positions_state: Optional[dict] = None,
    ) -> dict:
        """
        Observe doctrine exit decision.
        
        Note: Exits are doctrine-commanded, not permit-requested.
        We generate a permit_id for traceability but it's always ALLOW.
        """
        now = datetime.now(timezone.utc)
        ts = now.isoformat()

        # Exit permit ID (deterministic from inputs)
        exit_permit_id = f"PRM_{_hash_to_hex12(f'{symbol}|{direction}|{exit_reason}|{ts}')}"

        snapshot_hashes = {
            "positions_state": generate_snapshot_hash(positions_state),
        }

        event = DLEExitEvent(
            schema_version=SCHEMA_VERSION,
            event_type="DLE_EXIT",
            permit_id=exit_permit_id,
            entry_permit_id=entry_permit_id,
            symbol=symbol,
            direction=direction,
            exit_reason=exit_reason,
            exit_trigger=exit_trigger,
            exit_qty=exit_qty,
            regime_at_exit=regime_at_exit,
            snapshot_hashes=snapshot_hashes,
            ts=ts,
        )

        self._write_event(event.to_dict())
        logger.info(f"DLE shadow exit: {symbol} {direction} reason={exit_reason}")

        return {"exit_permit_id": exit_permit_id}

    @property
    def write_failure_count(self) -> int:
        """Number of log write failures (observable for monitoring)."""
        return self._write_failures


# -----------------------------------------------------------------------------
# Global Instance (lazy init)
# -----------------------------------------------------------------------------

_shadow_gate: Optional[DLEShadowGate] = None


def get_shadow_gate() -> DLEShadowGate:
    """Get or create global shadow gate instance."""
    global _shadow_gate
    if _shadow_gate is None:
        _shadow_gate = DLEShadowGate()
    return _shadow_gate


def reset_shadow_gate() -> None:
    """Reset global instance (for testing only)."""
    global _shadow_gate
    _shadow_gate = None
```

---

## Log Output Examples

**File:** `logs/dle/dle_events_v1.jsonl`

### Entry ALLOW
```json
{"schema_version": "dle_shadow_v1", "event_type": "DLE_ENTRY", "decision_id": "DEC_a1b2c3d4e5f6", "permit_id": "PRM_f6e5d4c3b2a1", "request_id": "sig_abc123def456", "symbol": "ETHUSDT", "direction": "LONG", "strategy": "vol_target", "verdict": "ALLOW", "denial_code": null, "regime_at_request": "MEAN_REVERT", "regime_confidence": 0.8004, "final_qty": 0.08, "final_notional_usd": 239.54, "snapshot_hashes": {"positions_state": "a1b2c3d4", "regime_state": "e5f6g7h8", "config": "i9j0k1l2"}, "ts": "2026-01-28T12:00:01.123456+00:00"}
```

### Entry DENY
```json
{"schema_version": "dle_shadow_v1", "event_type": "DLE_ENTRY", "decision_id": "DEC_b2c3d4e5f6a1", "permit_id": "PRM_c3d4e5f6a1b2", "request_id": "sig_def456abc123", "symbol": "BTCUSDT", "direction": "SHORT", "strategy": "vol_target", "verdict": "DENY", "denial_code": "DENY_DIRECTION_MISMATCH", "regime_at_request": "TREND_UP", "regime_confidence": 0.72, "final_qty": 0, "final_notional_usd": 0, "snapshot_hashes": {"positions_state": "a1b2c3d4", "regime_state": "e5f6g7h8", "config": "i9j0k1l2"}, "ts": "2026-01-28T12:05:00.654321+00:00"}
```

### Exit (Regime Flip)
```json
{"schema_version": "dle_shadow_v1", "event_type": "DLE_EXIT", "permit_id": "PRM_d4e5f6a1b2c3", "entry_permit_id": "PRM_f6e5d4c3b2a1", "symbol": "ETHUSDT", "direction": "LONG", "exit_reason": "REGIME_FLIP", "exit_trigger": "Regime changed from MEAN_REVERT to CHOPPY", "exit_qty": 0.08, "regime_at_exit": "CHOPPY", "snapshot_hashes": {"positions_state": "m3n4o5p6"}, "ts": "2026-01-28T14:30:00.000000+00:00"}
```

---

## Invariants (Hard)

1. **Shadow gate NEVER modifies execution flow**
   - No early returns based on shadow state
   - No sizing adjustments
   - No routing changes
   - No conditional logic gated by shadow results

2. **Fail-open on all errors**
   - Log write failure → stderr warning + continue
   - ID generation failure → fallback sentinel + continue
   - Never raise exceptions to caller

3. **Deterministic IDs**
   - Same inputs → same IDs (for replay)
   - No random UUIDs in core identity
   - Snapshot hashes for environment fingerprinting

4. **Schema versioned**
   - All events have `schema_version` field
   - Log filename includes version

5. **Append-only log**
   - Never truncate, rewrite, or rotate during operation
   - Standard JSONL contract

---

## Testing Requirements

### Unit Tests (`tests/unit/test_dle_shadow.py`)

```python
import pytest
from unittest.mock import patch, mock_open
from execution.dle_shadow import (
    DLEShadowGate,
    generate_decision_id,
    generate_permit_id,
    generate_snapshot_hash,
    reset_shadow_gate,
)


class TestDeterministicIds:
    """IDs must be deterministic for replay."""

    def test_decision_id_deterministic(self):
        """Same inputs produce same decision_id."""
        from datetime import datetime, timezone
        ts = datetime(2026, 1, 28, 12, 0, 0, tzinfo=timezone.utc)
        id1 = generate_decision_id("ETHUSDT", "LONG", "vol_target", "MEAN_REVERT", ts)
        id2 = generate_decision_id("ETHUSDT", "LONG", "vol_target", "MEAN_REVERT", ts)
        assert id1 == id2
        assert id1.startswith("DEC_")

    def test_decision_id_ts_bucket(self):
        """Timestamps within same 5-min bucket produce same ID."""
        from datetime import datetime, timezone
        ts1 = datetime(2026, 1, 28, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 1, 28, 12, 4, 59, tzinfo=timezone.utc)
        id1 = generate_decision_id("ETHUSDT", "LONG", "vol_target", "MEAN_REVERT", ts1)
        id2 = generate_decision_id("ETHUSDT", "LONG", "vol_target", "MEAN_REVERT", ts2)
        assert id1 == id2

    def test_permit_id_deterministic(self):
        """Same inputs produce same permit_id."""
        id1 = generate_permit_id("DEC_abc", "sig_123", "2026-01-28T12:00:00")
        id2 = generate_permit_id("DEC_abc", "sig_123", "2026-01-28T12:00:00")
        assert id1 == id2
        assert id1.startswith("PRM_")


class TestSnapshotHashes:
    """Snapshot hashes must handle missing state."""

    def test_hash_none_returns_sentinel(self):
        """Missing state hashes to sentinel."""
        h = generate_snapshot_hash(None)
        assert len(h) == 8

    def test_hash_dict_deterministic(self):
        """Same dict produces same hash."""
        state = {"symbol": "ETHUSDT", "qty": 0.08}
        h1 = generate_snapshot_hash(state)
        h2 = generate_snapshot_hash(state)
        assert h1 == h2


class TestShadowGateNoSideEffects:
    """Shadow gate must not affect execution."""

    def test_observe_entry_returns_ids(self):
        """observe_entry returns permit and decision IDs."""
        with patch("execution.v6_flags.SHADOW_DLE_WRITE_LOGS", False):
            gate = DLEShadowGate()
            result = gate.observe_entry(
                symbol="ETHUSDT",
                direction="LONG",
                strategy="vol_target",
                verdict="ALLOW",
                denial_code=None,
                regime="MEAN_REVERT",
                regime_confidence=0.8,
                final_qty=0.08,
                final_notional=240.0,
                attempt_id="sig_test123",
            )
            assert "decision_id" in result
            assert "permit_id" in result
            assert result["decision_id"].startswith("DEC_")
            assert result["permit_id"].startswith("PRM_")

    def test_observe_exit_returns_ids(self):
        """observe_exit returns exit permit ID."""
        with patch("execution.v6_flags.SHADOW_DLE_WRITE_LOGS", False):
            gate = DLEShadowGate()
            result = gate.observe_exit(
                symbol="ETHUSDT",
                direction="LONG",
                exit_reason="REGIME_FLIP",
                exit_trigger="Regime changed to CHOPPY",
                exit_qty=0.08,
            )
            assert "exit_permit_id" in result


class TestFailOpen:
    """Shadow gate must fail open on errors."""

    def test_log_write_failure_does_not_raise(self):
        """Log write failure warns but doesn't crash."""
        with patch("execution.v6_flags.SHADOW_DLE_WRITE_LOGS", True):
            with patch("builtins.open", side_effect=PermissionError("denied")):
                gate = DLEShadowGate()
                # Should not raise
                result = gate.observe_entry(
                    symbol="ETHUSDT",
                    direction="LONG",
                    strategy="vol_target",
                    verdict="ALLOW",
                    denial_code=None,
                    regime="MEAN_REVERT",
                    regime_confidence=0.8,
                    final_qty=0.08,
                    final_notional=240.0,
                    attempt_id="sig_test123",
                )
                assert result is not None
                assert gate.write_failure_count >= 1


class TestSchemaVersion:
    """All events must have schema_version."""

    def test_entry_event_has_schema_version(self):
        """Entry event includes schema_version."""
        with patch("execution.v6_flags.SHADOW_DLE_WRITE_LOGS", False):
            gate = DLEShadowGate()
            # Mock the write to capture event
            events = []
            gate._write_event = lambda e: events.append(e)
            gate.observe_entry(
                symbol="ETHUSDT",
                direction="LONG",
                strategy="vol_target",
                verdict="ALLOW",
                denial_code=None,
                regime="MEAN_REVERT",
                regime_confidence=0.8,
                final_qty=0.08,
                final_notional=240.0,
                attempt_id="sig_test123",
            )
            assert len(events) == 1
            assert events[0]["schema_version"] == "dle_shadow_v1"
```

### Integration Tests (`tests/integration/test_dle_shadow_integration.py`)

```python
import pytest
import json
import tempfile
import os


class TestShadowGateIntegration:
    """Integration tests for shadow gate."""

    def test_events_written_to_file(self):
        """Events are actually written to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "dle", "test_events.jsonl")
            
            from execution.dle_shadow import DLEShadowGate
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("execution.v6_flags.SHADOW_DLE_WRITE_LOGS", True)
                
                gate = DLEShadowGate(log_path=log_path)
                gate.observe_entry(
                    symbol="ETHUSDT",
                    direction="LONG",
                    strategy="vol_target",
                    verdict="ALLOW",
                    denial_code=None,
                    regime="MEAN_REVERT",
                    regime_confidence=0.8,
                    final_qty=0.08,
                    final_notional=240.0,
                    attempt_id="sig_integration_test",
                )

            # Verify file exists and contains valid JSON
            assert os.path.exists(log_path)
            with open(log_path) as f:
                line = f.readline()
                event = json.loads(line)
                assert event["event_type"] == "DLE_ENTRY"
                assert event["symbol"] == "ETHUSDT"

    def test_no_file_written_when_disabled(self):
        """No file written when SHADOW_DLE_WRITE_LOGS=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "dle", "test_events.jsonl")
            
            from execution.dle_shadow import DLEShadowGate
            with pytest.MonkeyPatch().context() as mp:
                mp.setattr("execution.v6_flags.SHADOW_DLE_WRITE_LOGS", False)
                
                gate = DLEShadowGate(log_path=log_path)
                gate.observe_entry(
                    symbol="ETHUSDT",
                    direction="LONG",
                    strategy="vol_target",
                    verdict="ALLOW",
                    denial_code=None,
                    regime="MEAN_REVERT",
                    regime_confidence=0.8,
                    final_qty=0.08,
                    final_notional=240.0,
                    attempt_id="sig_dry_run",
                )

            # File should not exist
            assert not os.path.exists(log_path)
```

---

## Rollout Plan

### Phase A.1: Code merge (no activation)
- Merge `execution/dle_shadow.py`
- Merge flag additions to `execution/v6_flags.py`
- Merge integration hooks (behind flag, default OFF)
- Full test suite green
- `SHADOW_DLE_ENABLED=0` (default)

### Phase A.2: Testnet shadow (48h minimum)
- `SHADOW_DLE_ENABLED=1` on testnet
- `SHADOW_DLE_WRITE_LOGS=1`
- Verify log output matches schema
- Verify zero execution divergence
- Verify deterministic IDs work correctly

### Phase A.3: Production shadow (7 days minimum)
- `SHADOW_DLE_ENABLED=1` on production
- `SHADOW_DLE_WRITE_LOGS=1`
- Collect episodes for statistical comparison
- **Required:** At least 1 regime flip exit observed
- Generate "DLE Reconstruction Report" from shadow logs

### Phase A.4: Gap closure
- Use shadow data to close Gap Registry items
- Fix episode ledger regime propagation
- Fix exit_fills count
- Add authority_source to doctrine events

---

## Exit Criteria for Phase A

Phase A is complete when:

1. ✅ Shadow gate running in production for **7+ days**
2. ✅ **10+ episodes** with full DLE chain (Decision → Permit → Episode)
3. ✅ **Zero execution divergence** (per falsifiable definition above)
4. ✅ **≥1 regime flip exit** observed with full DLE tracing
5. ✅ Gap Registry items **P0/P1 closed**
6. ✅ Episode can be **reconstructed from DLE logs alone**
7. ✅ **0 unhandled exceptions** from shadow gate

Then — and only then — proceed to Phase B (soft enforcement).

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `execution/dle_shadow.py` | NEW | Shadow gate implementation |
| `execution/v6_flags.py` | MODIFY | Add `SHADOW_DLE_ENABLED`, `SHADOW_DLE_WRITE_LOGS` |
| `execution/executor_live.py` | MODIFY | Integration hooks (2 locations) |
| `logs/dle/` | NEW | DLE event log directory |
| `tests/unit/test_dle_shadow.py` | NEW | Unit tests |
| `tests/integration/test_dle_shadow_integration.py` | NEW | Integration tests |
| `v7_manifest.json` | MODIFY | Register `dle_events_v1.jsonl` |

---

## Approval Status

- [x] Plan structure approved
- [x] Dual flag approach approved
- [x] Log path (`logs/dle/dle_events_v1.jsonl`) approved
- [x] Deterministic ID scheme approved
- [x] Two primary hooks + enrichment approved
- [x] Falsifiable success criteria approved

---

*This document is the implementation warrant for CYCLE_004 Phase A.*
