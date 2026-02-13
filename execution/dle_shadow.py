# execution/dle_shadow.py
"""
DLE Shadow Gate (Phase A)
- Logs DLE-shaped events (Decision, Request, Permit, EpisodeLink) WITHOUT affecting execution.
- Deterministic IDs (no random UUIDs).
- Dual flags:
  - SHADOW_DLE_ENABLED: master on/off
  - SHADOW_DLE_WRITE_LOGS: allow dry-run (construct events but do not write)
- Fail-open: logging failures never crash execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = "dle_shadow_v1"
SCHEMA_VERSION_V2 = "dle_shadow_v2"
# Canonical path — MUST match v7_manifest.json → dle_shadow_events.path
DEFAULT_LOG_PATH = "logs/execution/dle_shadow_events.jsonl"

# Manifest-declared path (used by startup invariant)
MANIFEST_LOG_PATH = "logs/execution/dle_shadow_events.jsonl"


# -----------------------
# Helpers (determinism)
# -----------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(obj: Any) -> str:
    """Stable JSON for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def derive_request_id(*, attempt_id: Optional[str], request_payload: Dict[str, Any]) -> str:
    """
    Prefer existing attempt_id if it is stable; else derive deterministically.
    """
    if attempt_id:
        return str(attempt_id)
    return "REQ_" + sha256_hex(_stable_json(request_payload))[:16]


def derive_decision_id(
    *, phase_id: str, action_class: str, constraints: Dict[str, Any], policy_version: str
) -> str:
    payload = {
        "phase_id": phase_id,
        "action_class": action_class,
        "constraints": constraints,
        "policy_version": policy_version,
    }
    return "DEC_" + sha256_hex(_stable_json(payload))[:16]


def derive_permit_id(*, decision_id: str, request_id: str, issued_at_iso: str) -> str:
    payload = {"decision_id": decision_id, "request_id": request_id, "issued_at": issued_at_iso}
    return "PERM_" + sha256_hex(_stable_json(payload))[:16]


def hash_snapshot(name: str, obj: Any) -> str:
    """
    Always return a hash, even if obj is missing.
    """
    if obj is None:
        return f"{name}_MISSING_" + sha256_hex("MISSING")[:16]
    return f"{name}_" + sha256_hex(_stable_json(obj))[:16]


# -----------------------
# Data Objects (shadow)
# -----------------------

@dataclass(frozen=True)
class DLEExecutionRequest:
    request_id: str
    ts: str
    requested_action: str
    symbol: str
    side: str
    strategy: str
    qty_intent: Optional[float]
    context: Dict[str, Any]


@dataclass(frozen=True)
class DLEDecision:
    decision_id: str
    ts: str
    authority_source: str  # DOCTRINE | MANUAL_OVERRIDE | ESCALATION_RESOLUTION
    phase_id: str
    action_class: str
    policy_version: str
    scope: Dict[str, Any]
    constraints: Dict[str, Any]
    risk: Dict[str, Any]


@dataclass(frozen=True)
class DLEPermit:
    permit_id: str
    ts: str
    decision_id: str
    request_id: str
    single_use: bool
    snapshots: Dict[str, str]  # hashes


@dataclass(frozen=True)
class DLEShadowEvent:
    schema_version: str
    event_type: str  # REQUEST | DECISION | PERMIT | LINK
    ts: str
    payload: Dict[str, Any]


# -----------------------
# Writer (fail-open)
# -----------------------

class DLEShadowWriter:
    def __init__(self, log_path: str = DEFAULT_LOG_PATH) -> None:
        self.log_path = log_path
        self._write_failures = 0

    def write(self, event: DLEShadowEvent) -> None:
        """
        Append event to DLE log.
        
        FAIL-OPEN: If write fails, warn to stderr and continue.
        Never raise, never crash executor.
        """
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            line = _stable_json(asdict(event))
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            self._write_failures += 1
            print(
                f"[DLE_SHADOW] Warning: log write failed ({self._write_failures} total): {e}",
                file=sys.stderr,
            )

    @property
    def write_failure_count(self) -> int:
        return self._write_failures


# -----------------------
# Public API
# -----------------------

def shadow_build_chain(
    *,
    enabled: bool,
    write_logs: bool,
    writer: Optional[DLEShadowWriter],
    # Request inputs
    attempt_id: Optional[str],
    requested_action: str,
    symbol: str,
    side: str,
    strategy: str,
    qty_intent: Optional[float],
    context: Dict[str, Any],
    # Decision inputs
    phase_id: str,
    action_class: str,
    policy_version: str,
    scope: Dict[str, Any],
    constraints: Dict[str, Any],
    risk: Dict[str, Any],
    authority_source: str = "DOCTRINE",
    # B.2 enrichment (optional — backward compatible)
    verdict: Optional[str] = None,           # PERMIT | DENY
    deny_reason: Optional[str] = None,       # canonical doctrine veto code
    doctrine_verdict: Optional[str] = None,  # raw DoctrineVerdict enum value
    context_snapshot: Optional[Dict[str, Any]] = None,  # regime, nav_usd, hashes
    provenance: Optional[Dict[str, Any]] = None,        # engine_version, git_sha, docs_version
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Build and optionally log a complete DLE chain (Request → Decision → Permit → Link).
    
    Returns (request_id, decision_id, permit_id).
    Shadow only: never throws, never blocks.
    
    INVARIANTS:
    - Never modifies execution flow
    - Never prevents orders
    - Fail-open on errors (stderr warning, continue)
    - Deterministic IDs for replay (except permit_id which includes timestamp)
    
    B.2 enrichment fields (when provided):
    - verdict/deny_reason/doctrine_verdict: canonical shadow semantics
    - context_snapshot: regime + NAV + position/score hashes
    - provenance: engine_version + git_sha + docs_version
    When any enrichment field is present, DECISION event uses schema_version v2.
    """
    if not enabled:
        return (None, None, None)

    ts = _utc_now_iso()

    # REQUEST
    req_payload = {
        "requested_action": requested_action,
        "symbol": symbol,
        "side": side,
        "strategy": strategy,
        "qty_intent": qty_intent,
        "context": context,
    }
    request_id = derive_request_id(attempt_id=attempt_id, request_payload=req_payload)

    req = DLEExecutionRequest(
        request_id=request_id,
        ts=ts,
        requested_action=requested_action,
        symbol=symbol,
        side=side,
        strategy=strategy,
        qty_intent=qty_intent,
        context=context,
    )

    # DECISION (deterministic)
    decision_id = derive_decision_id(
        phase_id=phase_id,
        action_class=action_class,
        constraints=constraints,
        policy_version=policy_version,
    )

    dec = DLEDecision(
        decision_id=decision_id,
        ts=ts,
        authority_source=authority_source,
        phase_id=phase_id,
        action_class=action_class,
        policy_version=policy_version,
        scope=scope,
        constraints=constraints,
        risk=risk,
    )

    # PERMIT
    permit_id = derive_permit_id(decision_id=decision_id, request_id=request_id, issued_at_iso=ts)

    snapshots = {
        "positions_state_hash": context.get("positions_state_hash", hash_snapshot("positions_state", None)),
        "regime_state_hash": context.get("regime_state_hash", hash_snapshot("regime_state", None)),
        "config_hash": context.get("config_hash", hash_snapshot("config", None)),
    }

    perm = DLEPermit(
        permit_id=permit_id,
        ts=ts,
        decision_id=decision_id,
        request_id=request_id,
        single_use=True,
        snapshots=snapshots,
    )

    if not write_logs:
        return (request_id, decision_id, permit_id)

    # Build enriched DECISION payload (B.2)
    decision_payload = asdict(dec)
    _has_enrichment = any(x is not None for x in (verdict, deny_reason, doctrine_verdict, context_snapshot, provenance))
    if _has_enrichment:
        if verdict is not None:
            decision_payload["verdict"] = verdict
        if deny_reason is not None:
            decision_payload["deny_reason"] = deny_reason
        if doctrine_verdict is not None:
            decision_payload["doctrine_verdict"] = doctrine_verdict
        if context_snapshot is not None:
            decision_payload["context_snapshot"] = context_snapshot
        if provenance is not None:
            decision_payload["provenance"] = provenance
    _decision_schema = SCHEMA_VERSION_V2 if _has_enrichment else SCHEMA_VERSION

    # Write events (fail-open)
    try:
        w = writer or DLEShadowWriter()
        w.write(DLEShadowEvent(
            schema_version=SCHEMA_VERSION,
            event_type="REQUEST",
            ts=ts,
            payload=asdict(req),
        ))
        w.write(DLEShadowEvent(
            schema_version=_decision_schema,
            event_type="DECISION",
            ts=ts,
            payload=decision_payload,
        ))
        w.write(DLEShadowEvent(
            schema_version=SCHEMA_VERSION,
            event_type="PERMIT",
            ts=ts,
            payload=asdict(perm),
        ))
        # LINK event allows easy episode binding later (even in PRE_DLE)
        w.write(DLEShadowEvent(
            schema_version=SCHEMA_VERSION,
            event_type="LINK",
            ts=ts,
            payload={
                "request_id": request_id,
                "decision_id": decision_id,
                "permit_id": permit_id,
                "symbol": symbol,
                "requested_action": requested_action,
                "strategy": strategy,
            },
        ))
    except Exception:
        # Shadow must never affect execution
        pass

    return (request_id, decision_id, permit_id)


# -----------------------
# Global instance (lazy init)
# -----------------------

_shadow_writer: Optional[DLEShadowWriter] = None


def get_shadow_writer() -> DLEShadowWriter:
    """Get or create global shadow writer instance."""
    global _shadow_writer
    if _shadow_writer is None:
        _shadow_writer = DLEShadowWriter()
    return _shadow_writer


def reset_shadow_writer() -> None:
    """Reset global instance (for testing only)."""
    global _shadow_writer
    _shadow_writer = None


# -----------------------
# Path invariant (B.2c)
# -----------------------

def verify_shadow_log_path() -> None:
    """
    Startup invariant: assert DEFAULT_LOG_PATH == MANIFEST_LOG_PATH.

    Raises ValueError if the writer would emit to a non-manifest path.
    Called once during executor startup — fail-loud (not fail-open).
    """
    if DEFAULT_LOG_PATH != MANIFEST_LOG_PATH:
        raise ValueError(
            f"DLE shadow log path mismatch: writer={DEFAULT_LOG_PATH!r} "
            f"manifest={MANIFEST_LOG_PATH!r}"
        )
