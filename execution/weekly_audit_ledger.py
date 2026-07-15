from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _validate_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float in decision payload")
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite(item)


def canonicalize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(decision))
    _validate_finite(payload)
    if payload.get("decision_hash"):
        payload.pop("decision_hash")
    return payload


def hash_decision(decision_without_hash: dict[str, Any]) -> str:
    encoded = json.dumps(
        decision_without_hash,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def append_decision_record(path: str | Path, decision: dict[str, Any]) -> dict[str, Any]:
    canonical = canonicalize_decision(decision)
    canonical["decision_hash"] = hash_decision(canonical)
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False))
        handle.write("\n")
    return canonical


def verify_decision_chain(path: str | Path) -> tuple[bool, list[str]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return (True, [])
    seen_cycles: set[str] = set()
    errors: list[str] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            decision_hash = payload.get("decision_hash")
            canonical = canonicalize_decision(payload)
            expected = hash_decision(canonical)
            if decision_hash != expected:
                errors.append(f"line {line_number}: decision hash mismatch")
            cycle_id = str(payload.get("cycle_id", ""))
            if cycle_id in seen_cycles:
                errors.append(f"line {line_number}: duplicate cycle_id {cycle_id}")
            seen_cycles.add(cycle_id)
    return (not errors, errors)
