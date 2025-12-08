from __future__ import annotations

import json
from collections.abc import Mapping as AbcMapping, MutableMapping as AbcMutableMapping
from pathlib import Path
from typing import Any, Mapping, MutableMapping

DEFAULT_ATR_SHRINK: Mapping[int, float] = {0: 1.0, 1: 1.0, 2: 0.7, 3: 0.5}
DEFAULT_DD_SHRINK: Mapping[int, float] = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.0}
DEFAULT_RISK_SHRINK: Mapping[str, float] = {
    "OK": 1.0,
    "WARN": 0.8,
    "DEFENSIVE": 0.5,
    "HALTED": 0.0,
}

STATE_DIR = Path("logs") / "state"
REGIMES_FILE = STATE_DIR / "regimes.json"
RISK_FILE = STATE_DIR / "risk_snapshot.json"


def _load_state_file(path: Path | str | None, default: Path) -> Mapping[str, Any]:
    target = Path(path) if path else default
    try:
        text = target.read_text(encoding="utf-8")
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp_factor(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_risk_mode(value: Any) -> str:
    if value is None:
        return "OK"
    text = str(value).strip().upper()
    return text or "OK"


def load_regime_snapshot(path: Path | str | None = None) -> Mapping[str, Any]:
    raw = dict(_load_state_file(path, REGIMES_FILE))
    raw.setdefault("atr_regime", _safe_int(raw.get("atr_regime"), 0))
    raw.setdefault("dd_regime", _safe_int(raw.get("dd_regime"), 0))
    return raw


def load_risk_snapshot(path: Path | str | None = None) -> Mapping[str, Any]:
    raw = dict(_load_state_file(path, RISK_FILE))
    raw["risk_mode"] = _normalize_risk_mode(raw.get("risk_mode"))
    return raw


def _merge_int_map(
    defaults: Mapping[int, float],
    overrides: Any,
) -> dict[int, float]:
    merged: dict[int, float] = {int(k): _clamp_factor(float(v)) for k, v in defaults.items()}
    if not isinstance(overrides, AbcMapping):
        return merged
    for raw_key, raw_value in overrides.items():
        try:
            normalized_key = int(raw_key)
        except Exception:
            continue
        try:
            normalized_value = float(raw_value)
        except Exception:
            continue
        merged[normalized_key] = _clamp_factor(normalized_value)
    return merged


def _merge_risk_map(
    defaults: Mapping[str, float],
    overrides: Any,
) -> dict[str, float]:
    merged: dict[str, float] = {str(k).upper(): _clamp_factor(float(v)) for k, v in defaults.items()}
    if not isinstance(overrides, AbcMapping):
        return merged
    for raw_key, raw_value in overrides.items():
        normalized_key = _normalize_risk_mode(raw_key)
        if not normalized_key:
            continue
        try:
            normalized_value = float(raw_value)
        except Exception:
            continue
        merged[normalized_key] = _clamp_factor(normalized_value)
    return merged


def adaptive_factor(
    atr_regime: Any,
    dd_regime: Any,
    risk_mode: Any,
    overrides: Mapping[str, Any] | None = None,
) -> float:
    override_map = overrides if isinstance(overrides, AbcMapping) else {}
    if override_map.get("adaptive_enabled") is False:
        return 1.0
    atr_map = _merge_int_map(DEFAULT_ATR_SHRINK, override_map.get("atr_shrink_map"))
    dd_map = _merge_int_map(DEFAULT_DD_SHRINK, override_map.get("dd_shrink_map"))
    risk_map = _merge_risk_map(DEFAULT_RISK_SHRINK, override_map.get("risk_mode_map"))

    atr_idx = _safe_int(atr_regime, 0)
    dd_idx = _safe_int(dd_regime, 0)
    risk_text = _normalize_risk_mode(risk_mode)

    atr_factor = atr_map.get(atr_idx, 1.0)
    dd_factor = dd_map.get(dd_idx, 1.0)
    risk_factor = risk_map.get(risk_text, 1.0)
    return _clamp_factor(atr_factor * dd_factor * risk_factor)


def adaptive_sizing(
    symbol: str,
    gross_usd: float,
    atr_regime: Any,
    dd_regime: Any,
    risk_mode: Any,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    factor = adaptive_factor(atr_regime, dd_regime, risk_mode, overrides)
    adjusted = max(0.0, float(gross_usd or 0.0) * factor)
    return adjusted, factor


def strategy_enablement(
    strategy_id: str,
    atr_regime: Any,
    dd_regime: Any,
    risk_mode: Any,
    overrides: Mapping[str, Any] | None = None,
) -> bool:
    _ = strategy_id, overrides  # reserved for future overrides
    if _normalize_risk_mode(risk_mode) == "HALTED":
        return False
    if _safe_int(dd_regime, 0) >= 3:
        return False
    return True


def attach_adaptive_metadata(
    intent: MutableMapping[str, Any],
    atr_regime: Any,
    dd_regime: Any,
    risk_mode: Any,
    final_factor: float,
) -> Mapping[str, Any] | None:
    if not isinstance(intent, AbcMutableMapping):
        return None
    metadata = intent.get("metadata")
    if not isinstance(metadata, AbcMutableMapping):
        metadata = {}
    adaptive_block = {
        "atr_regime": _safe_int(atr_regime, 0),
        "dd_regime": _safe_int(dd_regime, 0),
        "risk_mode": _normalize_risk_mode(risk_mode),
        "final_factor": _clamp_factor(final_factor),
    }
    metadata["adaptive"] = adaptive_block
    intent["metadata"] = metadata
    return adaptive_block
