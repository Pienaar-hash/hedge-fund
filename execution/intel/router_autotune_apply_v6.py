"""Router auto-tune application helpers (safely clamps suggestions)."""

from __future__ import annotations

import json
import math
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from execution.v6_flags import get_flags

LOG = logging.getLogger(__name__)

def _env_flag(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


APPLY_ENABLED = get_flags().router_autotune_v6_apply_enabled
def _parse_allowlist(raw: str) -> set[str]:
    if not raw:
        return set()
    try:
        maybe = json.loads(raw)
        if isinstance(maybe, list):
            return {str(item).strip().upper() for item in maybe if str(item).strip()}
    except Exception:
        pass
    return {sym.strip().upper() for sym in raw.split(",") if sym.strip()}


ALLOWLIST_RAW = os.getenv("ROUTER_AUTOTUNE_V6_SYMBOL_ALLOWLIST", "")
SYMBOL_ALLOWLIST = _parse_allowlist(ALLOWLIST_RAW)
MAX_BIAS_DELTA = _env_float("ROUTER_AUTOTUNE_V6_MAX_BIAS_DELTA", 0.05)
MAX_OFFSET_STEP_BPS = _env_float("ROUTER_AUTOTUNE_V6_MAX_OFFSET_STEP_BPS", 2.0)
MAX_OFFSET_ABS_BPS = _env_float("ROUTER_AUTOTUNE_V6_MAX_OFFSET_ABS_BPS", 10.0)
ALLOW_MAKER_FLIP = _env_flag("ROUTER_AUTOTUNE_V6_ALLOW_FLIP", "0")

QUAL_ALLOW = os.getenv("ROUTER_AUTOTUNE_V6_REQUIRE_QUALITY", "good,ok")
maybe_quality = _parse_allowlist(QUAL_ALLOW)
REQUIRE_QUALITY = {entry.lower() for entry in maybe_quality} if maybe_quality else {"good", "ok"}

SUGGESTIONS_PATH = Path(os.getenv("ROUTER_AUTOTUNE_V6_SUGGESTIONS_PATH") or "logs/state/router_policy_suggestions_v6.json")
RISK_ALLOC_PATH = Path(os.getenv("ROUTER_AUTOTUNE_V6_RISK_STATE_PATH") or "logs/state/risk_allocation_suggestions_v6.json")
RISK_STATE_MAX_AGE_S = _env_float("ROUTER_AUTOTUNE_V6_RISK_STATE_MAX_AGE_S", 1800.0)
_DEFAULT_RISK_MODE = "cautious"

_SUGGESTIONS_CACHE: Dict[str, Any] = {"mtime": None, "data": {}}
_RISK_MODE_CACHE: Dict[str, Any] = {}


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _coerce_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except Exception:
            return None
    return None


def get_current_risk_mode(now: Optional[float] = None) -> str:
    now_ts = float(now if now is not None else time.time())
    mtime = RISK_ALLOC_PATH.stat().st_mtime if RISK_ALLOC_PATH.exists() else None
    payload = _load_json(RISK_ALLOC_PATH)
    if not isinstance(payload, Mapping):
        event = {"event": "router_apply_no_allocator_state", "path": str(RISK_ALLOC_PATH)}
        if RISK_ALLOC_PATH.exists():
            event["event"] = "router_apply_bad_allocator_state"
        LOG.warning("%s", event)
        _RISK_MODE_CACHE["mtime"] = mtime
        _RISK_MODE_CACHE["mode"] = _DEFAULT_RISK_MODE
        return _DEFAULT_RISK_MODE

    risk_mode = (
        str((payload.get("global") or {}).get("risk_mode") or payload.get("risk_mode") or _DEFAULT_RISK_MODE)
        .strip()
        .lower()
    )
    ts = _coerce_timestamp(payload.get("updated_ts") or payload.get("generated_ts"))
    age = None
    if ts is not None:
        age = max(0.0, now_ts - ts)
    elif mtime is not None:
        age = max(0.0, now_ts - mtime)
    is_stale = age is not None and age > RISK_STATE_MAX_AGE_S
    if is_stale:
        LOG.warning(
            "%s",
            {
                "event": "router_apply_stale_allocator_state",
                "path": str(RISK_ALLOC_PATH),
                "age_s": round(age or 0.0, 6),
                "risk_mode": risk_mode,
            },
        )
        if risk_mode == "defensive":
            risk_mode = _DEFAULT_RISK_MODE
    risk_mode = risk_mode or _DEFAULT_RISK_MODE
    _RISK_MODE_CACHE["mtime"] = mtime
    _RISK_MODE_CACHE["mode"] = risk_mode
    return risk_mode


def _load_router_suggestions() -> Dict[str, Mapping[str, Any]]:
    mtime = SUGGESTIONS_PATH.stat().st_mtime if SUGGESTIONS_PATH.exists() else None
    cache_mtime = _SUGGESTIONS_CACHE.get("mtime")
    if mtime and cache_mtime == mtime:
        return _SUGGESTIONS_CACHE.get("data") or {}
    payload = _load_json(SUGGESTIONS_PATH) or {}
    data: Dict[str, Mapping[str, Any]] = {}
    for entry in payload.get("symbols", []):
        if not isinstance(entry, Mapping):
            continue
        sym = str(entry.get("symbol") or "").upper()
        if sym:
            data[sym] = entry
    _SUGGESTIONS_CACHE["mtime"] = mtime
    _SUGGESTIONS_CACHE["data"] = data
    return data


def get_symbol_suggestion(symbol: str) -> Optional[Mapping[str, Any]]:
    if not symbol:
        return None
    return _load_router_suggestions().get(symbol.upper())


def _bias_to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").lower()
    if text in {"prefer_maker", "maker"}:
        return 0.25
    if text in {"prefer_taker", "taker"}:
        return 0.75
    return 0.5


def _float_to_bias(value: float) -> str:
    if value <= 0.4:
        return "prefer_maker"
    if value >= 0.6:
        return "prefer_taker"
    return "balanced"


def apply_router_suggestion(
    current_policy: Mapping[str, Any],
    *,
    suggestion: Optional[Mapping[str, Any]],
    symbol: str,
    risk_mode: str,
    current_offset_bps: Optional[float],
) -> Tuple[Mapping[str, Any], bool, float]:
    policy_dict = dict(current_policy)
    offset = float(current_offset_bps or policy_dict.get("offset_bps") or 0.0)
    applied = False
    if not APPLY_ENABLED or not suggestion:
        return policy_dict, False, offset
    sym = str(symbol or "").upper()
    if SYMBOL_ALLOWLIST and sym not in SYMBOL_ALLOWLIST:
        return policy_dict, False, offset
    if str(risk_mode or "").lower() == "defensive":
        return policy_dict, False, offset
    quality = str(policy_dict.get("quality") or "").lower()
    if REQUIRE_QUALITY and quality and quality not in REQUIRE_QUALITY:
        return policy_dict, False, offset
    proposed = suggestion.get("proposed_policy") if isinstance(suggestion.get("proposed_policy"), Mapping) else {}
    if not proposed:
        return policy_dict, False, offset
    current_bias_val = _bias_to_float(policy_dict.get("taker_bias"))
    proposed_bias_val = _bias_to_float(proposed.get("taker_bias"))
    delta_bias = proposed_bias_val - current_bias_val
    if abs(delta_bias) > MAX_BIAS_DELTA:
        delta_bias = math.copysign(MAX_BIAS_DELTA, delta_bias)
    new_bias_val = min(1.0, max(0.0, current_bias_val + delta_bias))
    bias_changed = abs(new_bias_val - current_bias_val) > 1e-9

    proposed_offset = proposed.get("offset_bps")
    offset_changed = False
    new_offset = offset
    if proposed_offset is not None:
        try:
            proposed_offset = float(proposed_offset)
        except Exception:
            proposed_offset = offset
        delta_offset = proposed_offset - offset
        if abs(delta_offset) > MAX_OFFSET_STEP_BPS:
            delta_offset = math.copysign(MAX_OFFSET_STEP_BPS, delta_offset)
        new_offset = offset + delta_offset
        new_offset = max(-MAX_OFFSET_ABS_BPS, min(MAX_OFFSET_ABS_BPS, new_offset))
        offset_changed = abs(new_offset - offset) > 1e-6

    maker_first = bool(policy_dict.get("maker_first"))
    proposed_maker_first = proposed.get("maker_first")
    maker_changed = False
    if (
        isinstance(proposed_maker_first, bool)
        and proposed_maker_first != maker_first
        and ALLOW_MAKER_FLIP
        and str(risk_mode or "").lower() == "normal"
    ):
        maker_first = proposed_maker_first
        maker_changed = True

    if not (bias_changed or offset_changed or maker_changed):
        return policy_dict, False, offset

    policy_dict["taker_bias"] = _float_to_bias(new_bias_val)
    policy_dict["maker_first"] = maker_first
    policy_dict["offset_bps"] = new_offset
    applied = True
    return policy_dict, applied, new_offset


__all__ = [
    "APPLY_ENABLED",
    "SYMBOL_ALLOWLIST",
    "get_symbol_suggestion",
    "apply_router_suggestion",
    "get_current_risk_mode",
]
