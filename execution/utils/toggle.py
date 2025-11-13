"""Symbol toggle helpers for risk gates."""

from __future__ import annotations

import time
from typing import Dict, Optional

from execution.firestore_utils import publish_symbol_toggle, fetch_symbol_toggles


_SYMBOL_TOGGLES: Dict[str, Dict[str, float | str]] = {}
_BOOTSTRAPPED = False


def bootstrap_from_firestore() -> None:
    """Populate local cache from Firestore once per process."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    try:
        docs = fetch_symbol_toggles()
    except Exception:
        return

    now = time.time()
    for doc in docs:
        sym = str(doc.get("symbol") or "").upper()
        if not sym:
            continue
        until = float(doc.get("until", 0.0) or 0.0)
        if until and until < now:
            continue
        reason = doc.get("reason", "") or ""
        _SYMBOL_TOGGLES[sym] = {"until": until, "reason": reason}
    _BOOTSTRAPPED = True


def disable_symbol_temporarily(symbol: str, ttl_hours: float, reason: str = "") -> None:
    """Disable `symbol` for `ttl_hours`, recording the reason."""
    until = time.time() + max(ttl_hours, 0.0) * 3600
    symbol_key = symbol.upper()
    meta = {"until": until, "reason": reason}
    _SYMBOL_TOGGLES[symbol_key] = meta
    try:
        publish_symbol_toggle(symbol_key, dict(meta))
    except Exception:
        pass


def is_symbol_disabled(symbol: str) -> bool:
    """Return True if `symbol` is currently disabled."""
    bootstrap_from_firestore()
    symbol_key = symbol.upper()
    meta = _SYMBOL_TOGGLES.get(symbol_key)
    if not meta:
        return False
    if time.time() >= float(meta.get("until", 0.0)):
        _SYMBOL_TOGGLES.pop(symbol_key, None)
        return False
    return True


def get_symbol_disable_meta(symbol: str) -> Optional[Dict[str, float | str]]:
    """Return disable metadata if the symbol is currently disabled."""
    bootstrap_from_firestore()
    meta = _SYMBOL_TOGGLES.get(symbol.upper())
    if not meta:
        return None
    if time.time() >= float(meta.get("until", 0.0)):
        _SYMBOL_TOGGLES.pop(symbol.upper(), None)
        return None
    return dict(meta)
