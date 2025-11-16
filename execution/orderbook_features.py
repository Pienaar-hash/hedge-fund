"""Lightweight orderbook gate compatibility shim.

The original v5.x orderbook gate was retired, but newer screener code still
expects a `(veto, info)` tuple describing whether adverse depth should block an
entry.  This module exposes a deterministic stub that can be safely patched by
tests (see ``tests/test_orderbook_features.py``) and always returns structured
metadata so downstream probes never crash when unpacking gate results.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

_GATE_NAME = "orderbook"


def topn_imbalance(symbol: str, limit: int = 20) -> float:
    """Return a dummy top-N orderbook imbalance metric.

    The metric is defined as (bid_volume - ask_volume)/(bid+ask).  The live
    production implementation can monkeypatch this helper to source actual
    depth data, while tests override it to exercise veto paths.
    """

    # Without a real orderbook feed, stay neutral.
    return 0.0


def evaluate_entry_gate(
    symbol: str,
    side: str,
    *,
    enabled: bool = True,
    limit: int = 20,
) -> Tuple[bool, Dict[str, Any]]:
    """Return (veto, info) for the orderbook gate.

    When a veto occurs the tuple is ``(True, {"reason": "...", "gate": ...})``;
    otherwise ``(False, {"ok": True, "gate": ...})``.  The info payload always
    includes the evaluated metric so telemetry can expose alignment/boost data.
    """

    info: Dict[str, Any] = {
        "gate": _GATE_NAME,
        "symbol": str(symbol).upper(),
    }
    if not enabled:
        info["ok"] = True
        info["reason"] = "disabled"
        return False, info

    try:
        metric = float(topn_imbalance(symbol, limit=limit) or 0.0)
    except Exception as exc:  # pragma: no cover - defensive
        metric = 0.0
        info["error"] = str(exc)
    info["metric"] = metric

    veto = False
    reason = None
    side_norm = str(side).upper()
    if side_norm in {"BUY", "LONG"} and metric < -0.10:
        veto = True
        reason = "ob_adverse"
    elif side_norm in {"SELL", "SHORT"} and metric > 0.10:
        veto = True
        reason = "ob_adverse"
    else:
        aligned = (
            (side_norm in {"BUY", "LONG"} and metric >= 0.20)
            or (side_norm in {"SELL", "SHORT"} and metric <= -0.20)
        )
        if aligned:
            info["flag"] = "ob_aligned"
    if veto:
        info["reason"] = reason or "ob_adverse"
        info["ok"] = False
        return True, info
    info["ok"] = True
    return False, info


__all__ = ["evaluate_entry_gate", "topn_imbalance"]
