"""
Execution alert classifier.

Transforms execution_health snapshots into actionable alert payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List

from execution.utils.execution_health import (
    FALLBACK_WARN_THRESHOLD,
    SLIP_MEDIAN_WARN_BPS,
    DD_WARN_PCT,
    DD_KILL_PCT,
    SHARPE_BAD,
)

ATR_PANIC = "panic"
ATR_HOT = "hot"


def classify_alerts(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert execution_health payload → list of alert messages.
    Alerts are high-level actionable notices for operators.
    """
    symbol = str(snapshot.get("symbol") or "ALL")
    alerts: List[Dict[str, Any]] = []

    router = snapshot.get("router") or {}
    risk = snapshot.get("risk") or {}
    vol = snapshot.get("vol") or {}
    sizing = snapshot.get("sizing") or {}

    fallback_ratio = router.get("fallback_ratio")
    slip_q50 = router.get("slip_q50")
    sharpe = sizing.get("sharpe_7d")
    dd = risk.get("dd_today_pct")
    atr_regime = vol.get("atr_regime")
    toggle_active = risk.get("toggle_active")

    if fallback_ratio is not None and fallback_ratio > FALLBACK_WARN_THRESHOLD:
        alerts.append(
            {
                "symbol": symbol,
                "type": "router_fallback_high",
                "severity": "warning",
                "msg": f"{symbol}: High router fallback ratio {fallback_ratio:.2f}",
            }
        )

    if slip_q50 is not None and slip_q50 > SLIP_MEDIAN_WARN_BPS:
        alerts.append(
            {
                "symbol": symbol,
                "type": "slippage_high",
                "severity": "warning",
                "msg": f"{symbol}: Median slippage {slip_q50:.2f} bps exceeds threshold",
            }
        )

    if dd is not None:
        if dd <= DD_KILL_PCT:
            alerts.append(
                {
                    "symbol": symbol,
                    "type": "dd_kill",
                    "severity": "critical",
                    "msg": f"{symbol}: Daily DD {dd:.2f}% — symbol auto-disabled",
                }
            )
        elif dd <= DD_WARN_PCT:
            alerts.append(
                {
                    "symbol": symbol,
                    "type": "dd_warning",
                    "severity": "warning",
                    "msg": f"{symbol}: Daily DD {dd:.2f}% approaching risk limits",
                }
            )

    if toggle_active:
        alerts.append(
            {
                "symbol": symbol,
                "type": "symbol_disabled",
                "severity": "critical",
                "msg": f"{symbol}: Symbol is currently DISABLED",
            }
        )

    if sharpe is not None and sharpe <= SHARPE_BAD:
        alerts.append(
            {
                "symbol": symbol,
                "type": "sharpe_poor",
                "severity": "warning",
                "msg": f"{symbol}: Poor 7d Sharpe {sharpe:.2f}",
            }
        )

    if atr_regime == ATR_PANIC:
        alerts.append(
            {
                "symbol": symbol,
                "type": "atr_panic",
                "severity": "warning",
                "msg": f"{symbol}: ATR regime PANIC — hot volatility",
            }
        )
    elif atr_regime == ATR_HOT:
        alerts.append(
            {
                "symbol": symbol,
                "type": "atr_hot",
                "severity": "info",
                "msg": f"{symbol}: ATR regime HOT — elevated volatility",
            }
        )

    return alerts


__all__ = ["classify_alerts"]
