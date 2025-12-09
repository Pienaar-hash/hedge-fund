"""
Runtime Health Component — Risk Engine + Router Health panels.

Two matched-height panels showing system health metrics.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import streamlit as st


def _status_class(status: str) -> str:
    """Return CSS class for status badges."""
    s = str(status).lower() if status else "normal"
    if s in ("normal", "ok", "low", "healthy", "good", "excellent"):
        return "normal"
    elif s in ("elevated", "medium", "fair", "warning"):
        return "warning"
    return "critical"


def _format_age(age_seconds: Optional[float]) -> str:
    """Format age in human-readable form."""
    if age_seconds is None:
        return "n/a"
    try:
        age = float(age_seconds)
    except Exception:
        return "n/a"
    if age < 60:
        return f"{age:.0f}s"
    if age < 3600:
        return f"{age/60:.1f}m"
    return f"{age/3600:.1f}h"


def _row(label: str, value: str, badge_class: str) -> str:
    """Build a table row with status badge."""
    return f'''
        <tr>
            <td>{label}</td>
            <td style="text-align:right;">
                <span class="status-badge {badge_class}">{value}</span>
            </td>
        </tr>
    '''


def _metric_row(label: str, value: str) -> str:
    """Build a simple metric row."""
    return f'''
        <tr>
            <td>{label}</td>
            <td style="text-align:right;">{value}</td>
        </tr>
    '''


def build_risk_state(risk_snapshot: Dict[str, Any], nav_value: float, gross_exposure: float) -> Dict[str, Any]:
    """Build normalized risk state for rendering."""
    dd_state = risk_snapshot.get("dd_state", "normal")
    atr_regime = risk_snapshot.get("atr_regime", "normal")
    risk_mode = risk_snapshot.get("risk_mode", "normal")
    veto_counts = risk_snapshot.get("veto_counts", {})
    total_vetoes = sum(veto_counts.values()) if isinstance(veto_counts, dict) else 0
    
    # Handle nested dicts
    if isinstance(dd_state, dict):
        dd_state = dd_state.get("dd_state") or dd_state.get("state") or "normal"
    if isinstance(atr_regime, dict):
        atr_regime = atr_regime.get("atr_regime") or atr_regime.get("regime") or "normal"
    if isinstance(risk_mode, dict):
        risk_mode = risk_mode.get("risk_mode") or risk_mode.get("mode") or "normal"
    
    exposure_pct = (gross_exposure / nav_value * 100) if nav_value > 0 else 0
    
    return {
        "dd_state": str(dd_state),
        "atr_regime": str(atr_regime),
        "risk_mode": str(risk_mode),
        "total_vetoes": total_vetoes,
        "gross_exposure": gross_exposure,
        "exposure_pct": exposure_pct,
    }


def build_router_state(router_health: Dict[str, Any]) -> Dict[str, Any]:
    """Build normalized router state for rendering."""
    rh = router_health.get("router_health", router_health)
    quality = rh.get("quality") or rh.get("policy_quality") or "ok"
    maker_rate = float(rh.get("maker_fill_rate") or rh.get("maker_rate") or 0)
    fallback_ratio = float(rh.get("fallback_ratio") or 0)
    latency_ms = float(rh.get("avg_latency_ms") or rh.get("latency_ms") or 0)
    updated_ts = rh.get("updated_ts") or router_health.get("updated_ts")
    
    age_s = None
    if updated_ts:
        try:
            age_s = time.time() - float(updated_ts)
        except Exception:
            pass
    
    return {
        "quality": str(quality),
        "maker_rate": maker_rate,
        "fallback_ratio": fallback_ratio,
        "latency_ms": latency_ms,
        "age_s": age_s,
    }


def render_risk_engine_panel(risk_state: Dict[str, Any]) -> None:
    """Render risk engine panel as single HTML block."""
    dd_state = risk_state["dd_state"]
    atr_regime = risk_state["atr_regime"]
    risk_mode = risk_state["risk_mode"]
    total_vetoes = risk_state["total_vetoes"]
    gross_exposure = risk_state["gross_exposure"]
    exposure_pct = risk_state["exposure_pct"]
    
    rows = [
        _row("Drawdown State", dd_state.upper(), _status_class(dd_state)),
        _row("ATR Regime", atr_regime.upper(), _status_class(atr_regime)),
        _row("Risk Mode", risk_mode.upper(), _status_class(risk_mode)),
    ]
    
    html = f'''
    <div class="panel panel-equal">
        <div class="panel-header">
            <span class="panel-title">Risk Engine</span>
        </div>
        <table class="quant-table">
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        <div class="panel-footer">
            <div class="panel-footer-row">
                <span class="text-muted">Gross Exposure</span>
                <span>${gross_exposure:,.0f} <span class="text-muted">({exposure_pct:.0f}%)</span></span>
            </div>
            <div class="panel-footer-row">
                <span class="text-muted">Vetoes (24h)</span>
                <span>{total_vetoes}</span>
            </div>
        </div>
    </div>
    '''
    
    st.html(html)


def render_router_panel(router_state: Dict[str, Any]) -> None:
    """Render router health panel as single HTML block."""
    quality = router_state["quality"]
    maker_rate = router_state["maker_rate"]
    fallback_ratio = router_state["fallback_ratio"]
    latency_ms = router_state["latency_ms"]
    age_s = router_state["age_s"]
    
    # Determine badge classes
    maker_class = "normal" if maker_rate >= 0.8 else "warning" if maker_rate >= 0.5 else "critical"
    fallback_class = "normal" if fallback_ratio < 0.1 else "warning" if fallback_ratio < 0.3 else "critical"
    quality_class = _status_class(quality)
    
    html = f'''
    <div class="panel panel-equal">
        <div class="panel-header">
            <span class="panel-title">Router Health</span>
        </div>
        <table class="quant-table">
            <tbody>
                <tr>
                    <td>Maker Fill Rate</td>
                    <td style="text-align:right;">
                        <span class="status-badge {maker_class}">{maker_rate:.0%}</span>
                    </td>
                </tr>
                <tr>
                    <td>Fallback Ratio</td>
                    <td style="text-align:right;">
                        <span class="status-badge {fallback_class}">{fallback_ratio:.1%}</span>
                    </td>
                </tr>
                <tr>
                    <td>Avg Latency</td>
                    <td style="text-align:right;">{latency_ms:.0f}ms</td>
                </tr>
            </tbody>
        </table>
        <div class="panel-footer">
            <div class="panel-footer-row">
                <span class="text-muted">Policy Quality</span>
                <span class="status-badge {quality_class}">{quality.upper()}</span>
            </div>
            <div class="panel-footer-row">
                <span class="text-muted">Last Update</span>
                <span>{_format_age(age_s)} ago</span>
            </div>
        </div>
    </div>
    '''
    
    st.html(html)


def render_runtime_health_block(
    risk_snapshot: Dict[str, Any],
    router_health: Dict[str, Any],
    nav_value: float,
    gross_exposure: float,
) -> None:
    """Render the complete runtime health section with matched-height panels."""
    # Section header with gear icon
    st.html('''
    <div class="section-header">
        <span class="section-header-emoji">⚙️</span>
        <h2>Trading Engine</h2>
    </div>
    ''')
    
    # Build state
    risk_state = build_risk_state(risk_snapshot, nav_value, gross_exposure)
    router_state = build_router_state(router_health)
    
    # Two-column layout
    col_risk, col_router = st.columns([1, 1])
    
    with col_risk:
        render_risk_engine_panel(risk_state)
    
    with col_router:
        render_router_panel(router_state)
