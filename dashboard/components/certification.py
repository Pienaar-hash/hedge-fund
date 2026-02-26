"""
System Certification Panel — Activation Window v8.0 Governance Surface.

Displays Activation Window status, 7-gate verification result,
production scale eligibility, and structural integrity metrics.

Data sources:
  - logs/state/activation_window_state.json
  - logs/state/activation_verification_verdict.json

This is a governance-level surface — constitutional, not telemetry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# State Loaders
# ---------------------------------------------------------------------------

_AW_STATE_PATH = Path("logs/state/activation_window_state.json")
_AV_VERDICT_PATH = Path("logs/state/activation_verification_verdict.json")


def load_activation_window_state(
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load activation window per-loop state."""
    p = path or _AW_STATE_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


def load_verification_verdict(
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load day-14 verification verdict."""
    p = path or _AV_VERDICT_PATH
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_badge(label: str, color: str) -> str:
    """Return HTML for a colored status badge."""
    return (
        f'<span style="color:{color};font-weight:700;'
        f'font-size:0.85em;">{label}</span>'
    )


def _integrity_dot(ok: bool) -> str:
    """Green dot for intact, red dot for drift."""
    if ok:
        return '<span style="color:#22c55e;">●</span>'
    return '<span style="color:#ff1744;">●</span>'


def _gate_icon(passed: bool) -> str:
    if passed:
        return '<span style="color:#22c55e;">✓</span>'
    return '<span style="color:#ff1744;">✗</span>'


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_certification_panel(
    aw_state: Optional[Dict[str, Any]] = None,
    verdict: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the System Certification governance panel.

    Shows:
    - Window status: INACTIVE / ACTIVE / COMPLETED / HALTED
    - Progress bar (elapsed / 14 days)
    - Structural integrity (manifest + config hash)
    - Risk metrics (drawdown, vetoes, DLE mismatches)
    - Scale eligibility: LOCKED / AUTHORIZED
    - 7-gate verdict (if available)
    """
    if aw_state is None:
        aw_state = load_activation_window_state()
    if verdict is None:
        verdict = load_verification_verdict()

    active = aw_state.get("active", False)
    halted = aw_state.get("halted", False)
    window_expired = aw_state.get("window_expired", False)

    # --- Determine overall status ---
    if not aw_state or not active:
        status_label = "INACTIVE"
        status_color = "#666"
        border_color = "#2d3139"
    elif halted and window_expired:
        status_label = "COMPLETED"
        status_color = "#22c55e"
        border_color = "#22c55e"
    elif halted:
        status_label = "HALTED"
        status_color = "#ff1744"
        border_color = "#ff1744"
    else:
        status_label = "ACTIVE"
        status_color = "#00d4ff"
        border_color = "#00d4ff"

    # --- Scale eligibility ---
    verdict_text = verdict.get("verdict", "")
    verdict_passed = verdict.get("passed", 0)
    verdict_total = verdict.get("total_gates", 7)
    has_go = verdict_text == "GO" and verdict_passed == verdict_total

    if has_go:
        scale_label = "AUTHORIZED"
        scale_color = "#22c55e"
    else:
        scale_label = "LOCKED"
        scale_color = "#ff8a00"

    # --- Build the panel ---
    # Top-level status strip
    elapsed = aw_state.get("elapsed_days", 0)
    duration = aw_state.get("duration_days", 14)
    remaining = aw_state.get("remaining_days", 0)
    progress_pct = min(100, int(elapsed / max(1, duration) * 100)) if active else 0

    manifest_ok = aw_state.get("manifest_intact", True)
    config_ok = aw_state.get("config_intact", True)
    bl_ok = aw_state.get("binary_lab_freeze_ok", True)
    dd_pct = aw_state.get("drawdown_pct", 0.0)
    dd_kill = aw_state.get("drawdown_kill_pct", 0.05)
    nav_usd = aw_state.get("nav_usd", 0.0)
    episodes = aw_state.get("episodes_completed", 0)
    vetoes = aw_state.get("risk_veto_count", 0)
    dle_mm = aw_state.get("dle_mismatches", 0)
    sizing = aw_state.get("per_trade_nav_pct", 0.005)
    halt_reason = aw_state.get("halt_reason") or ""

    # DD color
    dd_color = "#ff1744" if dd_pct >= dd_kill else "#f59e0b" if dd_pct > dd_kill * 0.5 else "#22c55e"

    html = f'''
    <div style="
        background: linear-gradient(135deg, #0f1318 0%, #12141a 100%);
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    ">
        <!-- Header -->
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <span style="font-size:0.8em;color:#888;text-transform:uppercase;letter-spacing:1px;">
                🏛️ System Certification
            </span>
            <div style="display:flex;gap:12px;align-items:center;">
                <span style="font-size:0.7em;color:#888;">Scale:</span>
                {_status_badge(scale_label, scale_color)}
                <span style="font-size:0.7em;color:#888;">Window:</span>
                {_status_badge(status_label, status_color)}
            </div>
        </div>
    '''

    if active:
        # --- Progress bar ---
        bar_color = status_color
        if halted and not window_expired:
            bar_color = "#ff1744"

        html += f'''
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;font-size:0.7em;color:#888;margin-bottom:4px;">
                <span>Day {elapsed:.1f} / {duration}</span>
                <span>{remaining:.1f}d remaining</span>
            </div>
            <div style="background:#1e2128;border-radius:4px;height:8px;overflow:hidden;">
                <div style="
                    background: linear-gradient(90deg, {bar_color}88, {bar_color});
                    height:100%;
                    width:{progress_pct}%;
                    transition:width 0.5s;
                    border-radius:4px;
                "></div>
            </div>
        </div>

        <!-- Metrics grid -->
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
            <div style="background:#1a1d24;border-radius:6px;padding:8px 10px;">
                <div style="font-size:0.6em;color:#888;text-transform:uppercase;">NAV</div>
                <div style="font-size:0.95em;color:#e0e0e0;">${nav_usd:,.0f}</div>
            </div>
            <div style="background:#1a1d24;border-radius:6px;padding:8px 10px;">
                <div style="font-size:0.6em;color:#888;text-transform:uppercase;">Drawdown</div>
                <div style="font-size:0.95em;color:{dd_color};">{dd_pct:.2%}</div>
            </div>
            <div style="background:#1a1d24;border-radius:6px;padding:8px 10px;">
                <div style="font-size:0.6em;color:#888;text-transform:uppercase;">Sizing Cap</div>
                <div style="font-size:0.95em;color:#e0e0e0;">{sizing:.2%}</div>
            </div>
        </div>

        <!-- Integrity table -->
        <table style="width:100%;border-collapse:collapse;font-size:0.78em;">
            <tr>
                <td style="color:#888;padding:3px 0;">Manifest</td>
                <td style="text-align:right;padding:3px 0;">{_integrity_dot(manifest_ok)} {"Intact" if manifest_ok else "DRIFT"}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Config</td>
                <td style="text-align:right;padding:3px 0;">{_integrity_dot(config_ok)} {"Intact" if config_ok else "DRIFT"}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Binary Lab Freeze</td>
                <td style="text-align:right;padding:3px 0;">{_integrity_dot(bl_ok)} {"OK" if bl_ok else "VIOLATION"}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Episodes</td>
                <td style="text-align:right;padding:3px 0;">{episodes}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Risk Vetoes</td>
                <td style="text-align:right;padding:3px 0;">{vetoes}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">DLE Mismatches</td>
                <td style="text-align:right;padding:3px 0;">{dle_mm}</td>
            </tr>
        </table>
        '''

        # Halt banner
        if halted and not window_expired:
            html += f'''
            <div style="
                background:#ff174415;
                border:1px solid #ff174444;
                border-radius:6px;
                padding:6px 10px;
                margin-top:10px;
                font-size:0.72em;
                color:#ff8a80;
            ">🚨 HALTED — {halt_reason}</div>
            '''
        elif halted and window_expired:
            html += f'''
            <div style="
                background:#22c55e15;
                border:1px solid #22c55e44;
                border-radius:6px;
                padding:6px 10px;
                margin-top:10px;
                font-size:0.72em;
                color:#86efac;
            ">✓ Window completed — {duration} days elapsed. Run verification.</div>
            '''

    else:
        # Inactive state — minimal display
        html += '''
        <div style="padding:8px 0;font-size:0.78em;color:#555;">
            Activation window not enabled. Set <code>enabled: true</code>
            in <code>runtime.yaml</code> and <code>ACTIVATION_WINDOW_ACK=1</code> to start.
        </div>
        '''

    # --- Verdict section (if available) ---
    if verdict_text:
        if verdict_text == "GO":
            v_color = "#22c55e"
            v_icon = "✓"
        elif verdict_text == "EXTEND":
            v_color = "#f59e0b"
            v_icon = "⏳"
        else:
            v_color = "#ff1744"
            v_icon = "✗"

        evaluated_at = verdict.get("evaluated_at", "")
        gates = verdict.get("gates", {})

        html += f'''
        <div style="
            margin-top:12px;
            padding-top:10px;
            border-top:1px solid #2d3139;
        ">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                <span style="font-size:0.7em;color:#888;text-transform:uppercase;letter-spacing:0.5px;">
                    7-Gate Verification
                </span>
                <span style="font-size:1em;color:{v_color};font-weight:700;">
                    {v_icon} {verdict_text} ({verdict_passed}/{verdict_total})
                </span>
            </div>
        '''

        if gates:
            html += '<table style="width:100%;border-collapse:collapse;font-size:0.72em;">'
            for gate_name, gate_data in gates.items():
                passed = gate_data.get("pass", False)
                note = gate_data.get("note", "")
                html += f'''
                <tr>
                    <td style="color:#888;padding:2px 0;">{_gate_icon(passed)} {gate_name}</td>
                    <td style="text-align:right;padding:2px 0;color:#aaa;max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{note}</td>
                </tr>
                '''
            html += '</table>'

        if evaluated_at:
            html += f'''
            <div style="font-size:0.6em;color:#555;margin-top:6px;">
                Evaluated: {evaluated_at}
            </div>
            '''

        html += '</div>'  # close verdict section

    html += '</div>'  # close main panel
    st.html(html)
