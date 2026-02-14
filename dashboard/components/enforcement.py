"""
Phase C.1 Enforcement Widget — Governance Surface

Displays DLE enforcement status, split-brain integrity,
and Phase C readiness window progress.

Data source: logs/state/phase_c_readiness.json

This is a governance-level surface — not optional telemetry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_enforcement_state(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load Phase C readiness + enforcement state from file."""
    p = path or Path("logs/state/phase_c_readiness.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_enforcement_widget(state: Optional[Dict[str, Any]] = None) -> None:
    """
    Render Phase C.1 Enforcement governance widget.

    Shows:
    - Enforcement status (ON/OFF)
    - Rehearsal status (shadow layer active)
    - Entry denial rate
    - Split-brain count (consistency watchdog)
    - Phase C window progress (days / 14)
    - Gate satisfied status
    """
    if state is None:
        state = load_enforcement_state()

    if not state:
        return

    enforcement = state.get("enforcement", {})
    enforce_on = enforcement.get("enforce_enabled", False)
    rehearsal_on = state.get("rehearsal_enabled", False)

    # Enforcement metrics
    entry_evaluated = enforcement.get("entry_evaluated", 0)
    entry_permitted = enforcement.get("entry_permitted", 0)
    entry_denied = enforcement.get("entry_denied", 0)
    entry_blocks_pct = enforcement.get("entry_blocks_pct", 0.0)
    exit_passthrough = enforcement.get("exit_passthrough", 0)
    split_brain = enforcement.get("split_brain_count", 0)

    # Readiness window
    window_days_met = state.get("window_days_met", 0)
    window_days_required = state.get("window_days_required", 14)
    criteria_met = state.get("criteria_met", False)
    gate_satisfied = state.get("gate_satisfied", False)
    breach_reason = state.get("breach_reason")

    # Rehearsal metrics
    would_block_pct = state.get("current_metrics", {}).get("would_block_pct", 0.0)
    expired_count = state.get("current_metrics", {}).get("expired_permit_count", 0)
    missing_count = state.get("current_metrics", {}).get("missing_permit_count", 0)

    # --- Status badges ---
    if enforce_on:
        enforce_badge = '<span style="color:#21c354;font-weight:700;">● ON</span>'
        enforce_border = "#21c354"
    else:
        enforce_badge = '<span style="color:#888;font-weight:700;">○ OFF</span>'
        enforce_border = "#2d3139"

    if rehearsal_on:
        rehearsal_badge = '<span style="color:#9370db;">● SHADOW</span>'
    else:
        rehearsal_badge = '<span style="color:#555;">○ INACTIVE</span>'

    # Split-brain indicator
    if split_brain > 0:
        sb_color = "#ff1744"
        sb_badge = f'<span style="color:{sb_color};font-weight:700;">⚠ {split_brain}</span>'
    else:
        sb_color = "#21c354"
        sb_badge = f'<span style="color:{sb_color};">0</span>'

    # Window progress
    progress_pct = min(100, int(window_days_met / max(1, window_days_required) * 100))
    if gate_satisfied:
        window_color = "#21c354"
        window_label = "SATISFIED"
    elif criteria_met:
        window_color = "#f2c037"
        window_label = f"Day {window_days_met}/{window_days_required}"
    else:
        window_color = "#888"
        window_label = f"Day {window_days_met}/{window_days_required}"

    # --- Denial rate display ---
    if entry_evaluated > 0:
        denial_display = f"{entry_blocks_pct:.2f}%"
        denial_color = "#ff1744" if entry_blocks_pct > 1.0 else "#f2c037" if entry_blocks_pct > 0 else "#21c354"
    else:
        denial_display = "—"
        denial_color = "#888"

    # --- Build HTML ---
    html = f'''
    <div style="
        background: linear-gradient(135deg, #1a1d24 0%, #12141a 100%);
        border: 1px solid {enforce_border};
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    ">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <span style="font-size:0.75em;color:#888;text-transform:uppercase;letter-spacing:0.5px;">
                🛡️ DLE Authority Gate
            </span>
            <span style="font-size:0.75em;">
                {enforce_badge}
            </span>
        </div>

        <table style="width:100%;border-collapse:collapse;font-size:0.8em;">
            <tr>
                <td style="color:#888;padding:3px 0;">Enforcement</td>
                <td style="text-align:right;padding:3px 0;">{enforce_badge}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Rehearsal</td>
                <td style="text-align:right;padding:3px 0;">{rehearsal_badge}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Split-Brain</td>
                <td style="text-align:right;padding:3px 0;">{sb_badge}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Denial Rate</td>
                <td style="text-align:right;padding:3px 0;">
                    <span style="color:{denial_color};">{denial_display}</span>
                </td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Entries Evaluated</td>
                <td style="text-align:right;padding:3px 0;">{entry_evaluated}</td>
            </tr>
            <tr>
                <td style="color:#888;padding:3px 0;">Exits Passed</td>
                <td style="text-align:right;padding:3px 0;">{exit_passthrough}</td>
            </tr>
        </table>

        <!-- Window progress bar -->
        <div style="margin-top:10px;">
            <div style="display:flex;justify-content:space-between;font-size:0.7em;color:#888;margin-bottom:3px;">
                <span>Phase C Window</span>
                <span style="color:{window_color};">{window_label}</span>
            </div>
            <div style="background:#2d3139;border-radius:3px;height:6px;overflow:hidden;">
                <div style="
                    background:{window_color};
                    height:100%;
                    width:{progress_pct}%;
                    transition:width 0.3s;
                "></div>
            </div>
        </div>

        <!-- Rehearsal integrity -->
        <div style="margin-top:8px;font-size:0.65em;color:#555;">
            would_block={would_block_pct:.2f}%
            · expired={expired_count}
            · missing={missing_count}
        </div>
    '''

    # Breach warning
    if breach_reason and not criteria_met:
        html += f'''
        <div style="
            background:#ff174412;
            border:1px solid #ff174433;
            border-radius:4px;
            padding:4px 8px;
            margin-top:8px;
            font-size:0.65em;
            color:#ff8a80;
        ">⚠ {breach_reason}</div>
        '''

    # Split-brain critical alert
    if split_brain > 0:
        last_sb_sym = enforcement.get("last_split_brain_symbol", "")
        html += f'''
        <div style="
            background:#ff174422;
            border:1px solid #ff1744;
            border-radius:4px;
            padding:4px 8px;
            margin-top:8px;
            font-size:0.7em;
            color:#ff1744;
        ">🚨 SPLIT-BRAIN DETECTED — {split_brain} divergence(s), last: {last_sb_sym}</div>
        '''

    html += '</div>'
    st.html(html)
