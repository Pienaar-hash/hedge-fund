"""
Risk Health Panel (v7) - Clean, investor-ready risk visualization.

Shows:
- Drawdown % with fractional telemetry
- Daily Loss % with fractional telemetry
- Risk mode (ACTIVE / OK / DEFENSIVE)
- Position exposure as % of NAV
- Horizontal fraction bars for drawdown/daily-loss caps
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from dashboard.nav_helpers import safe_float
from dashboard.dashboard_utils import format_fraction


STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
RISK_SNAPSHOT_PATH = STATE_DIR / "risk_snapshot.json"


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    try:
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_risk_snapshot(state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the risk_snapshot.json state file."""
    base_dir = state_dir or STATE_DIR
    path = base_dir / "risk_snapshot.json"
    return _load_json(path)


def risk_metric(snapshot: Dict[str, Any], pct_field: str, frac_field: str):
    """
    Extract a risk metric's percent and fraction values.
    
    Args:
        snapshot: The risk snapshot dict
        pct_field: Dot-notation path to percent field (e.g., "dd_state.drawdown.dd_pct")
        frac_field: Key for the fractional field (e.g., "dd_frac")
    
    Returns:
        Tuple of (pct_value, frac_value) - both may be None
    """
    # Navigate dot-notation path for pct
    pct = None
    parts = pct_field.split(".")
    current = snapshot
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = None
            break
    pct = safe_float(current)
    
    # Get frac directly from top-level
    frac = safe_float(snapshot.get(frac_field))
    
    return pct, frac


def render_fraction_bar(
    label: str,
    frac: Optional[float],
    cap_frac: float,
    color_ok: str = "#21c354",
    color_warn: str = "#f2c037", 
    color_bad: str = "#d94a4a"
) -> None:
    """
    Render a horizontal progress bar showing fraction vs cap.
    
    Args:
        label: Label for the bar
        frac: Current fractional value (0-1)
        cap_frac: Cap fractional value (0-1)
        color_ok: Color when ratio < 30%
        color_warn: Color when ratio 30-70%
        color_bad: Color when ratio > 70%
    """
    if frac is None:
        st.write(f"**{label}**: —")
        return
    
    ratio = (frac / cap_frac) if cap_frac > 0 else 0
    ratio = min(max(ratio, 0), 1)
    
    # Determine color based on ratio (for future custom progress bar implementation)
    # color_ok, color_warn, color_bad are preserved for API compatibility
    _ = color_ok, color_warn, color_bad  # noqa: F841
    
    # Format cap as percentage for display
    cap_pct_display = cap_frac * 100
    
    st.markdown(f"**{label}**: {frac:.4f} / {cap_frac:.2f} ({cap_pct_display:.0f}% cap)")
    st.progress(ratio)


def render_circuit_breaker_status(snapshot: Dict[str, Any]) -> None:
    """
    Render the portfolio drawdown circuit breaker status.
    
    Args:
        snapshot: Risk snapshot from risk_snapshot.json
    """
    portfolio_dd_pct = safe_float(snapshot.get("portfolio_dd_pct"))
    circuit_breaker = snapshot.get("circuit_breaker") or {}
    
    max_dd_threshold = safe_float(circuit_breaker.get("max_portfolio_dd_nav_pct"))
    is_active = bool(circuit_breaker.get("active"))
    
    # Skip if circuit breaker is not configured
    if max_dd_threshold is None:
        return
    
    # Format values for display
    current_dd_str = f"{portfolio_dd_pct * 100:.1f}%" if portfolio_dd_pct is not None else "—"
    threshold_str = f"{max_dd_threshold * 100:.0f}%"
    
    if is_active:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #4a1a1a 0%, #3a1010 100%); 
                        border: 2px solid #ff4444; border-radius: 8px; padding: 12px; margin: 10px 0;">
                <span style="color: #ff4444; font-weight: bold;">🚨 CIRCUIT TRIPPED</span>
                <span style="color: #ccc; margin-left: 10px;">
                    DD {current_dd_str} (limit {threshold_str})
                </span>
                <div style="color: #888; font-size: 11px; margin-top: 5px;">
                    New orders are vetoed until NAV recovers or config changes.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background: #1a2a1a; border: 1px solid #333; border-radius: 8px; padding: 8px; margin: 10px 0;">
                <span style="color: #21c354;">✓</span>
                <span style="color: #aaa; margin-left: 8px; font-size: 13px;">
                    DD: {current_dd_str} (limit {threshold_str}) – circuit OK
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_tp_sl_registry_canary(snapshot: Dict[str, Any]) -> None:
    """
    V7.4_C2: Render TP/SL registry canary warning if mismatch detected.
    
    Shows a red banner when positions exist but TP/SL registry is empty,
    indicating the exit layer is impaired and positions cannot exit on SL/TP.
    
    Args:
        snapshot: Risk snapshot from risk_snapshot.json
    """
    tp_sl_info = snapshot.get("tp_sl_registry") or {}
    num_positions = tp_sl_info.get("num_positions", 0)
    num_registry_entries = tp_sl_info.get("num_registry_entries", 0)
    registry_mismatch = tp_sl_info.get("registry_mismatch", False)
    
    # Only show warning if mismatch detected
    if not registry_mismatch:
        return
    
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #5a1a1a 0%, #3a0a0a 100%); 
                    border: 3px solid #ff0000; border-radius: 8px; padding: 14px; margin: 10px 0;">
            <span style="color: #ff0000; font-weight: bold; font-size: 16px;">
                ⚠️ TP/SL REGISTRY EMPTY
            </span>
            <div style="color: #ffaaaa; margin-top: 8px; font-size: 13px;">
                <strong>{num_positions}</strong> open positions but 
                <strong>0</strong> registry entries — exit layer is impaired!
            </div>
            <div style="color: #888; font-size: 11px; margin-top: 8px;">
                Positions cannot exit on SL/TP triggers until registry is repopulated.
                Run: <code>python scripts/seed_exit_registry_from_open_positions.py</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_correlation_groups_table(snapshot: Dict[str, Any]) -> None:
    """
    Render table showing correlation group exposure vs caps.
    
    Args:
        snapshot: Risk snapshot from risk_snapshot.json
    """
    correlation_groups = snapshot.get("correlation_groups")
    if not correlation_groups or not isinstance(correlation_groups, dict):
        return
    
    # Build table data
    rows = []
    for group_name, group_data in correlation_groups.items():
        if not isinstance(group_data, dict):
            continue
        gross_nav_pct = safe_float(group_data.get("gross_nav_pct")) or 0.0
        max_group_nav_pct = safe_float(group_data.get("max_group_nav_pct")) or 0.0
        
        # Calculate status
        if max_group_nav_pct > 0:
            usage_ratio = gross_nav_pct / max_group_nav_pct
            if usage_ratio >= 1.0:
                status = "🔴 CAPPED"
            elif usage_ratio >= 0.8:
                status = "🟡 NEAR"
            else:
                status = "🟢 OK"
        else:
            usage_ratio = 0.0
            status = "⚪ N/A"
        
        rows.append({
            "Group": group_name,
            "Exposure": f"{gross_nav_pct * 100:.1f}%",
            "Cap": f"{max_group_nav_pct * 100:.0f}%",
            "Status": status,
        })
    
    if not rows:
        return
    
    st.markdown("##### Correlation Groups")
    
    # Render as styled table
    table_html = """
    <style>
    .corr-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .corr-table th { text-align: left; color: #888; border-bottom: 1px solid #333; padding: 6px 8px; }
    .corr-table td { padding: 6px 8px; border-bottom: 1px solid #222; }
    .corr-table tr:last-child td { border-bottom: none; }
    </style>
    <table class="corr-table">
        <tr><th>Group</th><th>Exposure</th><th>Cap</th><th>Status</th></tr>
    """
    for row in rows:
        table_html += f"""<tr>
            <td style="color: #ccc;">{row['Group']}</td>
            <td style="color: #aaa;">{row['Exposure']}</td>
            <td style="color: #888;">{row['Cap']}</td>
            <td>{row['Status']}</td>
        </tr>"""
    table_html += "</table>"
    
    st.markdown(table_html, unsafe_allow_html=True)


def render_risk_health_card(
    snapshot: Dict[str, Any],
    nav_value: float = 0,
    gross_exposure: float = 0,
    cap_drawdown: float = 0.30,
    cap_daily_loss: float = 0.10,
) -> None:
    """
    Render the v7 Risk Health card with fractional telemetry.
    
    Args:
        snapshot: Risk snapshot from risk_snapshot.json
        nav_value: Current NAV in USD
        gross_exposure: Current gross exposure in USD
        cap_drawdown: Max drawdown cap as fraction (default 0.30 = 30%)
        cap_daily_loss: Max daily loss cap as fraction (default 0.10 = 10%)
    """
    st.markdown("""
    <style>
    .risk-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #333;
        margin-bottom: 15px;
    }
    .risk-card-header {
        font-size: 14px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 15px;
        border-bottom: 1px solid #333;
        padding-bottom: 10px;
    }
    .risk-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #222;
    }
    .risk-metric-label {
        color: #aaa;
        font-size: 13px;
    }
    .risk-metric-value {
        font-size: 14px;
        font-weight: 600;
    }
    .risk-frac {
        color: #666;
        font-size: 11px;
        margin-left: 8px;
    }
    .risk-mode-ok { color: #21c354; }
    .risk-mode-warn { color: #f2c037; }
    .risk-mode-defensive { color: #d94a4a; }
    .risk-mode-halted { color: #ff0033; }
    .risk-mode-reason {
        color: #666;
        font-size: 11px;
        margin-left: 8px;
    }
    .risk-mode-score {
        color: #888;
        font-size: 10px;
        margin-left: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Extract values from snapshot
    dd_state_block = snapshot.get("dd_state") or {}
    drawdown_block = dd_state_block.get("drawdown") or {} if isinstance(dd_state_block, dict) else {}
    
    # Get percent-style values
    dd_pct = safe_float(drawdown_block.get("dd_pct"))
    daily_loss_block = drawdown_block.get("daily_loss") or {}
    daily_loss_pct = safe_float(daily_loss_block.get("pct"))
    
    # Get normalized fractions (new fields)
    dd_frac = safe_float(snapshot.get("dd_frac"))
    daily_loss_frac = safe_float(snapshot.get("daily_loss_frac"))
    
    # Get risk mode from new v7 fields (preferred) or fallback to dd_state
    risk_mode = snapshot.get("risk_mode")
    risk_mode_reason = snapshot.get("risk_mode_reason") or ""
    risk_mode_score = safe_float(snapshot.get("risk_mode_score"))
    
    # Fallback to dd_state if risk_mode not present
    if not risk_mode:
        if isinstance(dd_state_block, dict):
            risk_mode = dd_state_block.get("dd_state") or "OK"
        elif isinstance(dd_state_block, str):
            risk_mode = dd_state_block
        else:
            risk_mode = "OK"
    
    # Calculate exposure percentage
    exposure_pct = (gross_exposure / nav_value * 100) if nav_value > 0 else 0
    
    # Risk mode styling - v7 classification
    mode_upper = str(risk_mode).upper()
    if mode_upper == "OK":
        mode_class = "risk-mode-ok"
        mode_display = "OK"
    elif mode_upper == "WARN":
        mode_class = "risk-mode-warn"
        mode_display = "WARN"
    elif mode_upper == "DEFENSIVE":
        mode_class = "risk-mode-defensive"
        mode_display = "DEFENSIVE"
    elif mode_upper == "HALTED":
        mode_class = "risk-mode-halted"
        mode_display = "HALTED"
    else:
        # Legacy fallback
        if mode_upper in ("ACTIVE", "NORMAL"):
            mode_class = "risk-mode-ok"
            mode_display = "OK"
        else:
            mode_class = "risk-mode-ok"
            mode_display = mode_upper or "OK"
    
    # Format values for display
    dd_pct_str = f"-{dd_pct:.2f}%" if dd_pct is not None else "—"
    dd_frac_str = f"(frac: {dd_frac:.4f})" if dd_frac is not None else ""
    
    daily_pct_str = f"-{daily_loss_pct:.2f}%" if daily_loss_pct is not None else "—"
    daily_frac_str = f"(frac: {daily_loss_frac:.4f})" if daily_loss_frac is not None else ""
    
    exposure_str = f"{exposure_pct:.1f}% of NAV"
    
    # Format risk mode reason/score for display
    risk_mode_reason_display = f'<span class="risk-mode-reason">{risk_mode_reason}</span>' if risk_mode_reason else ""
    risk_mode_score_display = f'<span class="risk-mode-score">[{risk_mode_score:.2f}]</span>' if risk_mode_score is not None else ""
    
    # Render the card
    st.markdown(f"""
    <div class="risk-card">
        <div class="risk-card-header">🛡️ RISK HEALTH</div>
        <div class="risk-metric-row">
            <span class="risk-metric-label">Drawdown</span>
            <span class="risk-metric-value">{dd_pct_str} <span class="risk-frac">{dd_frac_str}</span></span>
        </div>
        <div class="risk-metric-row">
            <span class="risk-metric-label">Daily Loss</span>
            <span class="risk-metric-value">{daily_pct_str} <span class="risk-frac">{daily_frac_str}</span></span>
        </div>
        <div class="risk-metric-row">
            <span class="risk-metric-label">Risk Mode</span>
            <span class="risk-metric-value {mode_class}">{mode_display} {risk_mode_score_display}</span>{risk_mode_reason_display}
        </div>
        <div class="risk-metric-row" style="border-bottom:none;">
            <span class="risk-metric-label">Position Exposure</span>
            <span class="risk-metric-value">{exposure_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render circuit breaker status
    render_circuit_breaker_status(snapshot)
    
    # V7.4_C2: Render TP/SL registry canary warning if mismatch detected
    render_tp_sl_registry_canary(snapshot)
    
    # Render correlation groups table
    render_correlation_groups_table(snapshot)
    
    # Render fraction bars
    st.markdown("##### Utilization vs Caps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_fraction_bar("Drawdown", dd_frac, cap_drawdown)
    
    with col2:
        render_fraction_bar("Daily Loss", daily_loss_frac, cap_daily_loss)
    
    # v7.5_A1: Render advanced risk metrics (VaR, CVaR, Alpha Decay)
    render_risk_advanced_section(snapshot)


def render_risk_summary_compact(snapshot: Dict[str, Any]) -> None:
    """
    Render a compact risk summary (for sidebar or small spaces).
    
    Args:
        snapshot: Risk snapshot from risk_snapshot.json
    """
    dd_frac = safe_float(snapshot.get("dd_frac"))
    daily_loss_frac = safe_float(snapshot.get("daily_loss_frac"))
    
    # Get risk mode from new v7 fields (preferred) or fallback
    risk_mode = snapshot.get("risk_mode")
    if not risk_mode:
        dd_state_block = snapshot.get("dd_state") or {}
        if isinstance(dd_state_block, dict):
            risk_mode = dd_state_block.get("dd_state") or "OK"
        elif isinstance(dd_state_block, str):
            risk_mode = dd_state_block
        else:
            risk_mode = "OK"
    
    dd_str = format_fraction(dd_frac) if dd_frac is not None else "—"
    daily_str = format_fraction(daily_loss_frac) if daily_loss_frac is not None else "—"
    
    # Emoji mapping for v7 risk modes
    mode_upper = str(risk_mode).upper()
    if mode_upper == "OK":
        mode_emoji = "🟢"
    elif mode_upper == "WARN":
        mode_emoji = "🟡"
    elif mode_upper == "DEFENSIVE":
        mode_emoji = "🟠"
    elif mode_upper == "HALTED":
        mode_emoji = "🔴"
    else:
        # Legacy fallback
        mode_emoji = "🟢" if mode_upper in ("NORMAL", "ACTIVE") else "🔴"
    
    st.caption(f"{mode_emoji} {mode_upper} | DD: {dd_str} | Daily: {daily_str}")


# ---------------------------------------------------------------------------
# v7.5_A1: VaR, CVaR, and Alpha Decay Panels
# ---------------------------------------------------------------------------

def render_portfolio_var_panel(snapshot: Dict[str, Any]) -> None:
    """
    Render Portfolio VaR (Value at Risk) status panel (v7.5_A1).
    
    Shows:
    - Portfolio VaR as % of NAV
    - VaR limit and status (green/yellow/red)
    - Confidence level
    
    Args:
        snapshot: Risk snapshot from risk_snapshot.json
    """
    var_data = snapshot.get("var") or {}
    
    if not var_data:
        return  # VaR not computed
    
    var_nav_pct = safe_float(var_data.get("portfolio_var_nav_pct"))
    max_var_pct = safe_float(var_data.get("max_portfolio_var_nav_pct"))
    var_usd = safe_float(var_data.get("portfolio_var_usd"))
    confidence = safe_float(var_data.get("confidence")) or 0.99
    within_limit = var_data.get("within_limit", True)
    portfolio_vol = safe_float(var_data.get("portfolio_volatility"))
    
    if var_nav_pct is None:
        return
    
    # Determine color based on ratio to limit
    ratio = (var_nav_pct / max_var_pct) if max_var_pct and max_var_pct > 0 else 0
    
    if ratio < 0.70:
        status_color = "#21c354"  # Green
        status_emoji = "🟢"
        status_text = "OK"
    elif ratio < 1.0:
        status_color = "#f2c037"  # Yellow
        status_emoji = "🟡"
        status_text = "WARN"
    else:
        status_color = "#ff4444"  # Red
        status_emoji = "🔴"
        status_text = "BREACH"
    
    var_pct_display = f"{var_nav_pct * 100:.1f}%"
    limit_pct_display = f"{max_var_pct * 100:.0f}%" if max_var_pct else "—"
    var_usd_display = f"${var_usd:,.0f}" if var_usd else "—"
    vol_display = f"{portfolio_vol * 100:.1f}%" if portfolio_vol else "—"
    
    st.markdown(f"""
    <div style="background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 12px; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #aaa; font-weight: bold;">📊 Portfolio VaR ({confidence*100:.0f}%)</span>
            <span style="color: {status_color}; font-weight: bold;">{status_emoji} {status_text}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px;">
            <span style="color: #888;">VaR</span>
            <span style="color: #eee;">{var_pct_display} ({var_usd_display})</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Limit</span>
            <span style="color: #eee;">{limit_pct_display}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #888;">Portfolio Vol</span>
            <span style="color: #eee;">{vol_display} ann.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar for utilization
    st.progress(min(ratio, 1.0))


def render_position_cvar_table(snapshot: Dict[str, Any]) -> None:
    """
    Render Position CVaR (Expected Shortfall) table (v7.5_A1).
    
    Shows per-symbol CVaR as % of NAV with limit status.
    
    Args:
        snapshot: Risk snapshot from risk_snapshot.json
    """
    cvar_data = snapshot.get("cvar") or {}
    per_symbol = cvar_data.get("per_symbol") or {}
    
    if not per_symbol:
        return  # No CVaR data
    
    max_cvar_pct = safe_float(cvar_data.get("max_position_cvar_nav_pct")) or 0.04
    confidence = safe_float(cvar_data.get("confidence")) or 0.95
    
    st.markdown(f"##### Position CVaR ({confidence*100:.0f}% ES)")
    
    # Build table data
    rows = []
    for symbol, data in sorted(per_symbol.items()):
        cvar_pct = safe_float(data.get("cvar_nav_pct")) or 0
        within_limit = data.get("within_limit", True)
        
        status = "🟢" if within_limit else "🔴"
        rows.append({
            "Symbol": symbol,
            "CVaR % NAV": f"{cvar_pct * 100:.2f}%",
            "Limit": f"{max_cvar_pct * 100:.1f}%",
            "Status": status,
        })
    
    if rows:
        # Use streamlit dataframe for display
        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_alpha_decay_panel(decay_data: Dict[str, Any]) -> None:
    """
    Render Alpha Decay status panel (v7.5_A1).
    
    Shows decay multiplier per symbol indicating signal freshness.
    
    Args:
        decay_data: Alpha decay snapshot from alpha_decay.json or risk_snapshot
    """
    if not decay_data:
        return
    
    symbols_data = decay_data.get("symbols") or {}
    config = decay_data.get("config") or {}
    
    if not symbols_data:
        return
    
    half_life = config.get("half_life_minutes", 45)
    min_mult = config.get("min_decay_multiplier", 0.35)
    
    st.markdown(f"##### Alpha Decay (half-life: {half_life}min)")
    
    # Build decay display
    cols = st.columns(min(len(symbols_data), 4))
    col_idx = 0
    
    for symbol, directions in sorted(symbols_data.items()):
        # Get the primary direction's decay (prefer LONG if both exist)
        long_data = directions.get("long") or {}
        short_data = directions.get("short") or {}
        
        # Pick the one with lower decay (more aged signal)
        if long_data and short_data:
            decay_mult = min(
                long_data.get("decay_multiplier", 1.0),
                short_data.get("decay_multiplier", 1.0)
            )
            at_min = long_data.get("at_minimum") or short_data.get("at_minimum")
        elif long_data:
            decay_mult = long_data.get("decay_multiplier", 1.0)
            at_min = long_data.get("at_minimum", False)
        elif short_data:
            decay_mult = short_data.get("decay_multiplier", 1.0)
            at_min = short_data.get("at_minimum", False)
        else:
            continue
        
        # Determine color
        if at_min:
            color = "#888"  # Gray for at minimum
        elif decay_mult > 0.7:
            color = "#21c354"  # Green for fresh
        elif decay_mult > 0.5:
            color = "#f2c037"  # Yellow for aging
        else:
            color = "#ff8800"  # Orange for stale
        
        with cols[col_idx % len(cols)]:
            suffix = " (min)" if at_min else ""
            st.markdown(f"""
            <div style="text-align: center; padding: 5px;">
                <span style="color: #aaa; font-size: 11px;">{symbol}</span><br/>
                <span style="color: {color}; font-weight: bold;">{decay_mult:.2f}x{suffix}</span>
            </div>
            """, unsafe_allow_html=True)
        
        col_idx += 1


def load_alpha_decay_snapshot(state_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the alpha_decay.json state file."""
    base_dir = state_dir or STATE_DIR
    path = base_dir / "alpha_decay.json"
    return _load_json(path)


def render_risk_advanced_section(snapshot: Dict[str, Any], state_dir: Optional[Path] = None) -> None:
    """
    Render the full advanced risk section including VaR, CVaR, and Alpha Decay (v7.5_A1).
    
    Args:
        snapshot: Risk snapshot from risk_snapshot.json
        state_dir: Optional state directory override
    """
    st.markdown("---")
    st.markdown("#### Advanced Risk Metrics (v7.5)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_portfolio_var_panel(snapshot)
    
    with col2:
        render_position_cvar_table(snapshot)
    
    # Load and render alpha decay
    decay_snapshot = load_alpha_decay_snapshot(state_dir)
    if decay_snapshot:
        render_alpha_decay_panel(decay_snapshot)


def render_veto_heatmap(diag: Dict[str, Any]) -> None:
    st.markdown("#### 🛑 Veto Heatmap")
    veto = diag.get("veto_counters") if isinstance(diag, dict) else {}
    by_reason = veto.get("by_reason") if isinstance(veto, dict) else {}
    total = veto.get("total_vetoes") if isinstance(veto, dict) else 0
    if not by_reason:
        st.caption("No veto diagnostics recorded yet.")
        return
    rows = []
    for reason, count in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True):
        share = (float(count) / float(total) * 100.0) if total else 0.0
        rows.append({"Reason": reason, "Count": int(count), "Share": f"{share:.1f}%"})
    st.table(rows)
    if veto.get("last_veto_ts"):
        st.caption(f"Last veto at {veto.get('last_veto_ts')}")


def render_exit_pipeline_status(diag: Dict[str, Any]) -> None:
    st.markdown("#### 🏁 Exit Pipeline Health")
    exit_block = diag.get("exit_pipeline") if isinstance(diag, dict) else {}
    if not exit_block:
        st.caption("No exit diagnostics recorded yet.")
        return
    coverage_pct = exit_block.get("tp_sl_coverage_pct")
    cols = st.columns(5)
    cols[0].metric("Open Positions", exit_block.get("open_positions_count", 0))
    cols[1].metric("TP/SL Registered", exit_block.get("tp_sl_registered_count", 0))
    cols[2].metric("TP/SL Missing", exit_block.get("tp_sl_missing_count", 0))
    cols[3].metric("Underwater w/o TP/SL", exit_block.get("underwater_without_tp_sl_count", 0))
    if coverage_pct is not None:
        cols[4].metric("TP/SL Coverage", f"{float(coverage_pct)*100:.0f}%")
    else:
        cols[4].metric("TP/SL Coverage", "n/a")
    last_scan = exit_block.get("last_exit_scan_ts")
    last_trigger = exit_block.get("last_exit_trigger_ts")
    last_router_event = exit_block.get("last_router_event_ts")
    mismatch = exit_block.get("ledger_registry_mismatch")
    if mismatch:
        st.error("Ledger/registry mismatch detected.")
    if last_scan or last_trigger or last_router_event:
        st.caption(
            f"Last scan: {last_scan or 'n/a'} | "
            f"Last trigger: {last_trigger or 'n/a'} | "
            f"Last router event: {last_router_event or 'n/a'}"
        )


def render_liveness_status(diag: Dict[str, Any]) -> None:
    st.markdown("#### ⏱️ Liveness")
    live = diag.get("liveness") if isinstance(diag, dict) else {}
    if not live:
        st.caption("No liveness diagnostics recorded yet.")
        return

    def _fmt_val(key: str) -> str:
        try:
            seconds = float(details.get(key, 0) or 0)
        except Exception:
            return "n/a"
        if seconds <= 0:
            return "n/a"
        if seconds >= 3600:
            return f"{seconds/3600:.1f}h"
        if seconds >= 60:
            return f"{seconds/60:.1f}m"
        return f"{seconds:.0f}s"

    details = live.get("details") or {}
    missing = live.get("missing") or {}
    cols = st.columns(4)
    cols[0].metric("Signals idle", _fmt_val("signals_idle_seconds"), None, "inverse")
    cols[1].metric("Orders idle", _fmt_val("orders_idle_seconds"), None, "inverse")
    cols[2].metric("Exits idle", _fmt_val("exits_idle_seconds"), None, "inverse")
    cols[3].metric("Router idle", _fmt_val("router_idle_seconds"), None, "inverse")

    flags = []
    for label, key in (
        ("signals", "idle_signals"),
        ("orders", "idle_orders"),
        ("exits", "idle_exits"),
        ("router", "idle_router"),
    ):
        if live.get(key):
            flags.append(label)
    if flags:
        st.warning(f"Idle watchers tripped: {', '.join(flags)}")
    else:
        st.success("All watchers active recently.")
    missing_keys = [k for k, v in missing.items() if v]
    if missing_keys:
        st.info(f"Missing timestamps: {', '.join(sorted(missing_keys))}")
