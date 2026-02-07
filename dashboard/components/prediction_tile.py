"""
Prediction Layer Telemetry Tile — P1 Advisory

Minimal read-only tile showing prediction system health for the
P1_ADVISORY phase. All data comes from logs/prediction/ JSONL files.

This component NEVER imports from execution/ or prediction/.
It reads log files directly, matching the dashboard one-way dependency rule.

Surfaces:
    - rankings_applied rate (24h)
    - missing prediction state rate
    - last ranking timestamp
    - firewall denials (should stay 0 during P1)
    - current phase
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Log paths (read-only)
# ---------------------------------------------------------------------------

_PRED_LOG_DIR = Path(os.getenv("PRED_LOG_DIR", "logs/prediction"))
_RANKING_LOG = _PRED_LOG_DIR / "alert_ranking.jsonl"
_FIREWALL_LOG = _PRED_LOG_DIR / "firewall_denials.jsonl"


# ---------------------------------------------------------------------------
# State Loader
# ---------------------------------------------------------------------------

def load_prediction_telemetry(log_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load prediction telemetry from JSONL logs.

    Returns a dict with:
        phase: str
        enabled: bool
        ranking_total: int
        ranking_applied: int
        ranking_rate: float (0.0-1.0)
        no_snapshot_count: int
        last_ranking_ts: str | None
        firewall_denials: int
        firewall_by_verdict: dict[str, int]
    """
    pred_dir = log_dir or _PRED_LOG_DIR
    ranking_log = pred_dir / "alert_ranking.jsonl"
    firewall_log = pred_dir / "firewall_denials.jsonl"

    # Phase / enabled from env (same as prediction layer reads)
    phase = os.environ.get("PREDICTION_PHASE", "P0_OBSERVE")
    enabled = os.environ.get("PREDICTION_DLE_ENABLED", "0") == "1"

    # Alert ranking stats
    ranking_total = 0
    ranking_applied = 0
    no_snapshot_count = 0
    last_ranking_ts: Optional[str] = None

    if ranking_log.exists():
        try:
            with ranking_log.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        ranking_total += 1
                        if rec.get("rankings_applied"):
                            ranking_applied += 1
                        if rec.get("reason", "").startswith("No prediction"):
                            no_snapshot_count += 1
                        ts = rec.get("ts")
                        if ts:
                            last_ranking_ts = ts
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass

    ranking_rate = (ranking_applied / ranking_total) if ranking_total > 0 else 0.0

    # Firewall denials
    firewall_denials = 0
    firewall_by_verdict: Dict[str, int] = {}

    if firewall_log.exists():
        try:
            with firewall_log.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        firewall_denials += 1
                        verdict = rec.get("verdict", "UNKNOWN")
                        firewall_by_verdict[verdict] = firewall_by_verdict.get(verdict, 0) + 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass

    return {
        "phase": phase,
        "enabled": enabled,
        "ranking_total": ranking_total,
        "ranking_applied": ranking_applied,
        "ranking_rate": ranking_rate,
        "no_snapshot_count": no_snapshot_count,
        "last_ranking_ts": last_ranking_ts,
        "firewall_denials": firewall_denials,
        "firewall_by_verdict": firewall_by_verdict,
    }


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

_PHASE_COLORS = {
    "P0_OBSERVE": "#888",
    "P1_ADVISORY": "#f2c037",
    "P2_PRODUCTION": "#21c354",
}

_PHASE_LABELS = {
    "P0_OBSERVE": "OBSERVE",
    "P1_ADVISORY": "ADVISORY",
    "P2_PRODUCTION": "PRODUCTION",
}


def _format_ts(iso_ts: Optional[str]) -> str:
    """Format an ISO timestamp for display."""
    if not iso_ts:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_ts)
        now = datetime.now(timezone.utc)
        delta = now - dt.astimezone(timezone.utc)
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m ago"
        if hours < 24:
            return f"{hours:.1f}h ago"
        return f"{hours / 24:.1f}d ago"
    except Exception:
        return iso_ts[:19] if len(iso_ts) >= 19 else iso_ts


def render_prediction_tile(state: Optional[Dict[str, Any]] = None) -> None:
    """
    Render prediction telemetry tile.

    Collapsed by default, inside the Diagnostics expander.
    Shows only what P1 proved is safe to display.
    """
    if state is None:
        state = load_prediction_telemetry()

    phase = state.get("phase", "P0_OBSERVE")
    enabled = state.get("enabled", False)
    ranking_total = state.get("ranking_total", 0)
    ranking_applied = state.get("ranking_applied", 0)
    ranking_rate = state.get("ranking_rate", 0.0)
    no_snapshot_count = state.get("no_snapshot_count", 0)
    last_ranking_ts = state.get("last_ranking_ts")
    firewall_denials = state.get("firewall_denials", 0)

    # Phase badge
    phase_color = _PHASE_COLORS.get(phase, "#888")
    phase_label = _PHASE_LABELS.get(phase, phase)

    disabled_html = "" if enabled else ' <span style="color:#888;font-size:0.8em;">DISABLED</span>'

    st.markdown(
        f'<span style="background:{phase_color};color:#000;padding:2px 8px;'
        f'border-radius:4px;font-weight:600;font-size:0.85em;">'
        f'PREDICTION: {phase_label}</span>'
        f'{disabled_html}',
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rate_pct = f"{ranking_rate:.0%}" if ranking_total > 0 else "—"
        st.metric(
            label="Rankings Applied",
            value=rate_pct,
            help=f"{ranking_applied}/{ranking_total} batches reordered by prediction relevance",
        )

    with col2:
        if ranking_total > 0:
            miss_rate = f"{no_snapshot_count}/{ranking_total}"
        else:
            miss_rate = "—"
        st.metric(
            label="Missing Snapshots",
            value=miss_rate,
            help="Ranking calls where no prediction data was available",
        )

    with col3:
        st.metric(
            label="Last Ranking",
            value=_format_ts(last_ranking_ts),
            help="Time since last alert ranking event",
        )

    with col4:
        denial_color = "normal" if firewall_denials == 0 else "inverse"
        st.metric(
            label="Firewall Denials",
            value=str(firewall_denials),
            help="Total firewall denials (pre-P1 baseline; delta should be 0)",
        )
