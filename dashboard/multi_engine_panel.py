"""Multi-Engine Soak Command Center — single-screen observability.

Reads ``logs/state/fallback_metrics.json`` (written by FallbackTelemetry)
and renders four stacked blocks:

1. Portfolio Edge   — CEL, conflict rate
2. Engine Health    — HQD, participation, rescue, overconfidence
3. Calibration      — SDD (score scale delta)
4. Regime Edge      — per-regime RSD table
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:  # pragma: no cover
    HAS_STREAMLIT = False

_LOG = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("logs/state/fallback_metrics.json")


# ---------------------------------------------------------------------------
# State loader
# ---------------------------------------------------------------------------

def load_fallback_metrics(state_path: Path | str | None = None) -> Dict[str, Any]:
    path = Path(state_path or DEFAULT_STATE_PATH)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, IOError):
        return {}


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _color(val: float, good_lo: float, good_hi: float, warn_hi: float) -> str:
    """Return CSS color based on value vs healthy range."""
    if good_lo <= val <= good_hi:
        return "#21ba45"  # green
    if abs(val) <= warn_hi:
        return "#f2c037"  # amber
    return "#db2828"  # red


def _signed(val: float, digits: int = 4) -> str:
    return f"{val:+.{digits}f}"


_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], lo: float = 0.0, hi: float = 1.0) -> str:
    """Return a Unicode sparkline string for a list of floats."""
    if not values:
        return ""
    span = hi - lo if hi != lo else 1.0
    n = len(_SPARK_BLOCKS) - 1
    return "".join(
        _SPARK_BLOCKS[min(n, max(0, round((v - lo) / span * n)))]
        for v in values
    )


def _render_portfolio_edge(s: Dict[str, Any]) -> None:
    st.markdown("##### Portfolio Edge")
    c1, c2, c3 = st.columns(3)
    cel = s.get("conflict_edge_lift", 0.0)
    cel_color = _color(cel, 0.005, 0.06, 0.02)
    c1.metric("CEL", _signed(cel))
    c1.caption(f":{cel_color[1:]}[{'adding edge' if cel > 0.005 else 'neutral' if abs(cel) <= 0.005 else 'degrading'}]")
    c2.metric("Conflicts", s.get("cel_count", 0))
    cr = s.get("conflict_rate", 0.0)
    c3.metric("Conflict Rate", f"{cr:.1%}")
    cr_note = "healthy" if 0.10 <= cr <= 0.35 else "low" if cr < 0.10 else "high"
    c3.caption(cr_note)


def _render_engine_health(s: Dict[str, Any]) -> None:
    st.markdown("##### Engine Health")
    c1, c2, c3, c4 = st.columns(4)
    hqd = s.get("hydra_quality_diff", 0.0)
    c1.metric("HQD", _signed(hqd))
    part = s.get("hydra_participation", 0.0)
    c2.metric("Participation", f"{part:.1%}")
    c3.metric("Rescue Rate", f"{s.get('hydra_rescue_rate', 0.0):.1%}")
    oc = s.get("hydra_overconfidence", 0.0)
    c4.metric("Overconfidence", f"{oc:.1%}")
    if oc > 0.25:
        c4.caption(":red[high]")


def _render_calibration(s: Dict[str, Any]) -> None:
    st.markdown("##### Calibration")
    c1, c2, c3, c4 = st.columns(4)
    delta = s.get("score_scale_delta", 0.0)
    delta_color = _color(abs(delta), 0.0, 0.02, 0.05)
    c1.metric("Score Scale Δ", _signed(delta))
    if abs(delta) > 0.05:
        c1.caption(":red[drift warning — merge fairness at risk]")
    elif abs(delta) > 0.02:
        c1.caption(":orange[minor drift]")
    else:
        c1.caption(":green[aligned]")
    c2.metric("Hydra Mean", f"{s.get('sdd_hydra_mean', 0.0):.4f}")
    c3.metric("Legacy Mean", f"{s.get('sdd_legacy_mean', 0.0):.4f}")
    c4.metric("Samples", s.get("sdd_count", 0))


def _render_architecture_status(s: Dict[str, Any]) -> None:
    st.markdown("##### Architecture Status (MRI)")
    ecs_ready = s.get("ecs_ready", False)
    score = s.get("ecs_readiness_score", 0.0)

    # Overall status badge
    if ecs_ready:
        st.success("READY — fallback layer can be safely removed")
    else:
        st.warning(f"NOT READY — readiness score {score:.0%}")

    # Progress bar
    st.progress(min(score, 1.0))

    # Per-condition breakdown
    conditions = [
        ("Trade Volume", "ecs_ready_trades", f"cel_count ≥ 300 (current: {s.get('cel_count', 0)})"),
        ("Stable Recovery", "ecs_stable_recovery", f"fallback_rate < 5% (current: {s.get('fallback_rate', 0.0):.1%})"),
        ("Positive Edge", "ecs_positive_edge", f"CEL > 0 (current: {s.get('conflict_edge_lift', 0.0):+.4f})"),
        ("Score Calibrated", "ecs_score_calibrated", f"|SDD| ≤ 0.02 (current: {abs(s.get('score_scale_delta', 0.0)):.4f})"),
    ]
    for label, key, detail in conditions:
        ok = s.get(key, False)
        icon = "✅" if ok else "❌"
        st.markdown(
            f'<span style="font-family:monospace;font-size:14px;">'
            f'{icon} <b>{label}</b> — {detail}'
            f'</span>',
            unsafe_allow_html=True,
        )

    # MRI trend sparkline
    history = s.get("ecs_score_history") or []
    if len(history) >= 2:
        spark = _sparkline(history)
        st.markdown(
            f'<span style="font-family:monospace;font-size:14px;">'
            f'<b>MRI Trend</b> '
            f'{history[0]:.2f} {spark} {history[-1]:.2f}'
            f'</span>',
            unsafe_allow_html=True,
        )


def _render_regime_edge(s: Dict[str, Any]) -> None:
    st.markdown("##### Regime Edge (RSD)")
    rsd = s.get("regime_rsd") or {}
    if not rsd:
        st.caption("No regime data yet — need conflicts across multiple regimes.")
        return
    for regime, entry in sorted(rsd.items()):
        if isinstance(entry, dict):
            val = entry.get("rsd", 0.0)
            h_n = entry.get("hydra_n", 0)
            l_n = entry.get("legacy_n", 0)
        else:
            val = float(entry)
            h_n = l_n = "?"
        color = "#21ba45" if val > 0.01 else "#db2828" if val < -0.01 else "#888"
        st.markdown(
            f'<span style="font-family:monospace;font-size:14px;">'
            f'<b>{regime}</b> '
            f'<span style="color:{color};font-weight:bold">{val:+.4f}</span> '
            f'<span style="color:#666;">(hydra={h_n} legacy={l_n})</span>'
            f'</span>',
            unsafe_allow_html=True,
        )
    rdd = s.get("regime_dependence_spread", 0.0)
    if len(rsd) >= 2:
        rdd_color = "#21ba45" if rdd < 0.06 else "#f2c037" if rdd < 0.15 else "#db2828"
        rdd_label = "balanced" if rdd < 0.06 else "specializing" if rdd < 0.15 else "regime-locked"
        st.markdown(
            f'<span style="font-family:monospace;font-size:14px;">'
            f'<b>RDD</b> '
            f'<span style="color:{rdd_color};font-weight:bold">{rdd:.4f}</span> '
            f'<span style="color:#666;">({rdd_label})</span>'
            f'</span>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_multi_engine_panel(state_path: Path | str | None = None) -> None:
    """Render the Multi-Engine Soak Command Center."""
    if not HAS_STREAMLIT:
        return

    st.subheader("⚙️ Multi-Engine Soak")

    state = load_fallback_metrics(state_path)
    if not state:
        st.info("Multi-engine metrics not available — waiting for first conflicts.")
        return

    age = state.get("window_age_s", 0)
    total = state.get("cel_count", 0) + state.get("normal_count", 0)
    st.caption(f"Window: {age / 3600:.1f}h  ·  {total} signals tracked")

    _render_portfolio_edge(state)
    st.markdown("---")
    _render_engine_health(state)
    st.markdown("---")
    _render_calibration(state)
    st.markdown("---")
    _render_regime_edge(state)
    st.markdown("---")
    _render_architecture_status(state)
