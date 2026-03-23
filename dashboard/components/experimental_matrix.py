# mypy: ignore-errors
"""
Experimental Matrix Panel — S2 PM + Futures S2 Proxy unified surface.

Displays both experiments side-by-side as a single hypothesis test:
  "Does the PM probability model have tradeable edge, and does it transfer?"

Data sources:
  - logs/state/binary_lab_s2_state.json  (S2 PM paper trade)
  - logs/state/futures_s2_proxy_state.json  (Futures proxy)
  - logs/execution/futures_s2_proxy_trades.jsonl  (for transfer signal)

Auto-hides when both state files are empty/missing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_S2_STATE_PATH = Path("logs/state/binary_lab_s2_state.json")
_FSP_STATE_PATH = Path("logs/state/futures_s2_proxy_state.json")
_FSP_TRADES_PATH = Path("logs/execution/futures_s2_proxy_trades.jsonl")


# ---------------------------------------------------------------------------
# State Loaders
# ---------------------------------------------------------------------------

def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}


def _safe_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    results: List[Dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))
    except (json.JSONDecodeError, IOError):
        pass
    return results


def load_experimental_matrix_state(
    s2_path: Optional[Path] = None,
    fsp_path: Optional[Path] = None,
    fsp_trades_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load all state for the experimental matrix panel."""
    s2_p = s2_path or _S2_STATE_PATH
    fsp_p = fsp_path or _FSP_STATE_PATH
    s2 = _safe_json(s2_p)
    fsp = _safe_json(fsp_p)
    trades = _safe_jsonl(fsp_trades_path or _FSP_TRADES_PATH)
    return {
        "s2": s2,
        "fsp": fsp,
        "fsp_trades": trades,
        "s2_file_exists": s2_p.exists(),
        "fsp_file_exists": fsp_p.exists(),
    }


# ---------------------------------------------------------------------------
# Transfer Signal Computation
# ---------------------------------------------------------------------------

def _compute_transfer_signal(
    trades: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Compute corr(edge, return) from closed proxy trades. Needs n>=30."""
    closed = [t for t in trades if t.get("event_type") == "ROUND_CLOSED"]
    edges: List[float] = []
    returns: List[float] = []
    for t in closed:
        snap = t.get("signal_snapshot", {})
        edge = snap.get("edge")
        lr = t.get("log_return")
        if edge is not None and lr is not None:
            edges.append(float(edge))
            returns.append(float(lr))

    n = len(edges)
    if n < 30:
        return {"status": "insufficient", "n_closed": len(closed), "n_paired": n}

    mean_e = sum(edges) / n
    mean_r = sum(returns) / n
    cov = sum((e - mean_e) * (r - mean_r) for e, r in zip(edges, returns)) / n
    std_e = math.sqrt(sum((e - mean_e) ** 2 for e in edges) / n)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in returns) / n)
    corr = cov / (std_e * std_r) if std_e > 0 and std_r > 0 else 0.0

    return {
        "status": "ready",
        "corr": corr,
        "n": n,
        "mean_edge": mean_e,
        "mean_return": mean_r,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _status_color(status: str) -> str:
    s = status.upper()
    if s == "ACTIVE":
        return "#22c55e"
    if s in ("DISABLED", "KILLED"):
        return "#ff1744"
    if s == "COMPLETED":
        return "#2196f3"
    return "#888"


def _badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color}20;color:{color};'
        f'padding:2px 8px;border-radius:4px;font-weight:600;'
        f'font-size:0.8em;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_experimental_matrix(
    state: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the unified Experimental Matrix panel.

    Auto-hides when no experiment data exists.
    """
    if state is None:
        state = load_experimental_matrix_state()

    s2 = state.get("s2", {})
    fsp = state.get("fsp", {})
    trades = state.get("fsp_trades", [])
    s2_exists = state.get("s2_file_exists", bool(s2))
    fsp_exists = state.get("fsp_file_exists", bool(fsp))

    # Auto-hide: no experiments deployed
    if not s2 and not s2_exists and not fsp and not fsp_exists:
        return

    st.markdown("### Experimental Matrix")
    st.caption("PM edge hypothesis — binary market test + futures transfer")

    # ── Sub-A: S2 PM Paper Trade ──
    if s2:
        status = s2.get("status", "UNKNOWN")
        mode = s2.get("mode", "")
        metrics = s2.get("metrics", {})
        capital = s2.get("capital", {})
        n_trades = metrics.get("total_trades", 0)
        wins = metrics.get("wins", 0)
        losses = metrics.get("losses", 0)
        pnl = float(capital.get("pnl_usd", 0) or 0)
        nav = float(capital.get("current_nav_usd", 0) or 0)
        wr_raw = float(metrics.get("win_rate", 0) or 0)
        wr = wr_raw * 100 if wr_raw < 1 else wr_raw
        kill_breached = s2.get("kill_line", {}).get("breached", False)
        kill_dist = float(s2.get("kill_line", {}).get("distance_usd", 0) or 0)
        day = s2.get("day", "?")
        day_total = s2.get("day_total", "?")
        freeze = s2.get("freeze_intact", None)

        label = f"{status} ({mode})" if mode else status
        color = _status_color(status)

        st.markdown(
            f"**S2 PM Paper Trade** &nbsp; {_badge(label, color)}"
            f" &nbsp; Day {day}/{day_total}",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rounds", f"{n_trades}", f"W:{wins} / L:{losses}")
        c2.metric("PnL", f"${pnl:,.2f}")
        c3.metric("Win Rate", f"{wr:.1f}%")
        c4.metric("Kill Line", "BREACHED" if kill_breached else f"${kill_dist:,.0f} away")

        # Band breakdown
        bands = metrics.get("by_conviction_band", {})
        if bands:
            with st.expander("Edge Band Breakdown", expanded=False):
                rows = []
                for name, bd in sorted(bands.items()):
                    rows.append({
                        "Band": name,
                        "Trades": bd.get("trades", 0),
                        "PnL ($)": f"{float(bd.get('pnl_usd', 0) or 0):,.2f}",
                        "EV ($)": f"{float(bd.get('ev_usd', 0) or 0):,.2f}",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

        # Compact sub-metrics
        sub_parts = []
        if nav:
            sub_parts.append(f"Capital: ${nav:,.2f}")
        if freeze is not None:
            sub_parts.append(f"Freeze: {'✓' if freeze else '✗'}")
        if sub_parts:
            st.caption(" · ".join(sub_parts))
    else:
        st.markdown("**S2 PM Paper Trade** — not deployed")

    st.divider()

    # ── Sub-B: Futures S2 Proxy ──
    fsp_entries = fsp.get("total_entries", 0)
    fsp_exits = fsp.get("total_exits", 0)
    if fsp and (fsp_entries > 0 or fsp.get("dd_kill_active")):
        cum_pnl = float(fsp.get("cumulative_pnl", 0) or 0)
        fsp_wr = float(fsp.get("realized_win_rate", 0) or 0)
        mlr = float(fsp.get("mean_log_return", 0) or 0)
        dd_kill = fsp.get("dd_kill_active", False)
        open_count = len(fsp.get("open_trades", []))

        st.markdown(
            f"**Futures S2 Proxy** &nbsp;"
            f"{_badge('DD KILL' if dd_kill else 'LIVE', '#ff1744' if dd_kill else '#22c55e')}",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entries", str(fsp_entries), f"Open: {open_count}")
        c2.metric("Exits", str(fsp_exits))
        c3.metric("Cum. PnL", f"${cum_pnl:,.4f}")
        wr_label = f"{fsp_wr * 100:.1f}%" if fsp_exits > 0 else "N/A"
        c4.metric("Win Rate", wr_label)

        if fsp_exits > 0:
            st.caption(f"Mean Log Return: {mlr:+.8f}")
    elif fsp or fsp_exists:
        st.markdown("**Futures S2 Proxy** — deployed, awaiting first trade")
    else:
        st.markdown("**Futures S2 Proxy** — not deployed")

    # ── Sub-C: Transfer Signal ──
    signal = _compute_transfer_signal(trades)
    if signal and signal["status"] == "ready":
        st.divider()
        corr = signal["corr"]
        if corr > 0.1:
            interp = "Positive transfer"
            interp_color = "#22c55e"
        elif corr < -0.1:
            interp = "Inverse relationship"
            interp_color = "#ff1744"
        else:
            interp = "No significant transfer"
            interp_color = "#888"

        st.markdown(
            f"**Transfer Signal** &nbsp; {_badge(interp, interp_color)}",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("corr(edge, return)", f"{corr:+.4f}")
        c2.metric("Mean Edge", f"{signal['mean_edge']:+.4f}")
        c3.metric("Paired Samples", str(signal["n"]))
    elif signal and signal["n_closed"] > 0:
        st.divider()
        st.caption(
            f"Transfer Signal: {signal['n_paired']} paired samples"
            f" (need 30, have {signal['n_closed']} closed trades)"
        )
