"""v6 intel panel."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def _clamp_01(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def render_intel_panel(
    expectancy_v6: Dict[str, Any],
    symbol_scores_v6: Dict[str, Any],
    risk_allocator_v6: Dict[str, Any],
) -> None:
    """
    v6 Intel panel with normalized/clamped scores.
    """
    st.subheader("Intel v6 — Symbol Scores & Expectancy")

    scores_list = symbol_scores_v6.get("symbols") if isinstance(symbol_scores_v6, dict) else None
    if not isinstance(scores_list, list):
        scores_list = []

    exp_map = expectancy_v6.get("symbols") if isinstance(expectancy_v6, dict) else {}
    if not isinstance(exp_map, dict):
        exp_map = {}

    rows: List[Dict[str, Any]] = []
    for entry in scores_list:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        sym = str(symbol).upper()

        exp_entry = exp_map.get(sym) if isinstance(exp_map.get(sym), dict) else {}
        expectancy_raw = (exp_entry or {}).get("expectancy")
        hit_raw = (exp_entry or {}).get("hit_rate")
        dd_raw = (exp_entry or {}).get("max_drawdown")

        rows.append(
            {
                "symbol": sym,
                "score_raw": _safe_float(entry.get("score")),
                "score": _clamp_01(entry.get("score")),
                "expectancy_raw": _safe_float(expectancy_raw),
                "expectancy": _clamp_01(expectancy_raw),
                "hit_rate_raw": _safe_float(hit_raw),
                "hit_rate": _clamp_01(hit_raw),
                "max_drawdown": _safe_float(dd_raw),
            }
        )

    if not rows:
        st.info("No v6 intel snapshots yet — wait for INTEL_V6 to publish.")
        return

    df = pd.DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True, ignore_index=True)

    display_cols = [
        "symbol",
        "score",
        "expectancy",
        "hit_rate",
        "max_drawdown",
        "score_raw",
        "expectancy_raw",
        "hit_rate_raw",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.markdown("### Top Symbols (normalized to 0–1)")
    st.dataframe(df[display_cols].head(50), use_container_width=True, height=360)

    st.markdown("---")
    st.subheader("Risk Allocator v6 — Suggestions")

    if not isinstance(risk_allocator_v6, dict) or not risk_allocator_v6:
        st.caption("No v6 allocator suggestions yet.")
        return

    alloc_symbols = risk_allocator_v6.get("symbols")
    allocations = risk_allocator_v6.get("allocations")
    generated_ts = risk_allocator_v6.get("generated_ts") or risk_allocator_v6.get("ts")

    if isinstance(alloc_symbols, list):
        alloc_rows: List[Dict[str, Any]] = []
        for entry in alloc_symbols:
            if not isinstance(entry, dict):
                continue
            sym = str(entry.get("symbol") or "").upper()
            if not sym:
                continue
            row = {"symbol": sym}
            for key in ("weight", "target_notional", "max_exposure_pct", "dd_state"):
                if key in entry:
                    row[key] = entry.get(key)
            alloc_rows.append(row)
        if alloc_rows:
            st.dataframe(pd.DataFrame(alloc_rows), use_container_width=True, height=260)
        else:
            st.json(risk_allocator_v6)
    elif isinstance(allocations, dict):
        st.json(allocations)
    else:
        st.json(risk_allocator_v6)

    if generated_ts:
        st.caption(f"Allocator snapshot ts={generated_ts}")


def render_hybrid_scores_panel(
    hybrid_scores: Dict[str, Any],
) -> None:
    """
    Render hybrid score rankings from screener intents (v7.4 B1).
    
    Displays the blended trend/carry/expectancy/router scores with component breakdown.
    """
    st.subheader("Hybrid Score Rankings (v7.4)")
    
    if not isinstance(hybrid_scores, dict):
        st.info("No hybrid scores available yet.")
        return
    
    symbols_list = hybrid_scores.get("symbols")
    if not isinstance(symbols_list, list) or not symbols_list:
        st.info("No hybrid score data available.")
        return
    
    rows: List[Dict[str, Any]] = []
    for entry in symbols_list:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        
        components = entry.get("components", {})
        weights = entry.get("weights", {})
        
        rows.append({
            "symbol": str(symbol).upper(),
            "direction": entry.get("direction", ""),
            "hybrid_score": _clamp_01(entry.get("hybrid_score")),
            "passes_threshold": entry.get("passes_threshold", True),
            "trend": _clamp_01(components.get("trend")),
            "carry": _clamp_01(components.get("carry")),
            "expectancy": _clamp_01(components.get("expectancy")),
            "router": _clamp_01(components.get("router")),
            "trend_w": _safe_float(weights.get("trend")),
            "carry_w": _safe_float(weights.get("carry")),
            "regime": entry.get("regime"),
            "regime_mult": _safe_float(entry.get("regime_multiplier")),
        })
    
    if not rows:
        st.info("No hybrid score entries to display.")
        return
    
    df = pd.DataFrame(rows)
    df.sort_values("hybrid_score", ascending=False, inplace=True, ignore_index=True)
    
    # Add visual indicator for threshold pass/fail
    def _threshold_indicator(passes: bool) -> str:
        return "✅" if passes else "⚠️"
    
    if "passes_threshold" in df.columns:
        df["pass"] = df["passes_threshold"].apply(_threshold_indicator)
    
    display_cols = [
        "symbol",
        "direction",
        "hybrid_score",
        "pass",
        "trend",
        "carry",
        "expectancy",
        "router",
        "regime",
        "regime_mult",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    
    st.markdown("### Ranked by Hybrid Score (descending)")
    st.dataframe(df[display_cols].head(30), use_container_width=True, height=360)
    
    # Show weight configuration
    if rows:
        sample = rows[0]
        weights = {k: sample.get(k) for k in ["trend_w", "carry_w"] if sample.get(k) is not None}
        if weights:
            st.caption(f"Weights: trend={weights.get('trend_w', 0):.0%}, carry={weights.get('carry_w', 0):.0%}")
    
    updated_ts = hybrid_scores.get("updated_ts")
    if updated_ts:
        st.caption(f"Last updated: {updated_ts}")


def render_carry_scores_panel(
    funding_snapshot: Dict[str, Any],
    basis_snapshot: Dict[str, Any],
) -> None:
    """
    Render carry score components (funding rates and basis) (v7.4 B1).
    """
    st.subheader("Carry Components — Funding & Basis")
    
    funding_symbols = funding_snapshot.get("symbols", {})
    basis_symbols = basis_snapshot.get("symbols", {})
    
    if not funding_symbols and not basis_symbols:
        st.info("No funding/basis data available.")
        return
    
    all_symbols = sorted(set(funding_symbols.keys()) | set(basis_symbols.keys()))
    
    rows: List[Dict[str, Any]] = []
    for sym in all_symbols:
        funding_data = funding_symbols.get(sym, {})
        basis_data = basis_symbols.get(sym, {})
        
        # Handle both scalar and dict formats
        if isinstance(funding_data, (int, float)):
            funding_rate = float(funding_data)
        else:
            funding_rate = float(funding_data.get("rate", 0.0) if isinstance(funding_data, dict) else 0.0)
        
        if isinstance(basis_data, (int, float)):
            basis_pct = float(basis_data)
        else:
            basis_pct = float(basis_data.get("basis_pct", 0.0) if isinstance(basis_data, dict) else 0.0)
        
        # Annualize funding rate (8h rate * 3 * 365)
        annual_funding = funding_rate * 1095
        
        rows.append({
            "symbol": str(sym).upper(),
            "funding_8h": f"{funding_rate:.4%}" if funding_rate else "—",
            "funding_ann": f"{annual_funding:.1%}" if funding_rate else "—",
            "basis_pct": f"{basis_pct:.2%}" if basis_pct else "—",
            "long_bias": "+" if (funding_rate < 0 or basis_pct < 0) else "−",
            "short_bias": "+" if (funding_rate > 0 or basis_pct > 0) else "−",
        })
    
    if not rows:
        st.info("No carry data to display.")
        return
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, height=300)
    
    st.caption("+/− indicates directional bias from funding/basis")


def render_vol_regimes_panel(
    vol_regimes: Dict[str, Any],
) -> None:
    """
    Render volatility regime information for each symbol (v7.4 B2).
    
    Displays the EWMA-based volatility regime classification with
    short/long vol and ratio.
    """
    st.subheader("Volatility Regimes (v7.4)")
    
    if not isinstance(vol_regimes, dict):
        st.info("No volatility regime data available.")
        return
    
    # Show summary bar
    summary = vol_regimes.get("vol_regime_summary", {})
    if summary:
        cols = st.columns(4)
        regime_colors = {
            "low": "🔵",
            "normal": "🟢",
            "high": "🟠",
            "crisis": "🔴",
        }
        for i, (regime, count) in enumerate(summary.items()):
            color = regime_colors.get(regime, "⚪")
            cols[i].metric(f"{color} {regime.title()}", count)
    
    symbols_list = vol_regimes.get("symbols")
    if not isinstance(symbols_list, list) or not symbols_list:
        st.info("No per-symbol volatility data available.")
        return
    
    rows: List[Dict[str, Any]] = []
    for entry in symbols_list:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if not symbol:
            continue
        
        vol_data = entry.get("vol", {})
        regime = entry.get("vol_regime", "normal")
        
        # Badge for regime
        badge = {
            "low": "🔵 L",
            "normal": "🟢 N",
            "high": "🟠 H",
            "crisis": "🔴 C",
        }.get(regime, "⚪ ?")
        
        rows.append({
            "symbol": str(symbol).upper(),
            "regime": badge,
            "vol_short": _safe_float(vol_data.get("short")),
            "vol_long": _safe_float(vol_data.get("long")),
            "ratio": _safe_float(vol_data.get("ratio")),
        })
    
    if not rows:
        st.info("No volatility regime entries to display.")
        return
    
    df = pd.DataFrame(rows)
    df.sort_values("ratio", ascending=False, inplace=True, ignore_index=True)
    
    # Format vol columns as percentages
    for col in ["vol_short", "vol_long"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4%}" if x else "—")
    
    if "ratio" in df.columns:
        df["ratio"] = df["ratio"].apply(lambda x: f"{x:.2f}" if x else "—")
    
    st.dataframe(df, use_container_width=True, height=300)
    
    st.caption("L=Low, N=Normal, H=High, C=Crisis | Ratio = short_vol / long_vol")
    
    updated_ts = vol_regimes.get("updated_ts")
    if updated_ts:
        st.caption(f"Last updated: {updated_ts}")


def render_rv_momentum_panel(
    rv_momentum: Dict[str, Any],
) -> None:
    """
    Render RV momentum scores and basket spreads (v7.5_C1).
    
    Displays:
    - Per-symbol RV scores
    - Basket membership
    - Basket spread values (BTC vs ETH, L1 vs ALT, Meme vs Rest)
    """
    st.subheader("Relative Momentum (RV-MOMO) v7.5_C1")
    
    if not isinstance(rv_momentum, dict) or not rv_momentum:
        st.info("No RV momentum data available yet.")
        return
    
    # Show basket spreads in columns
    spreads = rv_momentum.get("spreads", {})
    if spreads:
        st.markdown("#### Basket Spreads")
        cols = st.columns(3)
        
        def spread_color(val: float) -> str:
            if val > 0.001:
                return "🟢"  # Positive spread
            elif val < -0.001:
                return "🔴"  # Negative spread
            return "⚪"  # Neutral
        
        btc_eth = float(spreads.get("btc_vs_eth", 0.0))
        l1_alt = float(spreads.get("l1_vs_alt", 0.0))
        meme_rest = float(spreads.get("meme_vs_rest", 0.0))
        
        with cols[0]:
            st.metric(
                f"{spread_color(btc_eth)} BTC vs ETH",
                f"{btc_eth:.4f}",
                help="Positive = BTC outperforming ETH"
            )
        with cols[1]:
            st.metric(
                f"{spread_color(l1_alt)} L1 vs ALT",
                f"{l1_alt:.4f}",
                help="Positive = L1 basket outperforming ALT basket"
            )
        with cols[2]:
            st.metric(
                f"{spread_color(meme_rest)} Meme vs Rest",
                f"{meme_rest:.4f}",
                help="Positive = Meme basket outperforming rest"
            )
    
    # Per-symbol table
    st.markdown("#### Per-Symbol RV Scores")
    
    per_symbol = rv_momentum.get("per_symbol", {})
    if not per_symbol:
        st.info("No per-symbol RV data available.")
        return
    
    rows: List[Dict[str, Any]] = []
    for symbol, data in per_symbol.items():
        if not isinstance(data, dict):
            continue
        
        score = _safe_float(data.get("score"))
        raw_score = _safe_float(data.get("raw_score"))
        baskets = data.get("baskets", [])
        
        # Score badge
        if score is not None:
            if score > 0.3:
                badge = "🟢"
            elif score < -0.3:
                badge = "🔴"
            else:
                badge = "⚪"
        else:
            badge = "⚫"
        
        rows.append({
            "symbol": str(symbol).upper(),
            "rv_score": score,
            "badge": badge,
            "baskets": ", ".join(baskets) if baskets else "REST",
            "raw_score": raw_score,
        })
    
    if not rows:
        st.info("No RV momentum entries to display.")
        return
    
    df = pd.DataFrame(rows)
    df.sort_values("rv_score", ascending=False, inplace=True, ignore_index=True)
    
    # Display with badge column
    display_df = df[["badge", "symbol", "rv_score", "baskets", "raw_score"]].copy()
    display_df.columns = ["", "Symbol", "RV Score", "Baskets", "Raw Score"]
    
    st.dataframe(display_df, use_container_width=True, height=300)
    
    st.caption("RV Score range: [-1, 1]. Positive = relative strength, Negative = relative weakness")
    
    updated_ts = rv_momentum.get("updated_ts")
    if updated_ts:
        st.caption(f"Last updated: {updated_ts}")


__all__ = ["render_intel_panel", "render_hybrid_scores_panel", "render_carry_scores_panel", "render_vol_regimes_panel", "render_rv_momentum_panel"]
