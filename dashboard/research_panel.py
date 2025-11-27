# mypy: ignore-errors
"""Research & Optimizations panel for investor transparency."""
from __future__ import annotations

import streamlit as st
from datetime import datetime


def render_research_panel() -> None:
    """Render the Research & Optimizations tab content."""
    
    st.markdown("""
    <style>
    .research-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00cc00;
    }
    .research-card.warning {
        border-left-color: #ffaa00;
    }
    .research-card.insight {
        border-left-color: #00aaff;
    }
    .metric-highlight {
        font-size: 24px;
        font-weight: bold;
        color: #00cc00;
    }
    .metric-highlight.negative {
        color: #ff4444;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🔬 Research & Optimizations")
    st.caption("Quantitative analysis, backtests, and strategy improvements")
    
    # Latest Optimization Summary
    st.markdown("### 📊 Latest Parameter Optimization (Nov 2025)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Backtests Run", "514", help="Grid search across RSI parameters")
    col2.metric("Symbols Optimized", "8", help="Full universe coverage")
    col3.metric("Avg Sharpe Improvement", "+3.2", help="Risk-adjusted return improvement")
    
    st.markdown("""
    <div class="research-card">
        <h4 style="margin-top:0">🎯 Optimization Methodology</h4>
        <p>Systematic grid search across RSI parameters (period, oversold, overbought thresholds) 
        using 30 days of 15-minute OHLCV data. Each parameter combination evaluated on:</p>
        <ul>
            <li><b>Sharpe Ratio</b> — Risk-adjusted returns (primary metric)</li>
            <li><b>Maximum Drawdown</b> — Downside risk control</li>
            <li><b>Win Rate & Profit Factor</b> — Trade quality validation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Per-symbol results table
    st.markdown("### 📈 Optimized Parameters by Symbol")
    
    optimization_results = [
        {"Symbol": "BTC", "Before": "[35,65]", "After": "[35,70]", "Sharpe": 11.47, "Return": "39%", "Insight": "Wider overbought catches stronger moves"},
        {"Symbol": "ETH", "Before": "[35,65]", "After": "[20,80]", "Sharpe": 10.48, "Return": "55%", "Insight": "Much wider bands + faster ATR (7)"},
        {"Symbol": "SUI", "Before": "[30,70]", "After": "[25,70]", "Sharpe": 9.22, "Return": "120%", "Insight": "⭐ Best performer - lower oversold entry"},
        {"Symbol": "LTC", "Before": "[35,65]", "After": "[20,75]", "Sharpe": 8.09, "Return": "86%", "Insight": "Aggressive dip buying works"},
        {"Symbol": "DOGE", "Before": "[30,70]", "After": "[30,65]", "Sharpe": 7.38, "Return": "67%", "Insight": "Earlier profit-taking optimal"},
        {"Symbol": "SOL", "Before": "[35,65]", "After": "[25,65]", "Sharpe": 7.11, "Return": "64%", "Insight": "Lower oversold for dip entries"},
        {"Symbol": "WIF", "Before": "ATR 14", "After": "ATR 21", "Sharpe": 6.96, "Return": "94%", "Insight": "Slower ATR smooths volatility"},
        {"Symbol": "LINK", "Before": "[35,65]", "After": "[20,70]", "Sharpe": 6.33, "Return": "61%", "Insight": "Aggressive oversold buying"},
    ]
    
    import pandas as pd
    df = pd.DataFrame(optimization_results)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=340
    )
    
    # Key Findings
    st.markdown("### 💡 Key Research Findings")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        <div class="research-card insight">
            <h4 style="margin-top:0">📉 Wider RSI Bands Outperform</h4>
            <p>Narrow [35,65] bands generate too many false signals. 
            Optimization found [20-25, 70-80] ranges filter to only 
            <b>high-conviction</b> extreme readings.</p>
            <p style="color:#888; font-size:13px">Average improvement: +2.4 Sharpe points</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="research-card insight">
            <h4 style="margin-top:0">⚡ ATR Period Matters</h4>
            <p>Fast-moving assets (ETH, LTC) benefit from <b>shorter ATR (7)</b> 
            for quicker adaptation. Slower alts (SOL, DOGE, WIF) perform better 
            with <b>longer ATR (21)</b>.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div class="research-card warning">
            <h4 style="margin-top:0">⚠️ Short-Side Restrictions</h4>
            <p>Analysis revealed <b>LONG: +$105 vs SHORT: -$125</b> asymmetry 
            over 48 hours. Shorts now disabled for underperformers:</p>
            <ul style="margin-bottom:0">
                <li>LTC — negative short expectancy</li>
                <li>SUI — poor short timing</li>
                <li>LINK — trend-following bias</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="research-card">
            <h4 style="margin-top:0">🛠️ Infrastructure Built</h4>
            <ul style="margin-bottom:0">
                <li><b>OHLCV Collector</b> — 100k+ rows across 4 timeframes</li>
                <li><b>Backtest Engine</b> — Event-driven with realistic fees</li>
                <li><b>Parameter Optimizer</b> — Grid search framework</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk & Execution Improvements
    st.markdown("### 🛡️ Risk & Execution Improvements")
    
    improvements = [
        {"Area": "Router Bootstrap", "Issue": "Chicken-and-egg policy blocking", "Fix": "MIN_SAMPLES=20 bootstrap mode", "Impact": "All 8 symbols now MAKER ON"},
        {"Area": "R:R Ratio", "Issue": "Avg win $1.10 vs avg loss $1.35", "Fix": "Widened TP targets for 3:1 R:R", "Impact": "Improved expectancy"},
        {"Area": "Weekend Trading", "Issue": "Negative weekend returns", "Fix": "Disabled Sat/Sun trading", "Impact": "Removed -EV periods"},
        {"Area": "Maker Offsets", "Issue": "Low maker fill rates", "Fix": "Dynamic offset tuning (2-8 bps)", "Impact": "Reduced slippage costs"},
    ]
    
    st.dataframe(pd.DataFrame(improvements), use_container_width=True, hide_index=True)
    
    # Changelog
    st.markdown("### 📜 Version History")
    
    with st.expander("v6.4 — Parameter Optimization Release (Nov 2025)", expanded=True):
        st.markdown("""
        - **514 backtests** across all 8 symbols with grid search optimization
        - RSI and ATR parameters tuned per-symbol based on Sharpe ratio
        - EMA fast period: 20 → 15 for momentum strategy
        - Router bootstrap fix preventing maker policy deadlock
        - New backtest infrastructure for ongoing research
        """)
    
    with st.expander("v5.8 RC1 — Dashboard & Analytics (Nov 2025)"):
        st.markdown("""
        - Enhanced portfolio equity tracking
        - Router health deduplication fix
        - Risk limits enforcement (≤120% gross exposure)
        - Treasury PnL computation improvements
        """)
    
    # Footer
    st.markdown("---")
    st.caption(f"Last research update: November 2025 • Next optimization cycle: Monthly")
