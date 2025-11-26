# PATCH SCOPE: Dashboard v7 Aesthetic Refresh (Investor-Grade UI)

# PURPOSE:
#   1) Beautify "Overview (v7)" using investor-grade layout.
#   2) Remove legacy v6 runtime header from the top of dashboard/app.py.
#   3) Fix truncated KPI text (e.g., “111…”, “nor…”, “unk…”).
#   4) Improve AUM donut readability:
#        - ZAR tooltips (if FX available)
#        - Visible titles & spacing
#        - No empty/blank donut
#   5) Add explicit FX line (USD→ZAR).
#   6) Add NAV (ZAR) + AUM (ZAR) tiles.
#   7) Add “Last updated” footer + freshness badge.
#   8) Polish spacing, alignment, and table rendering.
#
# FILES TO MODIFY:
#   dashboard/app.py
#   dashboard/kpi_panel.py
#   dashboard/nav_helpers.py
#   dashboard/live_helpers.py   (minor: remove legacy caption)
#
# ------------------------------------------------------------------------------
# 1) REMOVE LEGACY V6 HEADER STRIP
#
# In dashboard/app.py:
#   - The legacy v6 panel renders a runtime header near the top of the dashboard.
#   - REMOVE it entirely OR wrap in a collapsible "Advanced (v6 Telemetry)" section
#     that defaults to collapsed.
#
# Implementation:
#
#   - Find the block rendering v6 runtime flags (Intel_v6, Risk_Engine_v6, etc.).
#   - Comment it out OR wrap:
#
#       with st.expander("Advanced (v6 telemetry)", expanded=False):
#           render_runtime_header()
#
#   - For investor view, it must NOT appear at the top automatically.
#
# ------------------------------------------------------------------------------
# 2) FIX TRUNCATED KPI TEXT (COLUMN WIDTH + CSS)
#
# In dashboard/kpi_panel.py and dashboard/app.py:
#
#   Add the following CSS at the top of both modules (before layout code):
#
#       st.markdown("""
#       <style>
#       .stMetric, .css-1ht1j8u, .css-16idsys, [data-testid="stMetricValue"] {
#           overflow: visible !important;
#           text-overflow: initial !important;
#           white-space: nowrap !important;
#       }
#       </style>
#       """, unsafe_allow_html=True)
#
#   Replace KPI columns with wider ones:
#
#       col_nav, col_aum, col_ddatr, col_fee, col_router = st.columns([1.3, 1.3, 1.3, 1.3, 1.3])
#
# ------------------------------------------------------------------------------
# 3) FX RENDERING + ZAR METRICS
#
# In dashboard/app.py inside Overview (v7):
#
#   - After computing nav_total and aum_total, compute:
#
#       nav_zar = nav_total * usd_zar if usd_zar else None
#       aum_zar = aum_total * usd_zar if usd_zar else None
#
#   - Show NAV ZAR + AUM ZAR as secondary metrics:
#
#       if usd_zar:
#           col_fx = st.columns(2)
#           col_fx[0].metric("NAV (ZAR)", f"{nav_zar:,.0f}")
#           col_fx[1].metric("AUM (ZAR)", f"{aum_zar:,.0f}")
#           st.caption(f"FX: 1 USD = {usd_zar:.2f} ZAR (via Coingecko)")
#
# ------------------------------------------------------------------------------
# 4) IMPROVE AUM DONUT (TITLE, TOOLTIP, EMPTY HANDLING)
#
# In dashboard/nav_helpers.py:
#
#   - Ensure build_aum_slices returns keys: label, value, qty, zar, even if zar=None.
#
# In dashboard/app.py:
#
#   - Before rendering donut:
#
#       st.markdown("### AUM Breakdown (USD)")
#       st.markdown("#### Hover for ZAR values")
#
#   - Modify Altair tooltip:
#
#       tooltip=["label", "value", "qty", "zar"]
#
#   - If slices empty:
#
#       if not slices or sum(x["value"] or 0 for x in slices) <= 0:
#           st.info("AUM data unavailable or zero.")
#       else:
#           st.altair_chart(...)
#
# ------------------------------------------------------------------------------
# 5) DATA FRESHNESS + LAST UPDATED FOOTER
#
# In dashboard/app.py:
#
#   - Compute age using snapshot_age_seconds(kpis_overview).
#
#   - Add freshness badge:
#
#       if age < 60:
#           st.success(f"Data age: {age:.0f}s")
#       elif age < 300:
#           st.warning(f"Data age: {age:.0f}s")
#       else:
#           st.error(f"Data age: {age:.0f}s (stale)")
#
#   - Add last updated caption:
#
#       ts = kpis_overview.get("ts") or nav_v7.get("ts")
#       if ts:
#           st.caption(f"Last updated (UTC): {ts}")
#
# ------------------------------------------------------------------------------
# 6) TOP METRIC BAND IMPROVEMENTS
#
# In dashboard/app.py, inside Overview (v7):
#
#   Replace existing three metrics with:
#
#       col_nav, col_aum, col_ddatr = st.columns([1.3, 1.3, 1.3])
#       col_nav.metric("NAV (USD)", f"{nav_total:,.2f}")
#       col_aum.metric("AUM (USD)", f"{aum_total:,.2f}")
#       col_ddatr.metric("DD/ATR", f"{dd_state} / {atr_regime}")
#
#   Below them, add FX metrics (see section 3).
#
# ------------------------------------------------------------------------------
# 7) SECONDARY KPI BAND
#
# Under the AUM donut, add:
#
#   - Drawdown %  
#   - ATR ratio (if available)  
#   - Router maker_fill / fallback  
#
# Example:
#
#       col1, col2, col3 = st.columns(3)
#       col1.metric("Drawdown %", f"{dd_pct:.3f}%")
#       col2.metric("ATR Ratio", f"{atr_ratio:.3f}")
#       col3.metric("Router Fill", f"{maker_fill_ratio:.2f}")
#
# ------------------------------------------------------------------------------
# 8) POSITIONS TABLE REFINEMENT
#
# In dashboard/app.py:
#
#   - Replace the existing table with a cleaned DataFrame:
#
#       df = pd.DataFrame(items)
#       df = df[["symbol", "side", "qty", "notional", "pnl"]]
#       st.dataframe(df, use_container_width=True)
#
#   - If no positions:
#
#       st.info("No open positions.")
#
# ------------------------------------------------------------------------------
# 9) SPACING / VISUAL POLISH
#
# Add strategic spacing:
#
#       st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
#
# Between major sections:
#   - After top metrics
#   - After AUM donut
#   - Before footer
#
# ------------------------------------------------------------------------------
# 10) LEGACY TABS MARKED AS ADVANCED
#
# In the tab creation line:
#
#   Replace:
#       tabs = st.tabs(...)
#
#   With:
#       tabs = st.tabs(["Overview (v7)", "Advanced (v6)"])
#
# Inside "Advanced (v6)":
#
#       st.caption("Legacy v6 telemetry — visible for internal diagnostics only.")
#
# ------------------------------------------------------------------------------
# VALIDATION / TESTING INSTRUCTIONS
#
# 1. python -m py_compile dashboard/*.py execution/state_publish.py
# 2. Restart executor + sync_state so kpis_v7 contains FX.
# 3. Start Streamlit: streamlit run dashboard/app.py
# 4. Confirm:
#       • v6 header is hidden or collapsed.
#       • NAV (USD/ZAR), AUM (USD/ZAR) visible.
#       • FX caption visible.
#       • Donut shows all slices (BTC, XAUT, USDC, Futures) with ZAR tooltip.
#       • KPI text no longer truncated.
#       • “Last updated” + freshness badge visible.
#       • Positions table clean, wide, readable.
#       • Legacy tabs collected under "Advanced (v6)".
#
# END PATCH SCOPE.
