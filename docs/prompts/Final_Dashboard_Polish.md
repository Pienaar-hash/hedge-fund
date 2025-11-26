# PATCH SCOPE 2
# Final Dashboard Polish (Overview v7 + Safe Advanced Mode)

# FILES:
#   - dashboard/app.py
#   - dashboard/nav_helpers.py
#   - dashboard/kpi_panel.py
#   - dashboard/live_helpers.py (cast cleanups)
#
# ---------------------------------------------------------------------------
# 1) Force Overview (v7) to load ONLY nav.json

# In dashboard/nav_helpers.py:
# Modify load_nav_with_aum():

    # REMOVE any fallback to:
    #   nav_state.json
    #   nav_confirmed.json
    #   live/cache nav files
    #
    # KEEP:
    path = os.path.join(state_dir, "nav.json")
    # If file missing, show st.error in app layer, but do not fallback.

# ---------------------------------------------------------------------------
# 2) Update Overview header to show correct source

# In dashboard/app.py (inside the Overview (v7) tab):

    st.caption("Source: nav.json (v7 snapshot)")

# REMOVE any UI elements referring to nav_state.json or legacy v6 runtime headers.

# ---------------------------------------------------------------------------
# 3) KPI Panel Polish (safe formatting everywhere)

# In dashboard/kpi_panel.py:
# Ensure ALL st.metric() calls use safe_format(x)

    nav_fmt = safe_format(nav_usd)
    aum_fmt = safe_format(aum_usd)
    dd_fmt  = safe_format(dd_pct, nd=3)

# Add FX label formatting:

    if usd_zar:
        st.caption(f"FX: 1 USD ≈ {float(usd_zar):.2f} ZAR")

# ---------------------------------------------------------------------------
# 4) AUM Donut: consistent titles + spacing

# In dashboard/app.py (Overview v7 section):
    st.markdown("### AUM Breakdown (USD)")
    st.markdown("##### Hover for ZAR values")

# After chart, add spacing:
    st.markdown("<div style='margin-top: 1.2em;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 5) Harden v6 “Advanced” tabs (prevent ArrowInvalid in router/history tables)

# In dashboard/live_helpers.py:
# For any DataFrame df used in Advanced tabs, before returning:

    for col in df.columns:
        # If numeric-ish column, safe_cast
        if col.lower() in ["value", "qty", "pnl", "notional", "ratio", "price"]:
            df[col] = df[col].apply(lambda x: safe_float(x))

    # OR force string if semantic:
    if "value" in df.columns and df["value"].dtype == object and df["value"].apply(lambda x: isinstance(x, str)).any():
        df["value"] = df["value"].astype(str)

# ---------------------------------------------------------------------------
# 6) Typography polish

# In dashboard/app.py, before layout:

    st.markdown("""
    <style>
    .metric-title { font-size: 0.9rem !important; font-weight: 600 !important; }
    .metric-value { font-size: 1.3rem !important; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 7) Footer

# End of Overview (v7) tab:

    st.markdown("---")
    st.caption("Hedge — v7 Portfolio Telemetry • Updated automatically via executor_live + sync_state")

# END PATCH SCOPE 2
