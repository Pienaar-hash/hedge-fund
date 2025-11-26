# PATCH SCOPE:
#   Fix ArrowInvalid caused by mixed-type "value" column in the router
#   per-symbol DataFrame used in the v6 Advanced dashboard tab.

# FILES TO EDIT:
#   - dashboard/router_health.py    (preferred fix location)
#   - AND a defensive guard in dashboard/app.py (after df_router creation)

# GOAL:
#   Guarantee that df["value"] is ALWAYS numeric (float), coercing any
#   "95.40"-style strings into float64. Prevents ArrowInvalid in both
#   Altair and st.dataframe, and does NOT affect router logic.

# ----------------------------------------------------------------------------
# 1) Modify dashboard/router_health.py
#    Inside the function that builds per_symbol DF (likely _to_dataframe or
#    build_per_symbol), just after df is created:

    import pandas as pd

    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["value"] = df["value"].fillna(0.0).astype(float)

# Explanation:
#   - Coerces strings like "95.40" into float
#   - Converts None/NaN to 0.0
#   - Forces dtype=float64

# ----------------------------------------------------------------------------
# 2) Add defensive guard in dashboard/app.py
#    Locate where df_router is prepared in the Advanced (v6) tab, typically:

    df_router = router_health.per_symbol.copy()

# Immediately after this line, add:

    import pandas as pd
    if "value" in df_router.columns:
        df_router["value"] = pd.to_numeric(df_router["value"], errors="coerce").fillna(0.0)

# (Optional) cast to float if needed:
        df_router["value"] = df_router["value"].astype(float)

# ----------------------------------------------------------------------------
# 3) NO other changes. Do NOT modify v7 Overview or nav_helpers.

# ----------------------------------------------------------------------------
# VALIDATION:

#   python -m py_compile dashboard/router_health.py dashboard/app.py

#   Restart dashboard:
#       sudo supervisorctl restart hedge:hedge-dashboard

#   Check logs:
#       tail -f /var/log/hedge-dashboard.err.log
#
#   Expected:
#       - ArrowInvalid no longer appears.
#       - Dashboard loads both Overview (v7) and Advanced (v6) cleanly.
#
#   Confirm:
#       - router table displays numeric value column
#       - AUM donut unaffected
#       - v7 metrics fully functional

# END PATCH SCOPE.
