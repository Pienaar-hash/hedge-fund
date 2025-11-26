# PATCH SCOPE 2: Eliminate remaining ArrowInvalid sources in dashboard (mixed str/float "value" columns)

# Files:
#   - dashboard/app.py
#   - dashboard/live_helpers.py
#   - dashboard/kpi_panel.py  (verify we didn't miss anything)
#
# GOAL:
#   Ensure no DataFrame passed to Streamlit contains mixed-type numeric
#   columns (e.g., 'value' containing both "95.60" and 95.60).
#
# APPROACH:
#   - Use safe_float / safe_format helpers globally across all DataFrames.
#   - For any non-critical numeric column (e.g., historical metric "value"),
#     either:
#       * cast to float via safe_float, OR
#       * cast everything to string before st.dataframe.
#
# STEPS:
#
# 1) Search for all st.dataframe / st.table calls in:
#       dashboard/app.py
#       dashboard/live_helpers.py
#
#    and examine the DataFrame construction preceding them.
#
# 2) For each DataFrame `df` that contains a numeric column named "value"
#    or any numeric column that may mix strings and numbers:
#
#   - If the column is logically numeric (percentages, amounts, ratios),
#     cast via safe_float:
#
#         for col in ["value", "qty", "pnl", "notional", "dd_pct", "ratio"]:
#             if col in df.columns:
#                 df[col] = df[col].apply(safe_float)
#
#   - If the column is more like a generic metric label that can be shown
#     as text (not important for calculations), then cast whole column to
#     string:
#
#         if "value" in df.columns and not strictly numeric:
#             df["value"] = df["value"].astype(str)
#
# 3) In particular, check any advanced/legacy panels that build:
#       df = pd.DataFrame(history)
#       df["value"] = ...
#
#    and enforce safe_float or .astype(str) as appropriate.
#
# 4) For router or pipeline panels with complex numeric data:
#
#   - For numeric metrics: use safe_float then safe_format for display.
#   - If we need nice formatting but want to avoid PyArrow issues, create
#     separate *_fmt columns for display:
#
#         df["value_fmt"] = df["value"].apply(lambda x: safe_format(x))
#         st.dataframe(df[["metric", "value_fmt"]], use_container_width=True)
#
#   - Under the hood, keep df["value"] as float for any computations, but
#     do not feed mixed type into the DataFrame used for st.dataframe.
#
# 5) Make sure all uses of safe_float / safe_format are imported in
#    app.py and live_helpers.py (from dashboard.nav_helpers or a shared
#    utils module).
#
# VALIDATION:
#
#   1) python -m py_compile dashboard/app.py dashboard/live_helpers.py
#
#   2) Restart dashboard:
#        sudo supervisorctl restart hedge:hedge-dashboard
#
#   3) Watch logs:
#        tail -f /var/log/hedge-dashboard.err.log
#
#      Expect:
#        - No ArrowInvalid errors.
#
#   4) Navigate through both:
#        - Overview (v7)
#        - Advanced (v6) tabs
#
#      Confirm:
#        - All tables render without crash.
#        - Numeric values look correct (no "95.60" vs 95.6 type conflicts).
