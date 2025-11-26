# CODEX AUDIT REQUEST
# Dashboard ArrowInvalid / "value" Column Audit

# CONTEXT:
#   pyarrow raises:
#     ArrowInvalid: ("Could not convert '95.40' with type str: tried to convert to int64",
#                    'Conversion failed for column value with type object')
#   This happens when the Streamlit dashboard starts.
#
#   v7 AUM + nav.json now work, but at least one DataFrame feeding either
#   Altair or st.dataframe still has a "value" column with mixed types
#   (strings and numbers), causing Arrow to blow up.

# GOAL:
#   Perform a comprehensive READ-ONLY audit of the dashboard code to:
#
#     1) Find every place a DataFrame with a "value" column is created.
#     2) Determine the inferred dtype of that "value" column and where strings
#        like "95.40" could appear.
#     3) Identify which of these DataFrames are used with:
#          - Altair (mark_arc / Theta(field="value"))
#          - st.dataframe / st.table
#     4) For each candidate, explain how mixed types could arise
#        (e.g. some rows numeric, some formatted strings).
#     5) Pinpoint the MOST LIKELY source of the current ArrowInvalid.
#
#   No code changes. Produce an audit report only.

# SCOPE:
#   Search and inspect:
#     - dashboard/app.py
#     - dashboard/nav_helpers.py
#     - dashboard/kpi_panel.py
#     - dashboard/live_helpers.py
#
#   Look for:
#     - "value" as a dict key or DataFrame column
#     - Altair charts using field="value"
#     - st.dataframe / st.table calls and the DataFrames they receive
#
#   Specifically examine:
#     - AUM donut:
#         slices = build_aum_slices(...)
#         df = pd.DataFrame(slices)  # has "label", "value", "qty"
#         alt.Theta(field="value", type="quantitative")
#     - v7 ATR per-symbol panel:
#         atr_entries = kpis.get("atr", {}).get("symbols", [])
#         df = pd.DataFrame(atr_entries)
#     - Any v6 "Advanced" tables that include a "value" column.

# QUESTIONS TO ANSWER:

# 1. ENUMERATE ALL "value" COLUMNS
#    For each place where a DataFrame is constructed with a "value" column:
#       - Show the code snippet (function and lines).
#       - Explain what the entries look like (dict keys, sample fields).

# 2. ALTair / PYARROW PATHS
#    Identify all Altair charts that use "value" as a quantitative field:
#       - For each, show how df["value"] is constructed.
#       - Note whether any upstream code ever wraps numbers with formatting
#         (e.g. f"{x:.2f}") or uses safe_format() into "value".

# 3. ST.DATAFRAME / VALUE
#    Identify any DataFrames passed to st.dataframe or st.table that contain
#    a "value" column:
#       - Show how those frames are built.
#       - Note any post-processing (fillna, astype, formatting).

# 4. MIXED-TYPE RISK ANALYSIS
#    For each candidate DataFrame with "value":
#       - Explain how we could end up with "95.40" as a str in some rows
#         and integers or floats in others.
#       - Indicate whether pandas will infer dtype=object or int64/float64
#         from the construction.

# 5. LIKELY ROOT CAUSE
#    Based on the above, identify the SINGLE MOST LIKELY DataFrame causing
#    the ArrowInvalid error (with column "value"), and explain why.

# 6. FIX DESIGN (NO CODE YET)
#    For that root-cause DataFrame, propose a minimal, safe fix such as:
#       - df["value"] = pd.to_numeric(df["value"], errors="coerce")
#         (then fillna(0.0))
#       - OR casting "value" to string if it's meant to be textual.
#
#    Do NOT modify any files yet — just describe the fix location and logic.

# FORMAT:
#   Please structure the report as:

#     1. All "value" Column DataFrames
#     2. Altair Uses of "value"
#     3. st.dataframe/st.table Uses of "value"
#     4. Mixed-Type Risk Assessment
#     5. Most Probable ArrowInvalid Source
#     6. Recommended Fix Location + Logic (no code)

# IMPORTANT:
#   - This is an AUDIT ONLY. No code edits or new files.
#   - We want a precise diagnosis before patching.

# END AUDIT REQUEST.
