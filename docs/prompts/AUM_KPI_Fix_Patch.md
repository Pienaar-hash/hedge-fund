# PATCH SCOPE: FIX AUM/KPI Data Types, Safe Casting, Formatting, & PyArrow Errors

# Files to modify:
#   dashboard/nav_helpers.py
#   dashboard/kpi_panel.py
#   dashboard/app.py

# GOALS:
#   1) Prevent PyArrow ArrowInvalid errors by enforcing strict float type casting.
#   2) Ensure AUM slices always contain numeric values (floats or None).
#   3) Ensure KPIs NEVER contain string placeholders like “1…”, “–”, “n/a” when
#      they are passed into numeric fields.
#   4) Add safe formatting utilities for metric output.
#   5) Harden positions table using safe casting.
#   6) Ensure dashboard gracefully handles missing or stale data.

# ---------------------------------------------------------------------------
# 1) ADD SAFE CAST HELPERS IN nav_helpers.py
#
# Add at top of file:

def safe_float(x):
    """
    Convert x to float, or return None if conversion impossible.
    Accepts int, float, str, None.
    """
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            x2 = x.replace(",", "").strip()
            return float(x2)
        return None
    except:
        return None

def safe_round(x, nd=2):
    xf = safe_float(x)
    if xf is None:
        return None
    return round(xf, nd)

def safe_format(x, nd=2):
    xf = safe_float(x)
    if xf is None:
        return "–"
    return f"{xf:,.{nd}f}"

# ---------------------------------------------------------------------------
# 2) FIX AUM SLICE BUILDER IN nav_helpers.py
#
# Inside build_aum_slices():

Replace value extraction with:

    raw_usd = off.get("usd_value")
    raw_qty = off.get("qty")

    usd_val = safe_float(raw_usd)
    qty_val = safe_float(raw_qty)

    if usd_zar is not None and usd_val is not None:
        zar_val = usd_val * safe_float(usd_zar)
    else:
        zar_val = None

Return slice dict as:

    slices.append({
        "label": symbol_label,
        "value": usd_val if usd_val is not None else 0.0,
        "qty": qty_val,
        "zar": zar_val,
    })

ALSO ensure futures slice is cast safely:

    fut_val = safe_float(aum.get("futures"))
    slices.append({
        "label": "Futures",
        "value": fut_val if fut_val is not None else 0.0,
        "qty": None,
        "zar": fut_val * usd_zar if (usd_zar and fut_val is not None) else None,
    })

# ---------------------------------------------------------------------------
# 3) FIX KPI SAFE FORMATTING (dashboard/kpi_panel.py)
#
# Wherever st.metric() is called with numeric formatting:

Replace:

    st.metric("NAV", f"{nav_total:.2f}")

With:

    st.metric("NAV (USD)", safe_format(nav_total))

Replace:

    st.metric("AUM", f"{aum_total:.2f}")

With:

    st.metric("AUM (USD)", safe_format(aum_total))

Replace DD/ATR:

    st.metric("DD / ATR", f"{dd_state} / {atr_regime}")

If dd_pct exists:

    dd_pct_fmt = safe_format(dd_pct, nd=3)
    st.metric("DD %", dd_pct_fmt)

Router stats:

    mfr = safe_format(kpis.get("router_stats", {}).get("maker_fill_ratio"))
    fbr = safe_format(kpis.get("router_stats", {}).get("fallback_ratio"))

# ---------------------------------------------------------------------------
# 4) FIX KPI TRUNCATION IN app.py
#
# Replace any direct f"{val:...}" with safe_format(val).
#
# Example:

nav_fmt = safe_format(nav_total)
aum_fmt = safe_format(aum_total)
nav_zar_fmt = safe_format(nav_total * usd_zar) if usd_zar else "–"
aum_zar_fmt = safe_format(aum_total * usd_zar) if usd_zar else "–"

col_nav.metric("NAV (USD)", nav_fmt)
col_aum.metric("AUM (USD)", aum_fmt)

# ---------------------------------------------------------------------------
# 5) FIX POSITIONS TABLE TYPES IN app.py
#
# Before calling st.dataframe(df):

Cast numeric columns safely:

    for col in ["qty", "notional", "pnl"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

# Optional: Format notional & pnl as strings for display:

    df["notional_fmt"] = df["notional"].apply(lambda x: safe_format(x))
    df["pnl_fmt"] = df["pnl"].apply(lambda x: safe_format(x))

Render:

    st.dataframe(df[["symbol", "side", "qty", "notional_fmt", "pnl_fmt"]],
                 use_container_width=True)

# ---------------------------------------------------------------------------
# 6) HARDEN AUM DONUT TOOLTIP
#
# In altair chart:

.tooltip(["label", "value", "qty", "zar"])

Ensure values come from floats only.

# ---------------------------------------------------------------------------
# 7) ENSURE MISSING DATA NEVER PROPAGATES AS STRING
#
# For all missing numeric values, always set None → display "–".

Example:

    dd_pct = safe_float((kpis.get("drawdown") or {}).get("dd_pct"))

    st.metric("Drawdown %", safe_format(dd_pct, nd=3))

# ---------------------------------------------------------------------------
# 8) VERIFY NAV/AUM DATA ALWAYS EXISTS
#
# In app.py:

Before metrics:

    if not isinstance(nav_v7, dict):
        st.error("nav.json missing or malformed")
        return

    if "aum" not in nav_v7:
        st.warning("AUM block missing from nav.json")
        return

# ---------------------------------------------------------------------------
# VALIDATION STEPS:
#
# 1. python -m py_compile dashboard/*.py
#
# 2. sudo supervisorctl restart hedge:hedge-executor hedge:hedge-sync_state hedge:hedge-dashboard
#
# 3. tail -f /var/log/hedge-dashboard.err.log → confirm no ArrowInvalid errors.
#
# 4. Visit dashboard:
#       - NAV/AUM show numeric values
#       - Donut shows slices
#       - KPIs fully visible, no truncation
#       - ZAR tooltips appear
#       - Positions table readable and numeric
#
# 5. Inspect nav.json manually:
#       cat logs/state/nav.json | jq .
#    Confirm:
#       .aum.futures
#       .aum.offexchange
#       .aum.total
#
# END PATCH SCOPE
