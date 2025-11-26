# PATCH SCOPE: Off-Exchange AUM Fallback Pricing (Telemetry-Only)
#
# Files to modify:
#   - execution/nav.py
#   - (optional) execution/utils/price_utils.py  (only if a helper is desired)
#
# GOAL:
#   Off-exchange holdings (BTC, XAUT, USDC, etc.) must obtain a usable USD price
#   even when Binance futures UM mark prices are unavailable or zero.
#
#   This patch adds a safe fallback path:
#
#     1) Try get_mark_price_for_symbol()    # futures market (current behavior)
#     2) If mark <= 0 or None:
#            * Try cached spot price from our own price_utils (if available)
#            * Else try simple coingecko fetch (already in codebase)
#            * Else fall back to avg_cost (from config/offexchange_holdings.json)
#
#   IMPORTANT:
#     - NAV math stays unchanged (nav_total = futures NAV ONLY).
#     - AUM is telemetry-only.
#     - Risk engine, router, sizer, executor logic must NOT read AUM.
#     - This patch only affects _attach_aum() in execution/nav.py.
#
# ---------------------------------------------------------------------------
# 1) Add a small helper: get_usd_fallback_price(symbol)
#
# Insert near top of execution/nav.py (or inside the module namespace):

def _get_usd_fallback_price(symbol: str, default: float | None = None):
    """
    Fallback USD spot price for off-exchange holdings when UM mark price is unavailable.
    Order:
        1) coingecko price (if available)
        2) default (avg_cost from config)
    Returns None if nothing available.
    """
    try:
        from execution.exchange_utils import coingecko_price_usd
    except Exception:
        coingecko_price_usd = None

    # Try coingecko spot
    if coingecko_price_usd:
        try:
            px = coingecko_price_usd(symbol)
            if px and px > 0:
                return float(px)
        except Exception:
            pass

    # Fallback to supplied default
    return default


# ---------------------------------------------------------------------------
# 2) Patch _attach_aum() to use fallback pricing

# Locate _attach_aum(nav_snapshot, offexchange_cfg) in execution/nav.py.
# Replace the price logic inside the loop:

# OLD:
#     mark = get_mark_price_for_symbol(sym)
#     if not mark or mark <= 0:
#         usd_value = None
#     else:
#         usd_value = qty * mark

# NEW LOGIC:

    mark = get_mark_price_for_symbol(sym)

    # Try futures mark price first
    spot_usd = None
    if not mark or mark <= 0:
        # fallback: use coingecko or avg_cost
        spot_usd = _get_usd_fallback_price(sym, default=avg_cost)
        if spot_usd and spot_usd > 0:
            usd_value = qty * spot_usd
        else:
            usd_value = None
    else:
        # futures mark price OK
        usd_value = qty * mark

    # Store: use either mark or fallback price for reporting
    entry = {
        "qty": qty,
        "avg_cost": avg_cost,
        "mark": mark if mark and mark > 0 else spot_usd,
        "usd_value": usd_value,
    }

    offexchange_usd[sym] = entry


# ---------------------------------------------------------------------------
# 3) No changes needed for futures NAV (nav_total).
#    AUM block is purely additive telemetry:
#
#     snapshot["aum"] = {
#         "futures": nav_total,
#         "offexchange": offexchange_usd,
#         "total": nav_total + sum(v["usd_value"] or 0 for v in offexchange_usd.values())
#     }
#
#   Ensure this behavior is preserved.

# ---------------------------------------------------------------------------
# 4) Logging enhancement (optional but recommended)

# Inside _attach_aum(), after computing offexchange_usd:

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "[aum-v7] computed offexchange AUM: keys=%s totals_usd=%s",
        sorted(offexchange_usd.keys()),
        sum(v.get("usd_value") or 0 for v in offexchange_usd.values()),
    )

# ---------------------------------------------------------------------------
# VALIDATION STEPS:
#
#   1. python -m py_compile execution/nav.py
#
#   2. Restart executor + sync_state:
#        sudo supervisorctl restart hedge:hedge-executor hedge:hedge-sync_state
#
#   3. After ~10 seconds:
#        cat logs/state/nav.json | jq '.aum.offexchange'
#
#      Expect:
#        {
#          "BTC":  {"qty":0.035, "mark":<spot>, "usd_value":<computed>},
#          "XAUT": {"qty":0.59,  "mark":<spot>, "usd_value":<computed>},
#          "USDC": {"qty":2098.96, "mark":1, "usd_value":2098.96}
#        }
#
#   4. Dashboard:
#        - AUM donut shows BTC/XAUT/USDC slices
#        - ZAR tooltips available
#        - AUM (USD) ≠ NAV (USD)
#
#   5. Risk/sizer/router unaffected (safe).
#
# END PATCH SCOPE.
