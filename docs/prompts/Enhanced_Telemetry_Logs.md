# PATCH SCOPE 1
# Enhanced AUM + NAV Telemetry Logging
#
# FILES:
#   - execution/nav.py
#   - execution/state_publish.py
#   - execution/executor_live.py (light touch, optional)
#
# GOAL:
#   Strengthen observability around AUM and NAV snapshot generation so we can
#   trace:
#       - raw off-exchange holdings
#       - fallback pricing (mark vs coingecko vs avg_cost)
#       - usd_value calculation
#       - final totals
#       - any suppression, None, or skipped keys
#
#   No behavior change — logs only.

# ---------------------------------------------------------------------------
# 1) execution/nav.py — Add detailed AUM logs inside _attach_aum

# After computing mark, fallback price (spot_usd), and usd_value:

    import logging
    logger = logging.getLogger(__name__)

    logger.info(
        "[aum-v7-detail] symbol=%s qty=%s avg_cost=%s mark=%s fallback=%s usd_value=%s",
        sym,
        qty,
        avg_cost,
        mark if (mark and mark > 0) else None,
        spot_usd,
        usd_value,
    )

# After offexchange_usd is constructed but before building snapshot["aum"]:

    logger.info(
        "[aum-v7-summary] offexchange_keys=%s offexchange_usd_total=%s",
        sorted(offexchange_usd.keys()),
        sum(v.get("usd_value") or 0.0 for v in offexchange_usd.values()),
    )

# ---------------------------------------------------------------------------
# 2) execution/state_publish.py — Strengthen AUM telemetry before writing nav.json

# Inside write_nav_state(nav_snapshot), before writing the file:

    aum = nav_snapshot.get("aum")
    if not aum:
        logger.warning("[nav-state] AUM missing from nav_snapshot (will inject futures-only).")
    else:
        offx = aum.get("offexchange") or {}
        logger.info(
            "[nav-state] AUM present: futures=%s total=%s offexchange_keys=%s",
            aum.get("futures"),
            aum.get("total"),
            sorted(offx.keys()),
        )

# Also add fallback logging for missing fields:
    if aum and any(v.get("usd_value") is None for v in offx.values()):
        missing = [k for k,v in offx.items() if v.get("usd_value") is None]
        logger.warning(
            "[nav-state] AUM warning: entries missing usd_value=%s", missing
        )

# ---------------------------------------------------------------------------
# 3) execution/executor_live.py — optional snapshot logging

# In _compute_nav_with_detail, after computing nav_detail:

    logger.info(
        "[nav-detail] nav_total=%s aum_total=%s future_only_nav=%s",
        nav_detail.get("nav_total"),
        (nav_detail.get("aum") or {}).get("total"),
        nav_detail.get("nav_total"),   # redundancy explicit for clarity
    )

# END SCOPE 1
