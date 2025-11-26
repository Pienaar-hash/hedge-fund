# PATCH SCOPE: v7 AUM Cleanup & Off-Exchange Holdings Integration
#
# Files:
#   - execution/nav.py
#   - execution/state_publish.py
#   - execution/sync_state.py (light update if needed)
#   - config/offexchange_holdings.json (new)
#
# GOAL:
#   1) Keep nav_total as pure Binance futures NAV.
#   2) Introduce AUM breakdown by adding off-exchange holdings
#      sourced from config/offexchange_holdings.json.
#   3) Add AUM block to logs/state/nav.json.
#   4) (Optional) Surface AUM summary in kpis_v7.json.
#   5) Do NOT reintroduce reserves/treasury blending.
#
# REQUIREMENTS:
#
# 1) Create a new config file:
#      config/offexchange_holdings.json
#
#    Expected schema:
#    {
#      "BTC": { "qty": <float>, "avg_cost": <float> },
#      "XAUT": { "qty": <float>, "avg_cost": <float> },
#      "USDC": { "qty": <float>, "avg_cost": <float> }
#    }
#
#    Do not load reserves.json or treasury fields.
#
# 2) In execution/nav.py:
#    - Add helper: load_offexchange_holdings().
#    - After computing futures nav_total, compute AUM:
#         offexchange_usd = {}
#         for symbol, data in holdings.items():
#             qty = data["qty"]
#             mark = get_mark_price_for_symbol(symbol)  # reuse existing futures price feeds
#             offexchange_usd[symbol] = {
#                 "qty": qty,
#                 "usd_value": qty * mark,
#                 "avg_cost": data.get("avg_cost")
#             }
#         aum_total = nav_total + sum(v["usd_value"] for v in offexchange_usd.values())
#
#    - Add to nav snapshot:
#         snapshot["aum"] = {
#            "futures": nav_total,
#            "offexchange": offexchange_usd,
#            "total": aum_total
#         }
#
# 3) In execution/state_publish.py:
#    - Ensure nav_snapshot is passed through to the state file with the new AUM block included.
#    - (Optional) Add a small AUM summary entry in kpis_v7.json under:
#         kpis["aum_total"] = snapshot["aum"]["total"]
#
# 4) In execution/sync_state.py:
#    - Mirror kpis_v7.json and nav.json as usual. No special handling.
#
# 5) Safety:
#    - No additional API calls.
#    - If a mark price is missing, use "usd_value": null and still publish qty + avg_cost.
#    - Do not modify risk or NAV mathematics.
#
# TESTING PLAN:
#   1) python -m py_compile execution/nav.py execution/state_publish.py
#   2) Dry-run executor:
#        PYTHONPATH=. BINANCE_TESTNET=1 DRY_RUN=1 python execution/executor_live.py
#   3) Inspect:
#        cat logs/state/nav.json | jq .
#        cat logs/state/kpis_v7.json | jq .
#
# EXPECTED:
#   - nav_total unchanged.
#   - nav.json contains "aum" block as specified.
#   - No reserves/treasury fields present.
#   - Delete assets.json and reserves.json if no longer needed.

# == Patch Notes == #
• - Added off-exchange holdings support in execution/nav.py: loads config/offexchange_holdings.json,
    reuses mark prices via get_mark_price_for_symbol, and appends an AUM block (futures,
    offexchange, total) to NAV snapshots without altering futures nav_total. Manual and cache
    fallback paths also carry AUM.
  - Extended KPI builder in execution/state_publish.py to surface AUM totals (nav.aum_total and top-
    level aum_total) for dashboard/investor consumption.
  - Wired execution/sync_state.py to mirror the updated KPIs by importing build_kpis_v7 and writing
    logs/state/kpis_v7.json during sync.
  - Added config/offexchange_holdings.json template for off-exchange positions.

  Tests: python -m py_compile execution/nav.py execution/state_publish.py execution/sync_state.py.

  Next steps: 1) Populate config/offexchange_holdings.json with real quantities/costs. 2) Run
  executor in dry-run and inspect logs/state/nav.json to confirm the new aum block, and logs/state/
  kpis_v7.json for aum_total. 3) Ensure dashboard/sync picks up the mirrored KPI file.

  - Problems: 
    - the AUM donut for the dashboard needs to have asset slices e.g. BTC, XAUT, USDC, USDT
    - the slices should be dynamic e.g. as NAV grows the USDT slice should grow
    - assets.json and reserves.json from older versions still live in the config folder and should be deleted if not used
