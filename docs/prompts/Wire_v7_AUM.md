# PATCH SCOPE 1: Wire v7 AUM into logs/state/nav.json

# Files:
#   - execution/nav.py
#   - execution/state_publish.py

# GOAL:
#   Ensure that the NAV snapshot written to logs/state/nav.json always
#   includes the v7 AUM block:
#     {
#       "nav_total": <float>,
#       "nav_age_s": <float>,
#       "aum": {
#         "futures": <float>,
#         "offexchange": { ... },
#         "total": <float>
#       },
#       ...
#     }

# REQUIREMENTS:

# 1) In execution/nav.py:
#
#   - Find the function(s) responsible for building the main nav snapshot
#     used by state_publish (e.g. build_nav_snapshot / get_nav_snapshot).
#
#   - Ensure that after computing nav_total and off-exchange holdings from
#     config/offexchange_holdings.json, the snapshot dict includes:
#
#       snapshot["aum"] = {
#           "futures": <nav_total>,      # pure futures nav
#           "offexchange": <dict>,       # symbol → {qty, usd_value, avg_cost}
#           "total": <nav_total + offexchange_total_usd>
#       }
#
#   - Make sure **all** nav snapshot paths (including manual / fallback /
#     cache-based paths) attach this "aum" block before returning, not just
#     one code path.

# 2) In execution/state_publish.py:
#
#   - Identify where nav_snapshot is passed into the nav.json writer
#     (logs/state/nav.json).
#
#   - Verify that the nav_snapshot already includes "aum". Do NOT recompute
#     AUM here; instead, assert that it is present and pass it through.
#
#   - If the snapshot sometimes lacks "aum" (e.g., early in startup), add a
#     defensive guard:
#
#         if "aum" not in nav_snapshot:
#             nav_snapshot["aum"] = {
#                 "futures": nav_snapshot.get("nav_total"),
#                 "offexchange": {},
#                 "total": nav_snapshot.get("nav_total"),
#             }
#
#     so dashboard always sees a stable "aum" shape.

# 3) Do NOT change nav_total semantics.
#    nav_total must remain pure futures NAV (no treasury/reserves).

# VALIDATION:
#
#   1) python -m py_compile execution/nav.py execution/state_publish.py
#
#   2) Restart executor + sync_state:
#        sudo supervisorctl restart hedge:hedge-executor hedge:hedge-sync_state
#
#   3) Inspect:
#        cat logs/state/nav.json | jq .aum
#
#      Expect:
#        - fields: futures, offexchange, total
#        - offexchange populated when config/offexchange_holdings.json is set
#
#   4) Verify dashboard Overview (v7) now shows AUM values (USD + donut).
