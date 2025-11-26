# PATCH SCOPE: Finalize Workstream C – AUM Cleanup + Config Hygiene
#
# Files:
#   - execution/nav.py        (verify AUM path only)
#   - execution/state_publish.py  (verify AUM injection)
#   - config/assets.json      (delete or move to _deprecated)
#   - config/reserves.json    (delete or move to _deprecated)
#
# GOAL:
#   - Fully lock in WS-C by removing unused legacy config files,
#     ensuring no module imports or references them.
#
# ACTIONS:
#
# 1. Delete or rename:
#       config/assets.json       → config/_deprecated_assets.json
#       config/reserves.json     → config/_deprecated_reserves.json
#
# 2. Search entire repo for references to these files and remove/replace them.
#      Particularly check:
#        - execution/nav.py
#        - execution/sync_state.py
#        - execution/state_publish.py
#        - execution/intel/*
#        - legacy utils
#
# 3. Confirm that NAV and AUM logic does NOT depend on these files.
#
# 4. Add a module-level comment in execution/nav.py:
#       """
#       v7 AUM source: AUM = futures NAV (pure) + off-exchange holdings.
#       Legacy reserves/treasury handling removed.
#       """
#
# 5. Preserve offexchange_holdings.json intact.
#
# SAFETY:
#   - Do not alter any risk logic.
#   - Do not alter futures NAV computation.
#   - Keep all AUM additions strictly additive and non-interfering.
#
# VALIDATION AFTER PATCH:
#   - python -m py_compile execution/nav.py execution/state_publish.py
#   - Dry-run executor:
#         PYTHONPATH=. BINANCE_TESTNET=1 DRY_RUN=1 python execution/executor_live.py
#   - Confirm:
#         cat logs/state/nav.json | jq .   # AUM block present
#         cat logs/state/kpis_v7.json | jq .  # aum_total present
#   - Ensure no module tries to import / read assets.json or reserves.json.

# == Patch Notes == #
• - Renamed legacy configs to config/_deprecated_assets.json and config/_deprecated_reserves.json,
    removing code paths that read the old files across exchange utils, treasury helpers, and
    dashboard treasury sourcing so nothing imports the deprecated names.
  - Added a brief module note in execution/nav.py clarifying the v7 AUM source and removed reserves
    references from NAV logic; off-exchange holdings remain the additive AUM component.
  - Synced docs with the new AUM config by updating docs/v7_manifest.json and docs/
    v7_Developer_Guide.md to reference config/offexchange_holdings.json.

  Validation: python -m py_compile execution/nav.py execution/state_publish.py execution/
  sync_state.py execution/exchange_utils.py dashboard/live_helpers.py.

  Next steps: run a dry-run executor (BINANCE_TESTNET=1 DRY_RUN=1 python execution/
  executor_live.py), then inspect logs/state/nav.json and logs/state/kpis_v7.json for the AUM block/
  total.