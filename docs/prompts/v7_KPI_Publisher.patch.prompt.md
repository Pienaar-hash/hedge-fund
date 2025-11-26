# PATCH SCOPE: v7 KPI Publisher
#
# Files:
#   - execution/state_publish.py
#   - execution/sync_state.py
#
# GOAL:
#   Add a v7 KPI publisher that writes a consolidated KPI state file:
#     logs/state/kpis_v7.json
#   and ensure it is mirrored by sync_state.py for dashboard consumption.
#
# CONTEXT:
#   - RiskEngineV6 snapshots now include dd_state, atr_regime, fee_pnl_ratio, and nav health
#     (see execution/risk_engine_v6.py).
#   - state_publish.py already writes nav.json, positions.json, router.json, risk.json, etc.
#   - sync_state.py already mirrors existing state files under logs/state/ for the dashboard.
#
# REQUIREMENTS:
#
# 1) Implement a KPI builder in execution/state_publish.py
#
#   - Add a helper function, e.g.:
#
#       def build_kpis_v7(now_ts, nav_snapshot, risk_snapshot, router_state, expectancy_state=None):
#           """
#           Build a v7 KPI payload for logs/state/kpis_v7.json.
#
#           Expected schema (documented, not strictly enforced):
#           {
#             "ts": <ISO8601>,
#             "nav": {
#               "nav_total": <float>,
#               "nav_age_s": <float>,
#               "sources_ok": <bool>
#             },
#             "risk": {
#               "dd_state": <dict or null>,
#               "atr_regime": <str or null>,
#               "fee_pnl_ratio": <float or null>
#             },
#             "router": {
#               "policy_quality": <str or null>,
#               "maker_first": <bool or null>,
#               "maker_fill_share": <float or null>,  # fraction 0–1
#               "avg_slippage_bps": <float or null>,
#               "reject_rate": <float or null>        # fraction 0–1
#             },
#             "performance": {
#               "expectancy": <float or null>,
#               "sharpe_state": <str or null>
#             }
#           }
#           """
#
#         - Use whatever fields are already present in nav_snapshot, risk_snapshot,
#           and router_state. Do NOT change their existing schemas.
#         - It is OK for some KPI fields to be null if data is not available.
#
#   - Inject the KPI builder into the main state publishing path where nav/risk/router
#     state objects are already available.
#
#   - Write the result to:
#
#       logs/state/kpis_v7.json
#
#     using the same JSON dump helper / safe writer pattern already used for other
#     state files in state_publish.py.
#
# 2) Keep math + risk & router behaviour unchanged
#
#   - Do not alter risk, nav, or router calculations.
#   - Only read from existing snapshot/state structures and assemble a new KPI payload.
#   - If you need derived ratios (e.g. maker_fill_share, reject_rate, fee_pnl_ratio),
#     prefer existing metrics in router_state or risk_snapshot. If no clean source exists,
#     set the KPI field to None and leave a TODO comment.
#
# 3) Mirror KPIs via execution/sync_state.py
#
#   - Update sync_state.py so that logs/state/kpis_v7.json is mirrored to the dashboard
#     state directory just like nav.json, positions.json, router.json, etc.
#   - Follow the existing patterns in sync_state for which files get mirrored and how.
#   - Do not introduce new config knobs; treat kpis_v7.json as a first-class mirrored file.
#
# 4) Documentation & safety
#
#   - Add a short module-level comment or docstring in state_publish.py describing
#     the kpis_v7.json contract and intended consumers (dashboard panels, investor view).
#   - Ensure all new code is compatible with BINANCE_TESTNET=1 and DRY_RUN=1.
#   - Avoid importing heavy modules that are not already used in state_publish or sync_state.
#
# TESTING / VALIDATION (what we will run after this patch):
#
#   1. Import sanity:
#        python -m py_compile execution/state_publish.py execution/sync_state.py
#
#   2. Dry-run executor:
#        PYTHONPATH=. BINANCE_TESTNET=1 DRY_RUN=1 python execution/executor_live.py
#      Let it run through at least one full loop after startup-sync.
#
#   3. KPI file existence + basic shape:
#        cat logs/state/kpis_v7.json | jq .
#
#      We expect:
#        - top-level keys: ts, nav, risk, router, performance
#        - nav/nav_total and nav_age_s populated
#        - risk.dd_state and risk.atr_regime present if risk snapshot has them
#
#   4. sync_state mirror (assuming run-sync is active or triggered):
#        # depending on how sync_state mirrors, we will also check the mirrored path
#        # but you don't need to change any path conventions here; just ensure
#        # kpis_v7.json participates in the same mirroring pipeline.

# == Patch Notes == #
• - Added a KPI builder in execution/state_publish.py that assembles nav/risk/router/performance
    KPIs (with legacy convenience fields) and updated write_kpis_v7_state to build from snapshots
    when provided, plus module note on the kpis_v7 contract.
  - Wired sync_state to mirror KPIs: new state paths, KPI mirroring helper using build_kpis_v7, and
    automatic write to logs/state/kpis_v7.json each sync loop.
  - The KPI builder gracefully handles missing data and uses existing snapshots without altering
    risk/nav/router math.

  Tests: python -m py_compile execution/state_publish.py execution/sync_state.py.

  Next steps: run the executor (dry-run ok) to see logs/state/kpis_v7.json populate, and check the
  mirrored file via cat logs/state/kpis_v7.json | jq.