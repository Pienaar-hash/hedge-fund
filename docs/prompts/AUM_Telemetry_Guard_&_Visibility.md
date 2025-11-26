# PATCH SCOPE: Add AUM Telemetry Guard & Visibility to write_nav_state
# FILE: execution/state_publish.py
#
# GOAL:
#   Add lightweight logging inside write_nav_state() so we can trace:
#     - when nav_snapshot arrives without AUM
#     - what the futures/offexchange/total values are when AUM exists
#
#   NO changes to NAV math.
#   NO changes to strategy logic.
#   NO behavior change except logging + keeping nav.json shape stable.

# INSTRUCTIONS TO CODEX:
#   1) Locate function: write_nav_state(nav_snapshot)
#   2) Just before writing nav.json, insert the guard + telemetry block below.
#   3) Do not modify any other logic. Do not remove existing guards.

# INSERT (before the write_json call):

    import logging
    logger = logging.getLogger(__name__)

    # --- v7 AUM Telemetry Guard ---
    if "aum" not in nav_snapshot:
        logger.warning(
            "[nav-state] nav_snapshot missing AUM; injecting futures-only block. "
            "nav_total=%s",
            nav_snapshot.get("nav_total"),
        )
        # existing guard already injects minimal AUM – do not alter it
    else:
        aum = nav_snapshot.get("aum") or {}
        offx = aum.get("offexchange") or {}
        logger.info(
            "[nav-state] AUM present: futures=%s offexchange_keys=%s total=%s",
            aum.get("futures"),
            sorted(offx.keys()),
            aum.get("total"),
        )

# VALIDATION:
#   python -m py_compile execution/state_publish.py
#
#   Supervisor restart:
#       sudo supervisorctl restart hedge:hedge-executor hedge:hedge-sync_state hedge:hedge-dashboard
#
#   Observe executor + sync_state logs for:
#       [nav-state] AUM present: ...
#       or
#       [nav-state] nav_snapshot missing AUM; ...
#
#   Then inspect:
#       cat logs/state/nav.json | jq '.aum'
#
#   Confirm:
#       - It exists each time.
#       - futures/offexchange/total keys are present.
#
# END PATCH SCOPE.
