# CODEX AUDIT REQUEST
# v7 NAV / AUM / SYNC_STATE / DASHBOARD STATE AUDIT
#
# OBJECTIVE:
#   Before applying any cleanup patches, perform a full codebase audit to
#   identify:
#
#   1. All files that WRITE nav snapshots.
#   2. All files that READ nav snapshots.
#   3. All files that WRITE state files under logs/state/.
#   4. All files that WRITE nav_state.json, nav_confirmed.json, nav_live.json,
#      or any legacy nav files.
#   5. All code paths that bypass the v7 AUM logic.
#   6. All code paths that might OVERWRITE nav.json with a legacy snapshot.
#   7. Any risk, sizing, router, or execution logic that references `.aum`.
#   8. All dashboard modules that read nav_state.json instead of nav.json.
#   9. All references to "nav_confirmed", "nav_live", "nav_state", "nav_cache"
#      across the repo.
#
#   Produce an audit report (no code changes!) containing:
#
#   A. File-by-file list of where nav snapshots are produced.
#   B. File-by-file list of where nav snapshots are consumed.
#   C. Identify the PRIMARY nav snapshot path used by executor_live → state_publish.
#   D. Identify which nav snapshot sync_state mirrors into logs/state/.
#   E. Identify any modules that still read v6-era files (nav_state.json,
#      nav_live.json, nav_confirmed.json).
#   F. Identify if ANY risk-related module references `.aum`.
#   G. Identify dashboard helpers still referencing legacy files.
#   H. Identify ANY file that writes to /logs/cache/ and how it interacts with
#      /logs/state/ files.
#
#   The output should clearly label:
#       - "SAFE v7 path"
#       - "LEGACY v6 path (MUST be deprecated)"
#       - "DANGEROUS path (overwrites v7 AUM)"
#       - "CONFLICTING path (mixed responsibilities)"
#
#   Include direct code excerpts or line references for every match.
#
# SCOPE:
#   Search across:
#       execution/
#       dashboard/
#       state_publish/
#       sync_state/
#       utils/
#       tests/
#
#   Search terms to locate:
#       "nav.json"
#       "nav_state"
#       "nav_confirmed"
#       "nav_live"
#       "nav_cache"
#       "logs/state/"
#       "logs/cache/"
#       ".aum"
#       "offexchange"
#       "snapshot"
#       "update_nav"
#       "publish_nav"
#       "read_nav"
#
# FORMAT:
#   Deliver the audit as a structured report with these sections:
#
#       1. NAV WRITERS
#       2. NAV READERS
#       3. LEGACY NAV PATHS
#       4. DASHBOARD READ PATHS
#       5. STATE_PUBLISH BEHAVIOR
#       6. SYNC_STATE BEHAVIOR
#       7. RISK/EXECUTION REFERENCES TO AUM
#       8. PROBABLE ROOT CAUSE of AUM missing from nav.json
#       9. PROPOSED CLEANUP PLAN (NO CODE CHANGES YET)
#
# IMPORTANT:
#   - DO NOT modify any files yet.
#   - DO NOT create or delete files.
#   - This is an AUDIT ONLY.
#
# END AUDIT REQUEST.
