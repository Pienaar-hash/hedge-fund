# prediction/ — DLE-native prediction layer (belief ledger)
#
# Produces Decision+Permit objects for belief updates the same way
# execution/ does for trades.  No belief affects anything unless it
# is admitted (Dataset Admission Gate) and permitted (Decision Permit).
#
# Phase P0: observe-only (ingest + log, no downstream influence)
# Phase P1: advisory-only (alert ranking, dashboard overlay, no execution impact)
# Phase P2: production-eligible (advisory hints only, never overrides Sentinel-X)
#
# IMPORT BOUNDARY: execution/ may import from prediction/ ONLY via
# execution/telegram_utils._maybe_rank_alerts().  No other execution
# module may import from this package.  That function is try/except
# wrapped (fail-open) so prediction can never break executor liveness.
#
# See docs/PHASE_P1_PREDICTION_ADVISORY_DOCTRINE.md for P1 rules.
