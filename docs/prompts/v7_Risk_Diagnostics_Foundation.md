# PATCH SCOPE: v7 Risk Diagnostics Foundation
# Files:
# - execution/risk_limits.py
# - execution/risk_engine_v6.py
# - execution/utils/execution_health.py (small helper)

# Objective:
# Add v7-standard diagnostics: gate, thresholds, observations to each risk veto.
# Attach DD state, ATR regime, and fee/PnL ratio to risk snapshots and veto logs.

# Requirements:
# 1. In risk_limits.check_order():
#    - For every veto path, build a diagnostics dict:
#         {
#             "gate": <string name of guard>,
#             "thresholds": {...},
#             "observations": {...},
#             "nav_total": <number>,
#         }
#    - Pass diagnostics to _emit_veto().

# 2. Enrich veto diagnostics with:
#    - dd_state (from drawdown_tracker.current_state())
#    - atr_regime (from utils/vol.get_atr_regime or equivalent helper)
#    - fee_pnl_ratio (import from execution/utils/metrics.py)

# 3. In execution/risk_engine_v6.py:
#    - Extend the risk snapshot object returned to executor:
#         snapshot["dd_state"] = drawdown_tracker.current_state()
#         snapshot["atr_regime"] = vol_utils.get_atr_regime(nav_health_snapshot, positions)
#         snapshot["fee_pnl_ratio"] = metrics.fee_pnl_ratio(...)
#    - Ensure snapshot fields appear in risk.json via state_publish.

# 4. In state_publish:
#    - No changes here in this patch, just ensure new fields appear automatically
#      in the risk state payload passed in.

# 5. execution_health:
#    - Add optional recorder for "risk_gate_triggered" events
#      to support router/execution health displays.

# Notes:
# - Do not change risk math, only augment diagnostics.
# - Do not add new veto reasons or alter ordering.
# - Keep fractions (0.0–1.0) for thresholds.
# - Preserve existing veto schema fields and simply extend with v7 fields.

# == Patch Notes == #
• - Expanded execution/risk_limits.py veto diagnostics with nav_total/gate/thresholds/observations
    payloads, attached dd_state snapshot, ATR regime, and fee/PnL ratio data to both pass-through
    and veto detail, and enriched nav guard diagnostics with nav_total.
  - Enhanced execution/risk_engine_v6.py risk snapshots to carry portfolio dd state, ATR regime
    summary, fee/PnL ratio, and nav health context, using portfolio/position-aware helpers.
  - Added optional risk-gate event recorder and surfacing of recent risk gate events in execution
    health outputs via execution/utils/execution_health.py.

  Tests: python -m py_compile execution/risk_limits.py execution/risk_engine_v6.py execution/utils/
  execution_health.py.

  Next steps: *1) Run targeted pytest around risk and health helpers to confirm behavior (pytest
  tests/test_risk_engine_v6.py tests/test_execution_health.py). 2) Verify risk_snapshot.json
  contains new dd/atr/fee fields in a dry-run loop.*

  • - Added v7 diagnostics to risk veto paths and snapshots: nav_total, dd_state snapshot, ATR regime,
    and fee/PnL ratios now flow through risk_limits.check_order and veto logs.
  - RiskEngineV6 snapshots now include dd_state, ATR regime summary, fee/PnL ratio, and nav health
    so they surface in risk_snapshot.json.
  - Execution health gained an optional record_risk_gate_triggered recorder; risk gate events are
    exposed under events.risk_gates.
  - Restored screener/executor test shims for RiskEngineV6 (constants, placeholders, compatibility
    defaults).

  Tests: pytest tests/test_risk_engine_v6.py tests/test_execution_health.py (pass).

  Verification: Ran a dry snapshot build via RiskEngineV6.build_risk_snapshot(); output contained
  dd_state, atr_regime, fee_pnl_ratio, and fee_pnl fields with sample data.

  Next steps: 1) Run the executor in dry-run and inspect logs/state/risk_snapshot.json to confirm
  live payload includes the new fields. 2) If needed, wire record_risk_gate_triggered from veto
  emission points for health views.