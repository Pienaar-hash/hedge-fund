# P1 Prediction Layer — Post-Mortem

**Date:** 2026-02-08
**Author:** Engineering
**Status:** SEALED
**Commit:** `7be98d41` (tag: `prediction-p1-advisory`)
**Branch:** `v7.6-dev`

---

## 1. Objective

Build a DLE-native prediction layer that surfaces forward-looking
conviction signals **without influencing execution**. The layer
observes the existing Doctrine → Hydra → Cerberus pipeline and
produces advisory alerts ranked by confidence.

**Constraint:** P1 is advisory-only. No signal gating, no position
sizing influence, no doctrine override. Read-only observability.

---

## 2. What Was Delivered

### 2.1 Core Components (12 deliverables)

| # | Component | File | Purpose |
|---|-----------|------|---------|
| 1 | Belief Ingest | `execution/prediction/belief_ingest.py` | Normalizes DLE signals into belief objects |
| 2 | Prediction Firewall | `execution/prediction/prediction_firewall.py` | Hard barrier — predictions cannot reach executor |
| 3 | Alert Ranker | `execution/prediction/alert_ranker.py` | Ranks predictions by composite confidence score |
| 4 | Prediction State | `execution/prediction/prediction_state.py` | State surface writer (`logs/state/prediction_state.json`) |
| 5 | Prediction Episodes | `execution/prediction/prediction_episodes.py` | Tracks prediction accuracy over time |
| 6 | DLE Gate | `execution/prediction/dle_gate.py` | Controls feature flag gating for prediction layer |
| 7 | Rollback Triggers | `execution/prediction/rollback_triggers.py` | Auto-disables layer if error thresholds breached |
| 8 | Constraints | `execution/prediction/constraints.py` | Validates prediction inputs against schema |
| 9 | Alert Wiring | `execution/prediction/alert_wiring.py` | Connects alert ranker to notification pipeline |
| 10 | Dashboard Tile | `dashboard/components/prediction_tile.py` | Read-only prediction visibility in dashboard |
| 11 | Audit Bundle | `execution/prediction/audit_bundle.py` | Generates audit snapshots for post-mortem analysis |
| 12 | P2 Gate Checklist | `docs/PREDICTION_P2_GATE.md` | Promotion criteria for P1 → P2 |

### 2.2 Test Coverage

**88 unit tests** across 10 test files:

- `test_prediction_alert_ranker.py`
- `test_prediction_alert_wiring.py`
- `test_prediction_belief_ingest.py`
- `test_prediction_constraints.py`
- `test_prediction_dle_gate.py`
- `test_prediction_episodes.py`
- `test_prediction_firewall.py`
- `test_prediction_rollback_triggers.py`
- `test_prediction_state_surface.py`
- `test_prediction_tile.py`

### 2.3 Commit Stats

- **31 files changed**, 6,312 insertions, 21 deletions
- Full suite at commit: **2632 passed**, 0 failed, 47 skipped

---

## 3. Production Trial

### 3.1 Protocol

- **Duration:** 29 hours continuous
- **Mode:** P1_ADVISORY (firewall enforced, no execution influence)
- **Environment:** Production executor, live Binance futures
- **Monitoring:** Supervisor uptime, NAV telemetry, state file freshness

### 3.2 Results

| Metric | Result |
|--------|--------|
| Executor uptime | >43h continuous (no restart required) |
| Firewall breaches | 0 |
| Prediction errors | 0 |
| Rollback triggers fired | 0 |
| Execution interference | None detected |
| State file writes | Normal cadence |
| NAV impact | None (advisory-only confirmed) |

### 3.3 Verdict

**PASS.** P1 layer ran for 29+ hours in production without
incident. Firewall held. No execution influence detected.
Layer promoted to P1_ADVISORY status and tagged.

---

## 4. MHD Audit

Formal Messy/Heavy/Dependent audit conducted post-P1.

**Score: STRONG (2 yellow flags, 0 red)**

### Actions Taken

1. **Score Decomposition Ledger** (`daf92074`)
   - Added `weights_used` and `weighted_contributions` to hybrid
     score result in `symbol_score_v6.py`
   - Propagated to intents in `signal_screener.py`
   - JSONL logger to `logs/execution/score_decomposition.jsonl`

2. **Exchange Unavailable Doctrine** (`daf92074`)
   - Created `docs/EXCHANGE_UNAVAILABLE_DOCTRINE.md`
   - Fail-silent rules, gap inventory, acceptable failure cascade
   - Updated `v7_manifest.json` with score_decomposition entry

---

## 5. Dashboard Truth Surface (Post-P1)

During P1 observation, dashboard anomalies were identified and
resolved across 4 commits:

| Issue | Root Cause | Fix | Commit |
|-------|-----------|-----|--------|
| Vetoes (24h) = 0 | Read from empty `risk_snapshot`, not JSONL | Scan `risk_vetoes.jsonl` with 24h window | `62e0afc6` |
| KPI risk block empty | Flat JSON, loader expected nested | Flat-to-nested projection | `62e0afc6` |
| Sharpe = 0.00 | No upstream source computed it | Compute from episode PnL distribution | `9a76719a` |
| Max DD = -0.00% | Wrong equity base in episode_ledger | NAV-based equity curve | `9a76719a` |
| Sparkline invisible | 80x24px too small | Enlarged to 280x48px | `9a76719a` |
| 24h PnL = +$13 (true: +$455) | Episode-windowed PnL, not NAV delta | New `nav_pnl.py`: NAV(now) - NAV(24h ago) | `b16126e0` |
| "Realized (Session) +$0.00" | Stale executor session counter | Replaced with 24h NAV delta | `7c7e572b` |
| "Net PnL" ambiguous | Reads as portfolio PnL, actually episode-only | "Closed PnL (Episodes)" + "Last close: Nd ago" | `44c063a4` |

### Integrity Verification

Episode ledger cross-checked against fill logs:
- 461 episodes since Dec 12 vs 434 `order_close` events
- Delta of 27 explained by position flips (implicit closes)
- `net_pnl = gross_pnl - fees` verified for all 562 episodes (0 mismatches)
- 4,005 exit fills / 461 episodes = ~8.7 fills/episode (TWAP)

---

## 6. System State at Seal

| Layer | Status |
|-------|--------|
| Version | v7.9 |
| HEAD | `44c063a4` |
| Suite | 2632 passed, 0 failed |
| Executor | RUNNING (43h+ uptime) |
| NAV | $9,873 (live) |
| Positions | SOL LONG 25.0, ETH LONG 1.216, BTC LONG 0.05 |
| Episodes | 562 closed, last close Dec 12 |
| Closed PnL | -$270.58 (correct, verified) |
| Dashboard | All tiles truthful, no silent mismatches |

---

## 7. Open Items

None. All loops closed.

### P2 Promotion Criteria (documented in `docs/PREDICTION_P2_GATE.md`)

P2 requires:
- 7-day soak at P1 with zero incidents
- Prediction accuracy tracking shows signal value
- Formal review of advisory alerts vs actual outcomes
- Explicit sign-off before any execution influence

---

## 8. Lessons

1. **NAV delta is the only honest PnL.** Episode-windowed PnL
   systematically understates performance when unrealized gains
   dominate. Fixed permanently with `nav_pnl.py`.

2. **Labels are load-bearing.** "Net PnL" vs "Closed PnL
   (Episodes)" is a semantic difference that prevents trust
   erosion. The number was always correct — the reader's model
   was wrong.

3. **Firewalls work.** The prediction layer's hard barrier
   (no execution influence) meant 29h of production trial with
   zero risk. Worth the upfront investment.

4. **Cross-layer reconciliation catches what unit tests miss.**
   The fill → episode → dashboard trace proved integrity that
   no single test file could verify alone.
