# Veto Attribution Analysis — Phase A.3

**Generated:** 2026-02-02T15:15:11Z
**Total Vetoes:** 1,140
**Date Range:** 2026-02-01 → 2026-02-02

---

## 1. Veto Counts by Reason

This shows *which constraint dimension* is doing the most work.

| Veto Reason       | Count | %     |
|-------------------|-------|-------|
| symbol_cap        | 978   | 85.8% |
| min_notional      | 151   | 13.2% |
| leverage_cap      | 2     | 0.2%  |
| kill_switch       | 1     | 0.1%  |
| per_trade_cap     | 1     | 0.1%  |
| open_notional_cap | 1     | 0.1%  |
| nav_stale         | 1     | 0.1%  |
| max_trade_nav     | 1     | 0.1%  |
| whitelist         | 1     | 0.1%  |
| daily_loss        | 1     | 0.1%  |
| cooldown          | 1     | 0.1%  |
| circuit_breaker   | 1     | 0.1%  |


## 2. Veto Counts by Symbol

Identifies symbols that distort the feasible region.

| Symbol  | Count | %     |
|---------|-------|-------|
| SOLUSDT | 445   | 39.0% |
| BTCUSDT | 402   | 35.3% |
| ETHUSDT | 293   | 25.7% |


## 3. Veto Counts by Strategy Head

Shows which heads generate the most rejected intent.

| Strategy   | Count | %     |
|------------|-------|-------|
| unknown    | 818   | 71.8% |
| vol_target | 322   | 28.2% |


## 4. Veto Counts by Entry Regime

Shows regime-conditioned veto density.

| Regime      | Count | %     |
|-------------|-------|-------|
| unknown     | 818   | 71.8% |
| MEAN_REVERT | 322   | 28.2% |


## 5. Symbol × Reason Matrix

Cross-tabulation showing which symbols hit which constraints.

| Symbol  | Reason            | Count |
|---------|-------------------|-------|
| BTCUSDT | symbol_cap        | 390   |
| SOLUSDT | symbol_cap        | 324   |
| ETHUSDT | symbol_cap        | 264   |
| SOLUSDT | min_notional      | 121   |
| ETHUSDT | min_notional      | 28    |
| BTCUSDT | min_notional      | 2     |
| BTCUSDT | leverage_cap      | 2     |
| BTCUSDT | kill_switch       | 1     |
| BTCUSDT | per_trade_cap     | 1     |
| BTCUSDT | open_notional_cap | 1     |
| BTCUSDT | nav_stale         | 1     |
| BTCUSDT | max_trade_nav     | 1     |
| ETHUSDT | whitelist         | 1     |
| BTCUSDT | daily_loss        | 1     |
| BTCUSDT | cooldown          | 1     |
| BTCUSDT | circuit_breaker   | 1     |


## 6. Strategy × Reason Matrix

Shows which constraint dimensions each head pushes against.

| Strategy   | Reason            | Count |
|------------|-------------------|-------|
| unknown    | symbol_cap        | 656   |
| vol_target | symbol_cap        | 322   |
| unknown    | min_notional      | 151   |
| unknown    | leverage_cap      | 2     |
| unknown    | kill_switch       | 1     |
| unknown    | per_trade_cap     | 1     |
| unknown    | open_notional_cap | 1     |
| unknown    | nav_stale         | 1     |
| unknown    | max_trade_nav     | 1     |
| unknown    | whitelist         | 1     |
| unknown    | daily_loss        | 1     |
| unknown    | cooldown          | 1     |
| unknown    | circuit_breaker   | 1     |


## 7. Regime × Reason Matrix

Shows how feasible region size varies by regime.

| Regime      | Reason            | Count |
|-------------|-------------------|-------|
| unknown     | symbol_cap        | 656   |
| MEAN_REVERT | symbol_cap        | 322   |
| unknown     | min_notional      | 151   |
| unknown     | leverage_cap      | 2     |
| unknown     | kill_switch       | 1     |
| unknown     | per_trade_cap     | 1     |
| unknown     | open_notional_cap | 1     |
| unknown     | nav_stale         | 1     |
| unknown     | max_trade_nav     | 1     |
| unknown     | whitelist         | 1     |
| unknown     | daily_loss        | 1     |
| unknown     | cooldown          | 1     |
| unknown     | circuit_breaker   | 1     |


## 8. Daily Veto Trend

| Date       | Count |
|------------|-------|
| 2026-02-01 | 660   |
| 2026-02-02 | 480   |


## 9. Tier Distribution

| Tier      | Count | %     |
|-----------|-------|-------|
| CORE      | 465   | 40.8% |
| SATELLITE | 338   | 29.6% |
| unknown   | 337   | 29.6% |


---

## Key Observations

### Constraint Pressure Analysis

1. **Dominant Constraint:** `symbol_cap` accounts for 85.8% of all vetoes
2. **Highest Pressure Symbol:** `SOLUSDT` with 445 vetoes (39.0%)
3. **Strategy Attribution Coverage:** 28.2% of vetoes have strategy metadata

### Interpretation

- `symbol_cap` dominance indicates **position concentration pressure**
- Heads are trying to add to existing positions rather than diversify
- Potential upstream fix: incorporate open position notional into sizing before signal generation

⚠️ **Data Gap:** >50% of vetoes lack strategy attribution.
   Consider enriching veto logging to include source head.

---

## Phase A.3 Status

This analysis is **observational only**. No execution behavior changes.

Next steps (if warranted):
- [ ] Investigate whether symbol_cap can be pre-projected in sizing
- [ ] Enrich veto logging with strategy head for unattributed records
- [ ] Build regime-sliced veto rate dashboard panel
