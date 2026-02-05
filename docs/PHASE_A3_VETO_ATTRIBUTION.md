# Veto Attribution Analysis — Phase A.3

**Generated:** 2026-02-05T07:46:29Z
**Total Vetoes:** 1,895
**Date Range:** 2026-02-01 → 2026-02-05

---

## 1. Veto Counts by Reason

This shows *which constraint dimension* is doing the most work.

| Veto Reason          | Count | %     |
|----------------------|-------|-------|
| symbol_cap           | 1323  | 69.8% |
| min_notional         | 503   | 26.5% |
| portfolio_dd_circuit | 12    | 0.6%  |
| correlation_cap      | 10    | 0.5%  |
| leverage_cap         | 8     | 0.4%  |
| nav_stale            | 6     | 0.3%  |
| max_trade_nav        | 5     | 0.3%  |
| kill_switch          | 4     | 0.2%  |
| per_trade_cap        | 4     | 0.2%  |
| open_notional_cap    | 4     | 0.2%  |
| whitelist            | 4     | 0.2%  |
| daily_loss           | 4     | 0.2%  |
| cooldown             | 4     | 0.2%  |
| circuit_breaker      | 4     | 0.2%  |


## 2. Veto Counts by Symbol

Identifies symbols that distort the feasible region.

| Symbol  | Count | %     |
|---------|-------|-------|
| BTCUSDT | 760   | 40.1% |
| SOLUSDT | 618   | 32.6% |
| ETHUSDT | 517   | 27.3% |


## 3. Veto Counts by Strategy Head

Shows which heads generate the most rejected intent.

| Strategy   | Count | %     |
|------------|-------|-------|
| unknown    | 1436  | 75.8% |
| vol_target | 456   | 24.1% |
| probe_test | 3     | 0.2%  |


## 4. Veto Counts by Entry Regime

Shows regime-conditioned veto density.

| Regime      | Count | %     |
|-------------|-------|-------|
| unknown     | 1485  | 78.4% |
| MEAN_REVERT | 410   | 21.6% |


## 5. Symbol × Reason Matrix

Cross-tabulation showing which symbols hit which constraints.

| Symbol  | Reason               | Count |
|---------|----------------------|-------|
| BTCUSDT | symbol_cap           | 512   |
| SOLUSDT | symbol_cap           | 415   |
| ETHUSDT | symbol_cap           | 396   |
| SOLUSDT | min_notional         | 203   |
| BTCUSDT | min_notional         | 189   |
| ETHUSDT | min_notional         | 111   |
| BTCUSDT | portfolio_dd_circuit | 12    |
| BTCUSDT | leverage_cap         | 8     |
| BTCUSDT | correlation_cap      | 6     |
| BTCUSDT | max_trade_nav        | 5     |
| BTCUSDT | kill_switch          | 4     |
| BTCUSDT | per_trade_cap        | 4     |
| BTCUSDT | open_notional_cap    | 4     |
| BTCUSDT | nav_stale            | 4     |
| ETHUSDT | whitelist            | 4     |
| BTCUSDT | daily_loss           | 4     |
| BTCUSDT | cooldown             | 4     |
| BTCUSDT | circuit_breaker      | 4     |
| ETHUSDT | correlation_cap      | 4     |
| ETHUSDT | nav_stale            | 2     |


## 6. Strategy × Reason Matrix

Shows which constraint dimensions each head pushes against.

| Strategy   | Reason               | Count |
|------------|----------------------|-------|
| unknown    | symbol_cap           | 867   |
| unknown    | min_notional         | 503   |
| vol_target | symbol_cap           | 456   |
| unknown    | portfolio_dd_circuit | 12    |
| unknown    | leverage_cap         | 8     |
| unknown    | correlation_cap      | 7     |
| unknown    | nav_stale            | 6     |
| unknown    | max_trade_nav        | 5     |
| unknown    | kill_switch          | 4     |
| unknown    | per_trade_cap        | 4     |
| unknown    | open_notional_cap    | 4     |
| unknown    | whitelist            | 4     |
| unknown    | daily_loss           | 4     |
| unknown    | cooldown             | 4     |
| unknown    | circuit_breaker      | 4     |
| probe_test | correlation_cap      | 3     |


## 7. Regime × Reason Matrix

Shows how feasible region size varies by regime.

| Regime      | Reason               | Count |
|-------------|----------------------|-------|
| unknown     | symbol_cap           | 913   |
| unknown     | min_notional         | 503   |
| MEAN_REVERT | symbol_cap           | 410   |
| unknown     | portfolio_dd_circuit | 12    |
| unknown     | correlation_cap      | 10    |
| unknown     | leverage_cap         | 8     |
| unknown     | nav_stale            | 6     |
| unknown     | max_trade_nav        | 5     |
| unknown     | kill_switch          | 4     |
| unknown     | per_trade_cap        | 4     |
| unknown     | open_notional_cap    | 4     |
| unknown     | whitelist            | 4     |
| unknown     | daily_loss           | 4     |
| unknown     | cooldown             | 4     |
| unknown     | circuit_breaker      | 4     |


## 8. Daily Veto Trend

| Date       | Count |
|------------|-------|
| 2026-02-01 | 660   |
| 2026-02-02 | 583   |
| 2026-02-03 | 130   |
| 2026-02-04 | 329   |
| 2026-02-05 | 193   |


## 9. Tier Distribution

| Tier      | Count | %     |
|-----------|-------|-------|
| CORE      | 911   | 48.1% |
| unknown   | 492   | 26.0% |
| SATELLITE | 492   | 26.0% |


---

## Key Observations

### Constraint Pressure Analysis

### Constraint Geometry (Distance-to-Wall)

**Records with geometry data:** 253 / 1895

| Metric | Value |
|--------|-------|
| Avg Excess Notional | $148.15 |
| Max Excess Notional | $226.92 |
| Min Excess Notional | $5.61 |
| Total Shadow Feasible | $8,141.71 |
| Avg Shadow Feasible | $32.18 |
| Avg Overshoot % | 184.7% |
| Median Overshoot % | 28.3% |

**Interpretation:**

- Moderate overshoots → **sizing logic partially misaligned**
- Shadow feasible volume: **$8,141.71** could have been executed with clipping

1. **Dominant Constraint:** `symbol_cap` accounts for 69.8% of all vetoes
2. **Highest Pressure Symbol:** `BTCUSDT` with 760 vetoes (40.1%)
3. **Strategy Attribution Coverage:** 24.2% of vetoes have strategy metadata

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
