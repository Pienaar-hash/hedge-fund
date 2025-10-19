 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/docs/quant_audit.md b/docs/quant_audit.md
new file mode 100644
index 0000000000000000000000000000000000000000..dfe6a14a65c5c875e9dae1c9b4d52f7a45e36bcf
--- /dev/null
+++ b/docs/quant_audit.md
@@ -0,0 +1,40 @@
+# Quant Audit — Hedge Fund Stack
+
+## Executive Summary
+- The production risk gate currently continues trading on fallback capital values whenever live NAV data is unavailable, which can materially understate leverage and drawdown after exchange outages. Tighten the fail-closed path so capital-dependent guards halt order flow until fresh balances and positions arrive.
+- Daily loss protections rely on a manually maintained `peak_state.json` file instead of realized PnL, leaving the program blind to intraday drawdowns and increasing the probability of breaching investor mandates. Wire the risk state to live PnL snapshots and rework peak tracking to survive restarts automatically.
+- Strategy sizing, research, and machine-learning components remain static and shallow relative to market dynamics (fixed $5 notional clips, no volatility targeting, and single split validation). Introduce volatility-aware allocators, walk-forward testing, and intent-level performance filters before scaling capital.
+
+## Detailed Findings
+
+### 1. NAV-driven risk controls fail open when data is stale
+**Observation.** `RiskGate._portfolio_nav` falls back to `capital_base_usdt` when live wallet calls fail, so subsequent gross exposure and trade size checks use a static config value instead of verifiable equity.【F:execution/risk_limits.py†L441-L475】 The upstream NAV helper similarly drops to the same fallback whenever the futures wallet returns zero or errors.【F:execution/nav.py†L109-L121】 In aggregate, a Binance outage would preserve the last configured capital (often far above reality after losses) and allow new positions to open.
+
+**Recommendation.** Fail closed whenever balances or positions cannot be refreshed (set nav to zero, trip a circuit breaker, and surface telemetry). Persist the last confirmed NAV with timestamps and require freshness before admitting new orders. Couple the guard with an alerting channel so humans know trading stopped.
+
+### 2. Daily loss checks depend on manual peak files
+**Observation.** The per-day loss limiter reads `RiskState.daily_pnl_pct`, but the canonical `RiskGate` recomputes that value from `peak_state.json`, a local file updated only by the Firestore sync pipeline.【F:execution/risk_limits.py†L264-L275】【F:execution/risk_limits.py†L523-L538】 The sync process reconstructs peak equity by combining the Firestore document and the same file, so a cold restart or file corruption resets the peak and allows fresh trading without honoring prior drawdowns.【F:execution/sync_state.py†L380-L414】 There is no linkage to realized PnL or exchange-reported daily loss, leaving the control vulnerable to drift.
+
+**Recommendation.** Replace the file-based peak estimator with an exchange-derived realized PnL feed (wallet `cumRealizedPnl` or trade ledger). Persist peak equity in a durable datastore with monotonic updates and reload it on startup. Feed the intraday realized/mark-to-market drawdown directly into `RiskState` so the risk gate can enforce loss limits even if auxiliary services lag.
+
+### 3. Position sizing ignores volatility and cross-strategy concentration
+**Observation.** The global config drives every strategy to trade roughly $5–$18 gross notional per order with static leverage maps and tight gross exposure caps, regardless of recent volatility or correlation.【F:config/strategy_config.json†L13-L37】 The live momentum module likewise ranks signals without scaling by realized risk, then emits trades without accounting for slippage, borrow, or cross-asset correlation.【F:strategies/momentum.py†L29-L103】 Because multiple symbols are enabled, these static clips can overshoot the configured gross limits as volatility spikes, while also starving high-sharpe regimes when risk is low.
+
+**Recommendation.** Introduce a volatility-targeted allocator that scales clip size by recent ATR or variance and enforces portfolio-level risk budgets per asset class. Layer in correlation-aware throttles so simultaneous longs on highly correlated alts share the same gross cap. Backfill execution cost models (slippage, fees) when computing expected edge so marginal trades can be filtered before routing.
+
+### 4. Machine-learning pipeline lacks robustness checks
+**Observation.** Training uses a single chronological split with a basic logistic regression, storing the resulting model immediately after a lone ROC-AUC calculation.【F:execution/ml/train.py†L38-L90】 Feature engineering feeds from overlapping lookbacks and drops rows with NaNs wholesale, effectively shrinking the sample without testing stability under different periods.【F:execution/ml/features.py†L24-L42】 There is no walk-forward validation, probability calibration, or monitoring of live hit rate post-deployment.
+
+**Recommendation.** Move to rolling walk-forward or cross-validation splits that respect time ordering, add probability calibration, and record out-of-sample hit rate plus feature drift metrics in the model registry. Require a minimum lift over baseline and statistical confidence before refreshing the live model, and set up retraining alarms when performance decays.
+
+### 5. Signal routing lacks performance or capacity gates
+**Observation.** The signal generator simply loads a hard-coded list of modules, deduplicates by `(symbol, timeframe, side, candle_close)`, and forwards every surviving intent downstream.【F:execution/signal_generator.py†L11-L109】 There is no weighting by historical PnL, risk score, or order book liquidity, nor any suppression after consecutive failures. Static module lists also make it hard to sandbox new strategies safely.
+
+**Recommendation.** Introduce a registry that tags each strategy with capacity limits, confidence scores, and kill-switch states. Feed live performance analytics into the generator so underperforming or volatile modules self-throttle. Support per-intent metadata (expected edge, required liquidity) so the risk gate can blend sizing decisions with signal quality before firing orders.
+
+## Next Steps
+1. Implement fail-closed NAV freshness checks in the risk gate and executor loop, including telemetry and operator runbooks.
+2. Build a resilient drawdown tracker sourced from exchange PnL and wire it through `RiskState` so daily loss stops trip deterministically.
+3. Launch a research sprint on volatility-aware sizing and correlation budgeting, then backtest and phase into production behind feature flags.
+4. Expand the ML pipeline with walk-forward evaluation, calibrated probabilities, and performance monitoring hooks.
+5. Refactor the signal generator to consume a strategy registry that enforces capacity, sandboxing, and auto-suspension rules. 
EOF
)