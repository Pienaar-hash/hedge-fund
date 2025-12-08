## 📌 v7 Testing Pack

Scope:  
This pack verifies the **v7 sizing contract** after the refactor:

- Screener owns sizing (gross_usd + qty).
- Executor is pass-through (no re-sizing, no local caps).
- Risk engine (risk_limits / RiskEngineV6) is the **only** cap authority.
- Shadow pipeline uses the same sized intent (no re-sizing).

---

## 1. Pre-checks

Before running tests:

1. `git status` is clean (or only intentional local changes).
2. `.env` points to the correct environment:

   - For CI / dev: `BINANCE_TESTNET=1`, `DRY_RUN=1`.
   - For live testnet box: `BINANCE_TESTNET=1`, `DRY_RUN=0`.
   - For prod: `BINANCE_TESTNET=0`, `DRY_RUN=0`.

3. Configs are in sync:

   - `config/strategy_config.json` uses the v7 15m momentum config.
   - `config/risk_limits.json` uses **fractional** caps where expected and is compatible with the v7 normalization (e.g. `trade_equity_nav_pct: 0.02` for 2%).

---

## 2. Core Unit / Contract Tests

Run these first:

```bash
# Core risk engine + limits
pytest tests/test_risk_limits.py -q
pytest tests/test_risk_caps_v7.py -q

# Screener/intent pipeline
pytest tests/test_signal_pipeline.py -q

# Executor sizing contract (pass-through from screener)
pytest tests/test_executor_sizing_contract.py -q

# Shadow parity tests
pytest tests/test_pipeline_v6_shadow.py -q  # or updated shadow test module
````

### Expected outcomes

* `test_risk_limits.py`

  * Nav freshness checks behave as expected (stale nav → veto).
  * Min_notional, per-symbol caps, trade_equity_nav_pct / max_trade_nav_pct all enforced **only at risk layer**.
* `test_risk_caps_v7.py`

  * Trade % caps computed off **normalized fractions** (0.02 → 2%).
  * Stacked caps (equity vs max_trade_nav_pct vs portfolio) behave predictably.
* `test_signal_pipeline.py`

  * Screener emits intents with:

    * `gross_usd > 0`
    * `qty > 0`
    * `per_trade_nav_pct` is a fraction; sizing context present.
* `test_executor_sizing_contract.py`

  * Executor does **not** change `gross_usd` or `qty` when present.
  * Per-trade nav pct is only used when gross is absent.
* Shadow tests

  * Shadow `size_decision` and telemetry mirror screener intent (no re-sizing in shadow).

---

## 3. Integration Tests (Local)

With `BINANCE_TESTNET=1`:

### 3.1 Strategy probe

```bash
python -m scripts.strategy_probe
```

Checklist:

* Output shows **NAV** and **per-symbol intents** (e.g. BTC/ETH/SOL).

* For at least one symbol, see:

  * `reasons: []` (signal-level veto list empty).
  * Risk summary shows either:

    * `eligible_for_risk > 0`, or
    * Clear veto reasons from **risk** (not from executor sizing).

* Log lines show:

  * `[screener] attempted=<n> emitted=<m>`
  * For emitted intents:

    * `gross_usd` is non-zero.
    * `qty` is non-zero.
    * `nav_used` and `price_used` appear in sizing context (if logged).

### 3.2 Manual screener run

From REPL:

```bash
python - << 'PY'
from execution.signal_screener import run_once

batch = run_once()
print(batch)
PY
```

Checklist:

* `attempted` equals the number of symbols/timeframes in the live universe.
* `emitted` is ≥ 1 under reasonable market conditions.
* Each intent includes:

  * `symbol`, `timeframe`, `signal`
  * `gross_usd`, `qty`, `leverage`
  * `per_trade_nav_pct`, `min_notional`

---

## 4. Live Executor Test (Testnet)

With supervisor running on testnet:

```bash
sudo supervisorctl status
sudo tail -f /var/log/hedge-executor.out.log
```

Checklist:

1. Executor loop:

   * `[exutil] [executor] account OK — futures=True testnet=True`
   * `positions.json updated` with correct n / len.
   * `[v7-runtime] state write complete` (or equivalent runtime tag) showing `nav=True positions=True risk=True symbol_scores=True`.

2. Screener → Executor:

   * `[screener] attempted=<n> emitted=<m> submitted=<k>`
   * `[screener->executor] { ... 'gross_usd': <value>, 'qty': <value>, ... }`
   * No executor-side re-sizing logs (no `size_model`, `RiskGate`, etc).

3. Risk decisions:

   * `[risk]` vetos include reasons like `trade_gt_equity_cap`, `symbol_cap`, `max_portfolio_gross`, etc.
   * No references to RiskGate gross caps.

4. Exchange:

   * For test trades that pass risk, Binance testnet shows corresponding futures orders and positions.

---

## 5. Smoke Tests (Prod Simulation / DRY_RUN)

With `DRY_RUN=1` and `BINANCE_TESTNET=0` (prod market data, no live orders):

1. Start executor in foreground or via supervisor.
2. Confirm:

   * NAV pulls from the correct (prod) account snapshot.
   * Screener/intent flow still works (attempted/emitted).
   * Risk vetos align with prod nav and caps.

No real orders should be submitted in DRY_RUN mode.

---

## 6. Sign-off Criteria

* All core tests pass (`test_risk_limits`, `test_risk_caps_v7`, `test_signal_pipeline`, `test_executor_sizing_contract`, shadow tests).
* Strategy probe emits intents with clean reasons and correct sizing.
* Executor on testnet successfully:

  * Receives sized intents,
  * Passes them to risk,
  * Places actual testnet trades when risk allows.
* No log references to old sizing paths (`size_model`, `RiskGate`), and no unexpected below_min_notional vetos due to sizing mismatches.

Once all of the above is green, v7 sizing is considered **runtime-verified** for testnet.

