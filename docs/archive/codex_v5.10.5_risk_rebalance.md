## 🧠 Codex Audit + Patch Prompt — v5.10.5 Risk Rebalance

You are my Quant Infra & Risk Engineer working on a live Binance USD-M futures hedge-fund repo.

### Context

* We are on **v5.10** with:

  * Hardened risk limits (`execution/risk_limits.py` + `config/risk_limits.json`)
  * Structured `(veto, detail)` returns from `check_order`
  * Screener emitting structured reasons (`execution/signal_screener.py`, `scripts/screener_probe`)
  * Concurrency gates and per-symbol caps wired through `execution/executor_live.py` → `execution/size_model.py`

* Current behaviour:

  * `scripts.screener_probe` shows for BTCUSDT/ETHUSDT:

    * `would_emit = false`
    * `reasons = ["leverage_exceeded", "trade_gt_max_trade_nav_pct", "trade_gt_10pct_equity"]`
  * NAV ≈ **$4,400**.
  * Result: **no trades ever pass risk**, even though the logic is correct.

I want you to **rebalance the risk posture** from “ultra-conservative” to “moderately active but still safe”.

---

### Goal

Produce a patch that makes it realistically possible for BTCUSDT / ETHUSDT / SOLUSDT trades to pass the screener under normal volatility, while:

* Preserving all existing safety guarantees.
* Keeping overall portfolio risk bounded.
* Maintaining the structured `(bool, detail)` contract everywhere.

Specifically, rebalance these levers:

1. **`max_trade_nav_pct`** (per-trade cap as % of NAV)
2. **`per_symbol.max_nav_pct`** (per-symbol NAV cap)
3. **`per_symbol.max_leverage`**
4. **The screener’s 10% equity clamp** (the “trade_gt_10pct_equity” style gate)
5. **Concurrency gates** (per-strategy `max_concurrent_positions` and global concurrency cap)

You must **not** loosen anything by commenting out guards or bypassing checks: always modify caps/thresholds, never the guard itself.

---

### Target Risk Profile (Concrete Numbers)

Use these as the target configuration unless you find something inconsistent in the code:

1. **Global per-trade cap:**

   * `max_trade_nav_pct = 0.2` (20% of NAV per trade, i.e. ≈ $880 at $4,400 NAV)

2. **Per-symbol NAV caps (`per_symbol.max_nav_pct`):**

   * BTCUSDT: `0.25` (25% of NAV)
   * ETHUSDT: `0.20`
   * SOLUSDT: `0.15`

3. **Per-symbol leverage caps (`per_symbol.max_leverage`):**

   * BTCUSDT: `4`
   * ETHUSDT: `4`
   * SOLUSDT: `3`

4. **Screener “10% equity” clamp:**

   * Keep the same **gate**, but relax the threshold so that it **only vetoes trades above 15% of NAV**.
   * In other words, let trades up to 15% NAV pass this specific check, subject to all other limits.

5. **Concurrency:**

   * Ensure **every live strategy** (BTC/ETH/SOL) has:

     * `max_concurrent_positions >= 3` in `strategy_config`
   * Ensure there is a **global portfolio concurrency cap** that effectively limits:

     * Total **open positions** to something like **4–5 max** (e.g. “no more than 4 simultaneous symbols/legs”).
   * Concurrency gates should never be “infinite” or silently disabled.

If you discover existing constants for any of these values, **update them in one place** and reuse those constants/cfg fields rather than duplicating numbers.

---

### Files to Inspect and Patch

Focus on these (do not touch unrelated subsystems):

* **Config**

  * `config/risk_limits.json`
  * `config/strategy_config.json`

* **Risk & sizing**

  * `execution/risk_limits.py`
  * `execution/size_model.py`

* **Screener / intents**

  * `execution/signal_screener.py`
  * `scripts/screener_probe.py` (if needed for better debug output)

* **Executor integration**

  * `execution/executor_live.py` (only if you need to adjust how concurrency caps / risk limits are passed into the sizer or screener)

Do **not** change:

* Any Binance REST signing logic
* Exchange parameter whitelists or reduceOnly / closePosition plumbing
* Router policy or maker/taker offsets

---

### Behaviour Requirements

After your patch:

1. **BTCUSDT / ETHUSDT / SOLUSDT**:

   * For a realistic market snapshot and NAV ≈ $4,400, **at least one of BTCUSDT or ETHUSDT should sometimes pass**:

     * `leverage_exceeded` should not always fire.
     * `trade_gt_max_trade_nav_pct` should not always fire.
     * `trade_gt_10pct_equity` should not fire unless the proposed notional really exceeds 15% of NAV.

2. **Gates remain fully active:**

   * If NAV drops heavily or volatility spikes, the same gates **must** be able to veto trades again.
   * No gate may be bypassed or neutered; only thresholds and caps can change.

3. **Concurrency discipline:**

   * The system must never be able to open more than the intended number of concurrent positions (e.g. 4–5) even if all three symbols have active strategies.
   * `max_concurrent_positions` must not be zero or undefined; it should be explicit and enforced.

4. **Detail payloads:**

   * `check_order` and screener `would_emit()` must continue returning `(bool, detail)` with:

     * `detail["reasons"]` (list of veto reasons when vetoed)
     * `detail["nav_fresh"]` correctly set
     * Any updated thresholds reflected in the detail fields you already use for tests.

---

### Tests & Validation

Update or add tests as needed, but these **must pass**:

```bash
python -m pytest \
  tests/test_risk_limits.py \
  tests/test_screener_tier_caps.py \
  tests/test_config_parsing.py \
  tests/test_exchange_dry_run.py \
  tests/test_router_smoke.py \
  tests/test_order_router_ack.py \
  tests/test_order_metrics.py -q
```

Add or adjust tests so that:

1. There is at least one **positive** test scenario where, given:

   * NAV ≈ $4,400
   * A proposed BTCUSDT / ETHUSDT trade with reasonable notional
   * The gate returns `(False, detail)` (no veto) and **does not include** the three previous reasons:

     * `"leverage_exceeded"`
     * `"trade_gt_max_trade_nav_pct"`
     * `"trade_gt_10pct_equity"`

2. There is at least one **negative** test where:

   * A trade > 20% NAV is vetoed by `max_trade_nav_pct` and/or the screener’s equity clamp, with clear reasons.

3. `scripts.screener_probe`:

   * Should run cleanly and print structured JSON entries for BTCUSDT/ETHUSDT/SOLUSDT that include:

     * `would_emit`
     * `reasons`
     * `max_concurrent_positions`
   * It should be possible (in at least one test fixture) to see `would_emit=true` for a CORE symbol.

---

### Implementation Constraints

* Keep the design **config-driven**:

  * If a threshold can live in `config/risk_limits.json` or `config/strategy_config.json`, put it there and read it in.
  * Avoid introducing magic numbers inside code paths unless strictly necessary.

* Do not change or remove any of the following existing guarantees:

  * NAV freshness checks
  * Min notional enforcement for Binance
  * Per-symbol caps wiring (from config → risk_limits → size_model → screener)
  * The `(veto, detail)` function signatures

* Where you modify behaviour, **augment the log messages** rather than silencing them:

  * Logs should still make it clear *which* gate vetoed and *with what* thresholds.

---

### Deliverables

1. A **single unified diff** (or a clear patch summary) over the files listed above.
2. Brief comments in the diff explaining:

   * Where `max_trade_nav_pct`, `max_nav_pct`, `max_leverage`, and the screener equity clamp are defined and enforced.
3. Confirmation in a short summary that:

   * Tests listed above pass.
   * At least one BTCUSDT/ETHUSDT scenario can now pass the screener in normal conditions.
