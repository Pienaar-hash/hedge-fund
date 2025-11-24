## 📌 v7 Runtime Verification Checklist

Goal:  
Confirm that **live runtime** (testnet or prod) is consistent with the v7 sizing contract:

> Screener sizes → Executor passes through → Risk engine applies caps → Router submits.

---

## 1. Environment & Process Health

### 1.1 Environment flags

Check:

```bash
grep -E 'BINANCE_TESTNET|DRY_RUN|ENV' .env
````

Confirm:

* Testnet sandbox:

  * `ENV=prod` (or `ENV=testnet` depending on convention)
  * `BINANCE_TESTNET=1`
  * `DRY_RUN=0` (for real testnet orders) or `1` (for dry-run).
* Production:

  * `BINANCE_TESTNET=0`
  * `DRY_RUN=0`

### 1.2 Supervisor status

```bash
sudo supervisorctl status
```

Required:

* `hedge:hedge-executor` → RUNNING
* `hedge:hedge-dashboard` → RUNNING
* `hedge:hedge-sync_state` → RUNNING
* `hedge:hedge-pipeline-shadow-heartbeat` → RUNNING
* `hedge:hedge-pipeline-compare` → RUNNING

---

## 2. NAV & State Sanity

### 2.1 NAV snapshot

Tail executor log:

```bash
sudo tail -n 50 /var/log/hedge-executor.out.log
```

Expected:

* `[nav] snapshot ts=... nav_usd=<value> sources_ok=True path=logs/cache/nav_confirmed.json`
* nav_health / nav_health_snapshot show:

  * `fresh=True`
  * `age_s` under the configured threshold.
* `v6-runtime`/`v7-runtime` or similar state-line:

  * `state write complete state_dir=logs/state nav=True positions=True risk=True symbol_scores=True synced=True`

### 2.2 Positions/State files

Check JSONs:

```bash
ls -1 logs/state
cat logs/state/nav_state.json
cat logs/state/positions_state.json
cat logs/state/risk_state.json
```

Confirm:

* `nav_state.json` has `nav_total` consistent with exchange.
* `positions_state.json` reflects actual positions (should be empty when flat).
* `risk_state.json` has valid `daily_peak`, `nav`, `dd_pct`, etc.

---

## 3. Screener → Executor Flow

### 3.1 Screener logs

In executor log:

* Look for `[sigdbg] sym=... tf=...` followed by `[screener] ...`.

* Expect:

  * `per_trade_nav_pct` showing the fraction (e.g. 0.02 for 2%).
  * `min_notional`, `leverage`, `open_gross`.

* After decisions:

  * `[screener] attempted=<n> emitted=<m> submitted=<k>`

### 3.2 Intent payloads

Look for:

```text
[exutil] [screener->executor] {...}
```

Confirm each intent has:

* `symbol`, `timeframe`, `signal`
* `gross_usd > 0` (or 0 if using pure nav_pct/min_notional path intentionally)
* `qty > 0` where applicable
* `per_trade_nav_pct` as fraction
* `min_notional` matching symbol/exchange floors

Executor must **not** re-compute gross using size_model or RiskGate.

---

## 4. Risk Engine Behavior

### 4.1 Risk vetos

In logs, find lines like:

```text
[decision] {"symbol": "...", "tf": "...", "veto": [...], "detail": {...}}
```

Confirm:

* Veto reasons for trade caps come **only** from risk (e.g. `trade_gt_equity_cap`, `trade_gt_max_trade_nav_pct`, `symbol_cap`, `max_portfolio_gross`).
* Observed `trade_equity_nav_obs` values line up with configured percentages.

### 4.2 Stale NAV handling

Simulate or observe a stale NAV scenario (or rely on tests). At runtime:

* When nav age exceeds threshold:

  * `fresh=False`
  * Risk engine should veto further orders with a nav freshness reason.
* Screener/executor should **not** size off an obviously stale nav without risk blocking trades.

---

## 5. Shadow Pipeline & Router

### 5.1 Shadow heartbeat

Check:

```bash
sudo tail -n 50 /var/log/hedge-pipeline-shadow-heartbeat.out.log
```

Expect:

* Heartbeat logs indicating successful pulls from live intents.
* For each symbol/timeframe, shadow telemetry shows:

  * `sized_gross_usd` equal to screener intent’s gross.
  * No independent re-sizing in shadow.

### 5.2 Router parity

If router telemetry is enabled:

* Verify:

  * Router receives orders with the same `gross_usd` and `qty` that the executor uses.
  * Maker/taker decisions and offsets are independent of sizing logic.

---

## 6. Exchange-Level Confirmation (Testnet / Prod)

For testnet:

1. Let the system run until you see **at least one non-vetoed trade**.
2. Check Binance Futures testnet:

   * Positions page shows the opened position(s).
   * Notional roughly equals intent `gross_usd` (subject to tick/lot rounding).
   * Leverage matches intent leverage.

For prod (optional & carefully):

* Use a very small per_trade_nav_pct and DRY_RUN if you just want to validate flow.
* When comfortable, enable live trades with standard caps.

---

## 7. Final Runtime Checklist

✔ Environment flags correct (`BINANCE_TESTNET`, `DRY_RUN`).
✔ NAV health fresh, `nav_total` matches exchange.
✔ Screener emits sized intents (gross_usd + qty) with sensible per_trade_nav_pct.
✔ Executor logs show pass-through of screener sizing (no local re-sizing).
✔ Risk engine vetos & allows trades with consistent reasons (no ghost caps from executor).
✔ Shadow pipeline mirrors screener sizing only.
✔ Exchange reflects trades that pass risk, with notional ≈ intent gross.

When all boxes are ticked on testnet (and ideally again on mainnet DRY_RUN), v7 runtime is verified.
