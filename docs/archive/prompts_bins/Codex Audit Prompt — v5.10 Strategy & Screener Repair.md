# ✅ **Codex Audit Prompt — v5.10 Strategy & Screener Repair (Targeted Scope Only)**

**ROLE:**
You are the GPT-Hedge *v5.10 Quant & Infra Auditor*.
Your scope is restricted to the four repair areas listed in **TASKS** below.
Do **NOT** modify unrelated systems, files, or semantics.
Do **NOT** attempt architecture changes, refactors, or performance tuning.
Do **NOT** reintroduce USDC logic, rollback v5.10 patches, or mutate the router, executor, or exchange layers.

---

# **TARGET FILE SET**

Inspect and potentially patch only these files:

```
config/strategy_config.json
config/risk_limits.json
execution/size_model.py
execution/signal_screener.py
execution/risk_limits.py
execution/executor_live.py
```

If additional files appear relevant, flag them but **do not modify** unless absolutely required by one of the four tasks.

---

# **TASKS (STRICT SCOPE)**

## **1. Fix max_concurrent_positions**

* Identify every strategy in `strategy_config.json` whose `max_concurrent_positions` is incorrectly set to `0` or `null`.
* Set appropriate defaults:

  * Majors (BTCUSDT, ETHUSDT): `1`
  * High-liquidity alts (SOLUSDT): `1`
* Ensure the executor and screener interpret this properly.
* Validate via unit tests that strategies can emit at least one intent.

---

## **2. Align strategy sizing with risk limits**

* Compare live sizer output (`execution/size_model.py`) with per-symbol caps in `config/risk_limits.json`.
* Fix inconsistencies such that:

  * Suggested position size from sizing engine **never exceeds**:

    * `per_symbol.max_order_notional`
    * `per_symbol.max_nav_pct`
    * `global.max_gross_exposure_pct`
    * `global.max_symbol_exposure_pct`
* If the sizer attempts to exceed risk caps, enforce an internal clamp (NOT a veto).
* Veto should occur only when:

  * clamped size < exchange minNotional
* Update tests where needed, but keep original behaviors intact unless contradicted by v5.10 risk-limit semantics.

**Do not change:** order_router, executor open/close logic, hedging semantics, reduceOnly logic.

---

## **3. NAV freshness window**

* Current `nav_freshness_seconds` is set to `90`.
* The screener frequently experiences `nav_stale` because NAV updates slightly exceed this threshold.
* Set a safe production value between `120` and `180`.
* Ensure `risk_limits.check_order` correctly populates:

  * `detail_payload["nav_fresh"] = True/False`
* Ensure that stale NAV causes:

  * veto *only* when `fail_closed_on_nav_stale = true`
  * warning + allow *when* `fail_closed_on_nav_stale = false`

**Do not modify NAV writer or Coingecko module.**

---

## **4. Full screener / intent emission loop repair**

* The screener should:

  * generate intents when signals are valid
  * avoid emitting when vetoed correctly
  * produce structured `(intent, detail)` logs
  * never crash when gates return malformed or missing data

* Verify `would_emit()` uses correct ordering:

  1. strategy enabled?
  2. sufficient NAV?
  3. under max_concurrent_positions?
  4. sizing clamps applied
  5. risk_limits.check_order
  6. intent constructed

* Confirm all return contracts are consistent across:

  * `signal_screener.py::would_emit`
  * `executor_live.py::generate_intents`
  * `risk_limits.py::check_order`

* Ensure no missing keys (e.g., detail_payload crash scenarios).

**Do not** change signal models or strategy rules themselves.

---

# **NON-TASKS (ABSOLUTE BLOCKLIST)**

Codex must **not** modify:

* exchange_utils (except for sizing or guard bugs downstream of check_order)
* order_router or routing policy
* USDC fallback logic (already removed)
* pairs_universe merging logic
* universe_resolver or tiers
* execution semantics of reduceOnly or positionSide
* NAV computation or firestore publishing
* ML models or signal models
* any v6.0 features, migrations, or architecture changes

If Codex detects issues *outside scope*, report them in the summary but **do not patch**.

---

# **OUTPUT REQUIREMENTS**

Codex must output:

1. **Unified diff patch** touching only the allowed files.
2. **A short summary** explaining each change with references (file:line).
3. **Confirmation that all unit tests pass**, OR provide updated test patches if required.
4. **No stylistic refactors**—only functional repairs required by the four tasks.