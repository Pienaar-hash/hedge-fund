## 🎯 Patch Goal

**Goal:**
Decouple **leverage** from **screener sizing** so that:

* Screener sizes **purely in unlevered notional** (NAV × per_trade_nav_pct or fixed notional).
* Leverage is treated as **metadata / constraint**, not as a multiplier of notional.
* Risk engine (risk_limits / RiskEngineV6) is the **only place** where caps based on NAV and leverage are enforced.

We are *not* redesigning the whole sizing system here. We are only:

* Removing leverage from the **screener’s sizing math**.
* Ensuring `gross_usd` and `qty` are **unlevered** in the screener → executor path.
* Keeping `leverage` as a field on the intent but not using it to scale size.

---

## 🔍 Scope

Focus only on:

* `execution/signal_screener.py`
* Any tests that **directly depend on leverage-based screener sizing** (e.g. where `gross_usd ≈ nav * per_trade_nav_pct * leverage` or where caps are computed using leverage).

Do **not** touch:

* `execution/executor_live.py` sizing (it should already be pass-through for gross / qty).
* `execution/risk_limits.py` / `execution/risk_engine_v6.py` caps other than necessary test updates.
* Router, exchange utils, or dashboard code.

---

## 1. Remove Leverage from Screener Sizing

### 1.1 Find the sizing block

In `execution/signal_screener.py`, locate the core **sizing block** that:

* Pulls `nav_used` from a nav snapshot (often via `nav_health_snapshot` or similar).
* Computes some variant of:

  * `per_trade_nav_pct` and/or `capital_per_trade`
  * `min_notional` (floors: symbol, exchange, qty*price)
  * `requested_notional` or similar
  * `gross_usd` and `qty`
* Reads a leverage value, usually via:

  * `lev = symbol_target_leverage(...)` or similar helper, or
  * `leverage = intent.get("leverage")` or config-driven.

You’ll see patterns like:

* `gross_cap = cap_cfg * lev`
* `requested_notional = ...`
* `gross_request = requested_notional * lev` or similar
* Trade caps / floors using `lev` in their math.

### 1.2 Change requested_notional to be **unlevered**

Where the screener computes the **requested size** for a trade, make sure it’s **unlevered** and does *not* include leverage:

**Before (levered behaviour – conceptual):**

```python
# Conceptual example – do not copy literally
per_trade_nav_pct = intent.get("per_trade_nav_pct", 0.0)
lev = symbol_target_leverage(sym, tier, config, default=3.0)

base_notional = nav_used * per_trade_nav_pct  # fraction of NAV
requested_notional = max(base_notional, min_notional)

# Leverage applied here (this is what we are removing)
gross_request = requested_notional * lev
```

**After (unlevered):**

```python
per_trade_nav_pct = intent.get("per_trade_nav_pct", 0.0)

base_notional = nav_used * per_trade_nav_pct
requested_notional = max(base_notional, min_notional)

# No leverage application here – this is the final notional used.
gross_request = requested_notional
```

If there are paths where `capital_per_trade` is used:

* Treat `capital_per_trade` as **unlevered notional** too.
* Do **not** multiply it by leverage.

---

## 2. Don’t Use Leverage in Caps / Floors Inside Screener

### 2.1 Remove leverage-based caps in screener

Find any caps / floors in the screener that multiply by leverage. Examples (patterns, not exact code):

```python
gross_cap = cap_cfg * lev
floor_notional = min(max(gross_cap, min_notional), sym_max_order)
...
effective_notional = min(requested_notional, gross_cap)
```

Change these to **ignore leverage entirely**:

```python
gross_cap = cap_cfg  # cap_cfg is now interpreted as an unlevered notional cap
floor_notional = min(max(gross_cap, min_notional), sym_max_order)
effective_notional = min(requested_notional, gross_cap)
```

If you see a final `cap` computed as:

```python
cap = requested_notional / lev if lev > 0 else requested_notional
```

* Remove the leverage division altogether and use the unlevered notional:

```python
cap = requested_notional
```

If some of this “cap” is only for logging, you may simply log `requested_notional` as the cap and never divide or multiply by leverage.

### 2.2 Keep leverage only as metadata

Make sure leverage is still carried through for later layers to use, but **never used in sizing math**:

* In the final emitted intent, keep something like:

```python
"leverage": lev
```

but ensure that:

* `gross_usd` and `qty` **do not change** if leverage changes.
* No cap or floor inside screener recomputes `gross_usd` or `qty` based on leverage.

---

## 3. Make Intents Consistently Unlevered

In the final `intent` dict the screener sends to the executor (visible in logs as `[screener->executor] {...}`), ensure:

```python
intent = {
    ...
    "price": price_used,
    "gross_usd": gross_usd,   # unlevered notional
    "qty": qty,               # gross_usd / price_used
    "nav_used": nav_used,
    "nav_age_s": nav_age_s,
    ...
    "per_trade_nav_pct": per_trade_nav_pct,
    "leverage": lev,          # passthrough only
    ...
}
```

**Key invariant after patch:**

* For all **non-reduceOnly** trades with `per_trade_nav_pct > 0` and fresh NAV:

  ```python
  gross_usd ≈ max(nav_used * per_trade_nav_pct, min_notional)
  qty ≈ gross_usd / price_used
  ```

* `gross_usd` must **not** be `nav_used * per_trade_nav_pct * leverage`.

---

## 4. Adjust / Add Tests

Update or create tests so that they **explicitly assert** leverage is not used in screener sizing.

Pick or create a test file like `tests/test_screener_sizing_no_leverage.py` (or extend an existing screener sizing test) with patterns like:

```python
def test_screener_sizing_ignores_leverage(monkeypatch, screener_ctx_eth):
    """
    For a given per_trade_nav_pct and nav, gross_usd should be
    per_trade_nav_pct * nav (floored), regardless of leverage.
    """

    # Arrange
    nav_used = 10_000.0
    per_trade_nav_pct = 0.025  # 2.5% of NAV
    leverage = 3.0             # should NOT change gross_usd
    min_notional = 20.0
    price = 2_500.0

    # Force nav snapshot used by screener
    # monkeypatch nav_health_snapshot or screener's nav retrieval to return nav_used

    # Build a fake signal / intent that the screener consumes
    # Ensure it includes per_trade_nav_pct and leverage.

    # Act
    # Run one screener sizing step (or call the internal sizing helper if exposed).

    # Assert
    expected_notional = max(nav_used * per_trade_nav_pct, min_notional)
    assert math.isclose(intent["gross_usd"], expected_notional, rel_tol=1e-6)
    assert math.isclose(intent["qty"], expected_notional / price, rel_tol=1e-6)

    # Also assert that changing leverage does not change the size:
    # e.g. rerun with leverage=1 and leverage=10 if cheap to do so.
```

Also, update existing tests that currently expect:

* `gross_usd ≈ nav * per_trade_nav_pct * leverage`, or
* trade caps in screener using leverage.

Change them to expect **unlevered** behaviour.

---

## 5. Sanity Checks (Manual / Log-Level)

Once Codex has applied the patch:

1. **Restart executor / screener** (your usual supervisorctl flow).

2. Run in **DRY_RUN** and inspect `[screener->executor]` logs:

   * Confirm for ETH/SOL/BTC:

     * `gross_usd` ≈ `nav_used * per_trade_nav_pct` (floored to min_notional).
     * `qty` ≈ `gross_usd / price_used`.
     * Changing `leverage` in config does **not** change `gross_usd` or `qty`.
   * Confirm risk blocks like `trade_gt_equity_cap` now reflect **unlevered notional**, consistent with risk_limits caps.

3. Ensure there are **no warnings** or crashes from removed leverage-based fields.

---

## 6. Non-Goals / Guardrails

* Do **not** reintroduce `size_model` or any legacy sizing module.
* Do **not** add new caps or move caps out of `risk_limits` / `RiskEngineV6`.
* Do **not** alter executor’s pass-through contract:

  * Executor should still treat `gross_usd` / `qty` as **final** and only normalize to exchange filters + risk.
