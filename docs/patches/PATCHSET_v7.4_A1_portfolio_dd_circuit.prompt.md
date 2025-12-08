# PATCHSET v7.4_A1 — Portfolio Drawdown Circuit Breaker

## Objective

Introduce a **portfolio-level drawdown circuit breaker** that:

- Tracks portfolio drawdown as a percentage of peak NAV.
- Vetoes **all new orders** once drawdown exceeds a configured NAV percentage.
- Publishes circuit status to the risk state for dashboard consumption.
- Adds a clear, structured veto reason: `"portfolio_dd_circuit"`.

This must **not** break any existing v7 contracts for risk, state, or telemetry.

---

## Files to Touch

- `config/risk_limits.json`
- `execution/risk_loader.py`
- `execution/drawdown_tracker.py`
- `execution/risk_limits.py`
- `execution/state_publish.py`
- `dashboard/*` (minimal risk panel update, e.g. `dashboard/risk_panel_v7.py` or equivalent)
- `tests/test_risk_limits_portfolio_dd.py` (new) or extend existing risk tests

Do **not** touch executor entrypoints or router unless absolutely necessary.

---

## 1. Config: Add Circuit Breaker Threshold

**File:** `config/risk_limits.json`

### Change

1. Add a new top-level block `circuit_breakers` (if not present) with:

```json
"circuit_breakers": {
  "max_portfolio_dd_nav_pct": 0.10
}
````

* Value is a **fraction** (0.10 = 10%) consistent with existing percentage normalization rules.
* Keep it optional and safe: if missing or `null`, the circuit breaker should behave as **disabled**.

---

## 2. Risk Loader: Normalize Circuit Breaker Value

**File:** `execution/risk_loader.py`

### Change

1. Extend the config loading/normalization logic to:

* Read `circuit_breakers.max_portfolio_dd_nav_pct`.
* Normalize it using the same helper used for other percentage configs, e.g. `normalize_percentage`.

Pseudo-logic:

```python
def load_risk_limits(...) -> RiskLimitsConfig:
    ...
    cb_cfg = raw.get("circuit_breakers", {}) or {}
    max_dd = cb_cfg.get("max_portfolio_dd_nav_pct")

    if max_dd is not None:
        max_dd = normalize_percentage(max_dd)

    circuit_breakers = CircuitBreakerConfig(
        max_portfolio_dd_nav_pct=max_dd
    )
    ...
```

2. If the value is missing, leave `max_portfolio_dd_nav_pct` as `None` — this should mean **no circuit breaker active**.

3. Make sure `RiskLimitsConfig` (or equivalent config dataclass) contains a `circuit_breakers` field with `max_portfolio_dd_nav_pct: Optional[float]`.

No behaviour changes for any other risk fields.

---

## 3. Drawdown Tracker: Expose Portfolio DD State

**File:** `execution/drawdown_tracker.py`

### Change

1. Introduce a small dataclass:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class PortfolioDDState:
    current_dd_pct: float          # e.g. 0.032 for 3.2%
    peak_nav_usd: float
    latest_nav_usd: float
```

2. Add a helper function:

```python
def get_portfolio_dd_state(nav_history: Sequence[float]) -> Optional[PortfolioDDState]:
    """
    Given a sequence of NAV observations (in USD), compute:
    - peak NAV
    - latest NAV
    - current drawdown percentage from peak

    Return None if nav_history is empty or invalid.
    """
```

* Behaviour:

  * If `nav_history` is empty or all non-positive → return `None`.
  * `peak_nav_usd = max(nav_history)`
  * `latest_nav_usd = nav_history[-1]`
  * `current_dd_pct = 0.0` if `peak_nav_usd <= 0`, else `(peak_nav_usd - latest_nav_usd) / peak_nav_usd`.

3. If there’s already a portfolio DD computation function, **extend it** instead of duplicating logic. The key is to expose a **stable API** `get_portfolio_dd_state(...)` that other modules can call.

4. Ensure unit tests for this module (later in this patch) cover:

* Monotonic increasing NAV → `current_dd_pct == 0.0`.
* NAV off peak by 10% → `current_dd_pct == 0.10` (within tolerance).
* Empty or invalid input → `None`.

---

## 4. Risk Limits: Add `portfolio_dd_circuit` Veto

**File:** `execution/risk_limits.py`

### Change

1. In or near `check_order(...)`, after reading NAV and before per-trade caps, insert a **portfolio circuit breaker** check.

2. Use the loaded config:

```python
max_dd = risk_limits_config.circuit_breakers.max_portfolio_dd_nav_pct
if max_dd is not None:
    dd_state = drawdown_tracker.get_portfolio_dd_state(nav_history)
    if dd_state is not None and dd_state.current_dd_pct >= max_dd:
        # emit veto
```

3. Emit a veto using the existing structured veto path (e.g. `_emit_veto(...)`), with:

* `veto_reason = "portfolio_dd_circuit"`
* Observed & limits:

```python
observed = {
    "current_dd_pct": dd_state.current_dd_pct,
    "peak_nav_usd": dd_state.peak_nav_usd,
    "latest_nav_usd": dd_state.latest_nav_usd,
}
limits = {
    "max_portfolio_dd_nav_pct": max_dd
}
```

4. Contract rules:

* If `max_dd` is `None` → **do nothing** (no circuit breaker).
* If `dd_state` is `None` → **do nothing** (do not accidentally block due to lack of data).
* If `current_dd_pct < max_dd` → do nothing, continue with existing checks.
* Only one veto should be emitted per order; the **first** veto reason should be `portfolio_dd_circuit` when it triggers.

5. Ensure the veto event gets written to `risk_vetoes.jsonl` with the new `veto_reason` and payload.

---

## 5. State Publisher: Expose Circuit Status

**File:** `execution/state_publish.py`

### Change

1. Wherever the risk state is assembled for `logs/state/risk.json`, extend it to include:

```python
"portfolio_dd_pct": dd_state.current_dd_pct if dd_state is not None else None,
"circuit_breaker": {
    "max_portfolio_dd_nav_pct": max_dd,
    "active": bool(
        max_dd is not None
        and dd_state is not None
        and dd_state.current_dd_pct >= max_dd
    ),
}
```

2. Do **not** remove any existing fields. This is an additive change to the risk state contract.

3. Handle missing values safely:

* If circuit breaker disabled → `circuit_breaker.active = False`, `max_portfolio_dd_nav_pct` may be `None`.
* If NAV history missing → `portfolio_dd_pct = None`, `active = False`.

---

## 6. Dashboard: Show Circuit Status

**File:** `dashboard/risk_panel_v7.py` (or equivalent)

### Change

1. Read the new risk state fields:

* `risk["portfolio_dd_pct"]`
* `risk["circuit_breaker"]["max_portfolio_dd_nav_pct"]`
* `risk["circuit_breaker"]["active"]`

2. Add a small, unobtrusive widget:

* Display **current DD** as a percentage.
* Display **max DD threshold**.
* Display an **“ACTIVE” badge** when `circuit_breaker.active == True`.

For example (pseudo-UI):

* When inactive:

  * Text: `DD: 3.2% (limit 10%) – circuit OK`
* When active:

  * Red badge: `CIRCUIT TRIPPED — DD 11.4% (limit 10%)`
  * Optional hint: “New orders are vetoed until NAV recovers or config changes.”

3. Do **not** alter layout heavily; just integrate into the existing risk KPIs.

---

## 7. Tests

### 7.1 Drawdown Tracker Tests

**File:** `tests/test_drawdown_tracker.py` (or new)

* Test `get_portfolio_dd_state`:

  * Case: `nav = [1000, 1100, 1200]` ⇒ `current_dd_pct == 0.0`.
  * Case: `nav = [1000, 1200, 1080]` ⇒ `current_dd_pct ≈ (1200-1080)/1200 = 0.10`.
  * Case: `nav = []` ⇒ `None`.

### 7.2 Risk Limits Circuit Tests

**File:** `tests/test_risk_limits_portfolio_dd.py` (new) or extend existing `test_risk_limits.py`.

Create at least three scenarios:

1. **No circuit configured**

   * `max_portfolio_dd_nav_pct = None` in config.
   * With arbitrary NAV history, ensure `check_order` **does not** veto with `portfolio_dd_circuit`.

2. **DD below threshold**

   * `max_portfolio_dd_nav_pct = 0.10`
   * NAV history yields `current_dd_pct = 0.05`.
   * Ensure `check_order` does not emit `portfolio_dd_circuit` and existing behaviour stays intact.

3. **DD above threshold (circuit tripped)**

   * `max_portfolio_dd_nav_pct = 0.10`.
   * NAV history yields `current_dd_pct = 0.15`.
   * Ensure:

     * `check_order` returns a veto.
     * `veto_reason == "portfolio_dd_circuit"`.
     * Observed/limits payload contains expected fields and values within tolerance.

### 7.3 State Publish Tests

**File:** `tests/test_state_publish_risk_dd.py` (new) or extend existing state tests.

* Ensure risk state JSON contains:

```json
"portfolio_dd_pct": ...,
"circuit_breaker": {
  "max_portfolio_dd_nav_pct": ...,
  "active": ...
}
```

under correct conditions.

All existing tests must remain green.

---

## 8. Acceptance Criteria

The patch is complete when:

1. `max_portfolio_dd_nav_pct` is configurable and normalized via `risk_loader`.
2. `get_portfolio_dd_state` correctly computes DD on sample NAV series.
3. `risk_limits.check_order`:

   * Vetoes orders with reason `portfolio_dd_circuit` when DD exceeds threshold.
   * Never vetoes due to DD when circuit is disabled or NAV data missing.
4. `logs/state/risk.json` includes:

   * `portfolio_dd_pct`
   * `circuit_breaker.max_portfolio_dd_nav_pct`
   * `circuit_breaker.active`
5. Dashboard risk panel shows:

   * Current DD
   * Threshold
   * Clear indicator when circuit is active.
6. All tests (`pytest` with the usual command, including `PYTHONPATH=.` as per copilot-instructions) pass.

Do not change any other behaviour or contracts beyond what is described here.

---