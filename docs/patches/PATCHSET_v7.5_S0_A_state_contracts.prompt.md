# PATCHSET_v7.5_S0_A — State & Runtime Contract Stabilization

## Objective

Stabilize the **runtime state contract** so that:

1. `logs/state/positions_state.json` is written by **one owner** (executor), with **real prices and PnL**.
2. NAV / DD state is resilient to **bogus spikes** (e.g. 37k ghost NAV).
3. We have **contract tests** that fail loudly if:
   - positions_state is zeroed / inconsistent
   - NAV anomaly filter is not applied

This patchset must NOT add features. It only:

- Cleans up state writers,
- Makes NAV/DD robust,
- Adds tests + docs.

---

## 1. State Contract — Single Writer Policy

### 1.1 New doc: `docs/v7.5_State_Contract.md`

Create a short doc that defines, for each `logs/state/*.json`, the canonical writer.

**File:** `docs/v7.5_State_Contract.md`

Include a table like:

| State File                          | Canonical Writer        | Purpose                            |
|------------------------------------|-------------------------|------------------------------------|
| logs/state/nav_state.json          | sync_state.py           | NAV, AUM, FX snapshots             |
| logs/state/positions_state.json    | executor_live.py        | Live futures positions snapshot    |
| logs/state/positions_ledger.json   | executor_live.py        | Unified position + TP/SL ledger    |
| logs/state/risk_snapshot.json      | executor_live.py        | Risk KPIs, DD, VaR, CVaR, caps     |
| logs/state/router_state.json       | executor_live.py        | Router quality, slippage, TWAP     |
| logs/state/factor_diagnostics.json | executor_live.py        | Factor diag, weights, ortho status |

Key rules to state explicitly:

- **Single-writer rule:** At most one process is allowed to write any given `logs/state/*.json`.
- Helper functions like `_write_json_state(path, payload)` may be shared, but the **callsite ownership** is unique.

This doc is guidance for this patch and future ones.

---

## 2. positions_state — Single Writer + Correct Payload

### 2.1 Identify and remove secondary writers

**Files involved (likely):**

- `execution/sync_state.py` (or similar)
- `execution/executor_live.py`
- Possibly other helpers in `execution/state_publish.py`

**Actions:**

1. **Search for writes** to `positions_state.json`:

   - Any call to `_write_json_state("logs/state/positions_state.json", ...)`
   - Any hard-coded `positions_state.json` path.

2. Decide canonical owner:

   > **Canonical writer: `executor_live.py` via a dedicated helper.**

3. In `sync_state.py` (or any non-executor module):

   - Remove or refactor any direct writes to `positions_state.json`.
   - If sync_state needs its own snapshot, it must either:
     - consume `positions_state.json` as read-only, or
     - write a separate file (e.g. `logs/state/positions_state_sync.json`).

### 2.2 Introduce a dedicated writer helper

**File:** `execution/executor_live.py` (or `execution/state_publish.py` if that’s already central)

Add:

```python
POSITIONS_STATE_PATH = "logs/state/positions_state.json"

def _write_positions_state(positions_rows: list[dict]) -> None:
    """
    Canonical writer for logs/state/positions_state.json.

    positions_rows must already contain correct:
    - symbol, side
    - entry_price, mark_price
    - size, notional
    - realized_pnl, unrealized_pnl
    - leverage/margin if available
    """
    payload = {
        "updated_ts": utc_now_iso(),
        "positions": positions_rows,
    }
    _write_json_state(POSITIONS_STATE_PATH, payload)
````

Then:

* Replace any existing ad-hoc writes from the executor like:

  ```python
  _write_json_state(POSITIONS_STATE_PATH, some_partial_payload)
  ```

  with:

  ```python
  _write_positions_state(rows)
  ```

### 2.3 Source of truth: `_collect_rows()` only

From the Codex log you shared:

> `_collect_rows()` has the real prices/PnL; another process overwrote them with zeros.

So:

* Ensure `_collect_rows()` (or equivalent function) is the **only source** for `positions_rows` passed into `_write_positions_state`.
* Do NOT derive positions_state from some stale in-memory cache.

If necessary:

* Move `_collect_rows()` to a shared module (e.g. `positions_state_utils.py`) if it’s currently buried in a different context.
* But the *call* to `_write_positions_state()` must be in the executor main loop where we already have fresh exchange positions data.

### 2.4 Enforce non-zero semantics

Make sure `_collect_rows()` and `_write_positions_state()` enforce:

* If `qty != 0`:

  * `entry_price > 0`
  * `mark_price > 0`
* `unrealized_pnl` is consistent with `(mark - entry) * contract_size * direction` within a small tolerance.

Add a lightweight assertion (optional in production, mandatory in tests) in `_write_positions_state`:

```python
for row in positions_rows:
    if abs(row.get("qty", 0)) > 0:
        assert row.get("entry_price", 0) > 0
        assert row.get("mark_price", 0) > 0
```

In production you can log a warning instead of raising.

---

## 3. NAV / Peak NAV Anomaly Guard

The 37k ghost peak NAV must never be allowed to push us into DEFENSIVE again.

### 3.1 NAV anomaly filter

**File:** `execution/drawdown_tracker.py` or `execution/nav_tracker.py` (wherever peak NAV is computed)

Add a function:

```python
@dataclass
class NavAnomalyConfig:
    enabled: bool
    max_multiplier_intraday: float  # e.g. 3.0
    max_gap_abs_usd: float          # e.g. 20000.0

def is_nav_anomalous(
    previous_peak: float,
    new_nav: float,
    cfg: NavAnomalyConfig
) -> bool:
    """
    Returns True if new_nav is likely bogus relative to previous peak:
    - too large a multiple of previous_peak
    - or too large an absolute jump
    """
    if not cfg.enabled:
        return False
    if previous_peak <= 0:
        return False

    if new_nav > previous_peak * cfg.max_multiplier_intraday:
        return True
    if new_nav - previous_peak > cfg.max_gap_abs_usd:
        return True
    return False
```

### 3.2 Guard peak NAV updates

Wherever peak NAV is updated:

```python
if not is_nav_anomalous(prev_peak, nav, cfg):
    peak_nav = max(prev_peak, nav)
else:
    log.warning(
        "nav_anomaly_detected",
        extra={"prev_peak": prev_peak, "nav": nav}
    )
    # do not update peak_nav; optionally record anomaly event
```

Configuration for `NavAnomalyConfig` should be added to `risk_limits.json` or `strategy_config.json` under a small `nav_anomalies` block, e.g.:

```json
"nav_anomalies": {
  "enabled": true,
  "max_multiplier_intraday": 3.0,
  "max_gap_abs_usd": 20000.0
}
```

---

## 4. Contract Tests

### 4.1 Test: positions_state correctness

**File:** `tests/test_state_positions_contract.py` (NEW)

Tests:

1. **Single writer semantics**

   * Use monkeypatch to simulate `_collect_rows()` returning a list with 2 positions.
   * Call `_write_positions_state()` directly.
   * Assert:

     ```python
     data = json.load(open(POSITIONS_STATE_PATH))
     assert "positions" in data
     assert len(data["positions"]) == 2
     ```

2. **Non-zero fields for live positions**

   * Build a `positions_rows` fixture with:

     * `qty != 0`, `entry_price > 0`, `mark_price > 0`, `unrealized_pnl` computed.
   * Call `_write_positions_state()`.
   * Assert:

     ```python
     for row in data["positions"]:
         if abs(row["qty"]) > 0:
             assert row["entry_price"] > 0
             assert row["mark_price"] > 0
     ```

3. **Zero quantities allowed**

   * For a row with `qty == 0`, `entry_price` and `mark_price` may be 0.
   * Ensure writer does not interfere.

### 4.2 Test: NAV anomaly filter

**File:** `tests/test_nav_anomaly_guard.py` (NEW)

Synthetic cases:

1. **No anomaly, small moves**

   * prev_peak=10_000, new_nav=10_500
   * Expect `is_nav_anomalous()` → False

2. **Multiplier anomaly**

   * prev_peak=10_000, new_nav=40_000, max_multiplier_intraday=3.0
   * Expect True

3. **Gap anomaly**

   * prev_peak=10_000, new_nav=32_000, max_gap_abs_usd=20_000
   * Expect True

4. **Initial peak**

   * prev_peak=0, new_nav=37_000
   * Should not treat as anomaly (no baseline).

5. **Integration: peak nav update**

   * Simulate a simple peak nav update function which uses `is_nav_anomalous`.
   * Feed a stream of navs: [10_000, 10_500, 40_000].
   * Expect that peak_nav ends at 10_500, not 40_000.

### 4.3 Optional: State contract smoke test

**File:** `tests/test_state_contract_manifest.py` (NEW)

* Load `docs/v7.5_State_Contract.md` (if easily parseable) or keep a small Python dict in the test listing canonical writers.
* Assert that there is **no more than one writer** in code for `positions_state.json`:

  * e.g., by grepping for `positions_state.json` and listing callsites; at least confirm expected modules.

(Simpler: just assert that `sync_state.py` no longer contains `positions_state.json`.)

---

## 5. Acceptance Criteria

The patchset is complete when:

1. **Single-writer policy**:

   * Only `executor_live.py` (or a single module) writes `logs/state/positions_state.json`.
   * Any previous writes from `sync_state.py` or other modules are removed/refactored.

2. **Positions payload correctness**:

   * `positions_state.json` is built exclusively from `_collect_rows()` (or equivalent).
   * Non-zero positions have non-zero entry/mark prices.
   * A contract test (`test_state_positions_contract.py`) asserts correct structure and non-zero semantics.

3. **NAV anomaly guard**:

   * A `nav_anomalies` config block exists and is loaded.
   * `is_nav_anomalous()` is used to filter peak NAV updates.
   * A test (`test_nav_anomaly_guard.py`) validates typical anomaly and non-anomaly cases.

4. **State contract doc**:

   * `docs/v7.5_State_Contract.md` exists with a table documenting:

     * each state file
     * canonical writer
     * purpose.

5. **Tests**:

   * All new tests pass.
   * Existing test suite remains green (no new failures introduced).

This patchset must **not**:

* Change risk veto logic semantics (aside from making NAV/DD robust to spikes).
* Change execution/router behavior.
* Introduce new features — only stabilize & document state.

```
