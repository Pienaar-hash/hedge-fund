# PATCHSET v7.5_S0_B — Risk/Screener/Exit Diagnostics + Veto Heatmap

## Objective

Stabilize and **make observable** the full pipeline:

> Screener → Risk → Router → Positions → Ledger/Registry → Exit Scanner

by adding:

1. **Veto Diagnostics & Heatmap**
   - Count veto reasons (max_concurrent, symbol_cap, DD, VaR, CVaR, etc.)
   - Expose them in state + dashboard.

2. **Screener / Risk / Router Liveness Metrics**
   - Per-loop counters for signals, orders, vetoes.
   - Last-success timestamps, per-symbol gating flags.

3. **Exit Pipeline Health Diagnostics**
   - Explicit checks across:
     - positions_state.json
     - positions_ledger.json
     - TP/SL registry
     - exit scanner activity
   - Tests that ensure exits fire when TP/SL are breached.

This patch MUST NOT change the semantics of veto logic or execution.  
It only **observes** and **reports** what the engine is already doing.

---

## Files to Touch / Add

- Config
  - `config/strategy_config.json` (add diagnostics block)
- Risk / Screener / Exit
  - `execution/risk_limits.py`
  - `execution/signal_screener.py`
  - `execution/exit_scanner.py`
  - `execution/position_ledger.py` (light extension if needed)
  - `execution/position_tp_sl_registry.py` (light extension if needed)
- Diagnostics & State
  - `execution/diagnostics_metrics.py` (NEW)
  - `execution/state_publish.py`
  - `execution/state_v7.py`
  - `execution/executor_live.py`
- Dashboard
  - `dashboard/risk_panel.py`
  - `dashboard/intel_panel.py`
  - `dashboard/app.py` (wire new panels/sections if needed)
- Docs
  - `docs/v7.5_Incident_Playbook_Positions_Stuck.md` (NEW)
- Tests
  - `tests/test_veto_metrics.py` (NEW)
  - `tests/test_exit_pipeline_contract.py` (NEW)
  - `tests/test_state_publish_diagnostics.py` (NEW/extend)
  - (optional) `tests/test_signal_screener_diagnostics.py` (NEW)

---

## 1. Config: Diagnostics Block

**File:** `config/strategy_config.json`

Add a new top-level block (or under an existing diagnostics section):

```json
"diagnostics": {
  "enabled": true,
  "veto_metrics": {
    "window_signals": 500,          // how many recent signals to aggregate
    "window_seconds": 3600          // fallback time window (1h) if needed
  },
  "exit_pipeline": {
    "tp_sl_canary_enabled": true,
    "tp_sl_underwater_threshold_pct": -0.02  // -2% underwater triggers canary check
  }
}
````

Notes:

* `diagnostics.enabled` can disable all diagnostics with minimal overhead.
* `veto_metrics.window_signals` defines how many recent signals we track counters for (e.g., via in-memory ring buffer or simple counters reset periodically).
* `exit_pipeline.tp_sl_underwater_threshold_pct` is a sanity threshold: if a position is worse than this and no TP/SL is registered, we raise a diagnostic canary flag.

---

## 2. Diagnostics Module

**File:** `execution/diagnostics_metrics.py` (NEW)

Create a central home for runtime diagnostic counters.

### 2.1 Dataclasses & Types

```python
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

@dataclass
class VetoCounters:
    by_reason: Dict[str, int] = field(default_factory=dict)
    total_signals: int = 0
    total_orders: int = 0
    total_vetoes: int = 0
    last_veto_ts: Optional[str] = None
    last_order_ts: Optional[str] = None
    last_signal_ts: Optional[str] = None

@dataclass
class ExitPipelineStatus:
    last_exit_scan_ts: Optional[str] = None
    last_exit_trigger_ts: Optional[str] = None
    open_positions_count: int = 0
    tp_sl_registered_count: int = 0
    tp_sl_missing_count: int = 0
    underwater_without_tp_sl_count: int = 0

@dataclass
class RuntimeDiagnosticsSnapshot:
    veto_counters: VetoCounters
    exit_pipeline_status: ExitPipelineStatus
```

### 2.2 Global Diagnostics State

You can use a simple module-level singleton:

```python
_veto_counters = VetoCounters()
_exit_status = ExitPipelineStatus()

def get_veto_counters() -> VetoCounters:
    return _veto_counters

def get_exit_status() -> ExitPipelineStatus:
    return _exit_status
```

### 2.3 Update helpers

```python
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"
```

Signal-level:

```python
def record_signal_emitted():
    vc = _veto_counters
    vc.total_signals += 1
    vc.last_signal_ts = now_iso()
```

Order-level:

```python
def record_order_placed():
    vc = _veto_counters
    vc.total_orders += 1
    vc.last_order_ts = now_iso()
```

Veto-level:

```python
def record_veto(reason: str):
    vc = _veto_counters
    vc.total_vetoes += 1
    vc.by_reason[reason] = vc.by_reason.get(reason, 0) + 1
    vc.last_veto_ts = now_iso()
```

Exit pipeline:

```python
def record_exit_scan_run():
    _exit_status.last_exit_scan_ts = now_iso()

def record_exit_trigger():
    _exit_status.last_exit_trigger_ts = now_iso()

def update_exit_pipeline_status(
    open_positions_count: int,
    tp_sl_registered_count: int,
    tp_sl_missing_count: int,
    underwater_without_tp_sl_count: int
):
    es = _exit_status
    es.open_positions_count = open_positions_count
    es.tp_sl_registered_count = tp_sl_registered_count
    es.tp_sl_missing_count = tp_sl_missing_count
    es.underwater_without_tp_sl_count = underwater_without_tp_sl_count
```

### 2.4 Snapshot builder

```python
def build_runtime_diagnostics_snapshot() -> RuntimeDiagnosticsSnapshot:
    return RuntimeDiagnosticsSnapshot(
        veto_counters=_veto_counters,
        exit_pipeline_status=_exit_status,
    )
```

---

## 3. Integrations

### 3.1 Risk Veto Counters

**File:** `execution/risk_limits.py`

In `check_order()` (or equivalent central veto function):

* After determining a veto:

```python
from execution.diagnostics_metrics import record_veto

# Example where a veto reason is set:
if some_circuit_condition:
    reason = "portfolio_dd_circuit"
    record_veto(reason)
    return VetoResult(vetoed=True, reason=reason, ...)
```

Repeat for all major reasons:

* `max_concurrent`
* `symbol_cap`
* `portfolio_dd_circuit`
* `var_limit`
* `cvar_limit`
* `min_notional`
* `per_trade_nav_pct`
* `tier_block`
* etc.

Do NOT change logic; just call `record_veto(reason)` at the decision points.

### 3.2 Screener Diagnostics

**File:** `execution/signal_screener.py`

Where signals/intents are generated:

```python
from execution.diagnostics_metrics import record_signal_emitted

def some_screener_loop(...):
    ...
    # When a signal/intents object is emitted:
    record_signal_emitted()
    ...
```

If screener passes an intent to executor (or queue):

* Optionally record that as “signal that reached risk”.

### 3.3 Order Placement Diagnostics

**File:** `execution/order_router.py` or `execution/executor_live.py` (where orders are accepted and sent)

Add:

```python
from execution.diagnostics_metrics import record_order_placed

# After an order is successfully sent (not vetoed):
record_order_placed()
```

Again, no logic change, just instrumentation.

---

## 4. Exit Pipeline Diagnostics

### 4.1 Exit Scanner Integration

**File:** `execution/exit_scanner.py`

At the top of the main scan function:

```python
from execution.diagnostics_metrics import (
    record_exit_scan_run,
    record_exit_trigger,
    update_exit_pipeline_status,
)

def scan_tp_sl_exits(...):
    record_exit_scan_run()
    ...
```

After scanning, compute diagnostic counts:

* Use:

  * `positions_ledger` (C3) for open positions
  * `position_tp_sl_registry` for registered TP/SL
  * mark/PNL from `positions_state.json` or live positions if needed

Pseudocode:

```python
open_positions = ledger.get_open_positions()
tp_sl_entries = registry.get_all_entries()

open_symbols = {p.symbol for p in open_positions}
tp_sl_symbols = {e.symbol for e in tp_sl_entries}

open_positions_count = len(open_positions)
tp_sl_registered_count = len(open_symbols & tp_sl_symbols)
tp_sl_missing_count = len(open_symbols - tp_sl_symbols)

underwater_without_tp_sl_count = 0
for p in open_positions:
    if p.symbol not in tp_sl_symbols and p.unrealized_pnl_pct <= underwater_threshold:
        underwater_without_tp_sl_count += 1
```

Call:

```python
update_exit_pipeline_status(
    open_positions_count=open_positions_count,
    tp_sl_registered_count=tp_sl_registered_count,
    tp_sl_missing_count=tp_sl_missing_count,
    underwater_without_tp_sl_count=underwater_without_tp_sl_count,
)
```

When an actual exit is triggered (TP/SL hit):

* Inside the logic that emits exit orders:

```python
record_exit_trigger()
```

---

## 5. State Publishing

### 5.1 Diagnostics State Block

**File:** `execution/state_publish.py`

Import:

```python
from execution.diagnostics_metrics import build_runtime_diagnostics_snapshot
```

Add a writer:

```python
def write_runtime_diagnostics_state(state: dict) -> None:
    snapshot = build_runtime_diagnostics_snapshot()
    vc = snapshot.veto_counters
    es = snapshot.exit_pipeline_status

    state["runtime_diagnostics"] = {
        "veto_counters": {
            "by_reason": vc.by_reason,
            "total_signals": vc.total_signals,
            "total_orders": vc.total_orders,
            "total_vetoes": vc.total_vetoes,
            "last_signal_ts": vc.last_signal_ts,
            "last_order_ts": vc.last_order_ts,
            "last_veto_ts": vc.last_veto_ts,
        },
        "exit_pipeline": {
            "last_exit_scan_ts": es.last_exit_scan_ts,
            "last_exit_trigger_ts": es.last_exit_trigger_ts,
            "open_positions_count": es.open_positions_count,
            "tp_sl_registered_count": es.tp_sl_registered_count,
            "tp_sl_missing_count": es.tp_sl_missing_count,
            "underwater_without_tp_sl_count": es.underwater_without_tp_sl_count,
        }
    }
```

Call this inside your existing intel/state publish routine:

```python
def compute_and_write_intel_state(...):
    ...
    write_runtime_diagnostics_state(state)
    _write_json_state(INTEL_STATE_PATH, state)
```

Or, if you prefer a dedicated `diagnostics_state.json`, point `_write_json_state` there.

### 5.2 Loaders

**File:** `execution/state_v7.py`

Add:

```python
def load_runtime_diagnostics_state() -> dict:
    """
    Loads runtime diagnostics (veto_counters, exit_pipeline) from state.
    Returns {} if not present.
    """
    path = "logs/state/intel.json"  # or diagnostics.json
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("runtime_diagnostics", {})
    except FileNotFoundError:
        return {}
```

If you chose a dedicated diagnostics state file, update the path accordingly.

---

## 6. Dashboard: Veto Heatmap & Exit Health

### 6.1 Risk Panel — Veto Heatmap

**File:** `dashboard/risk_panel.py`

Use `load_runtime_diagnostics_state()`:

```python
from execution.state_v7 import load_runtime_diagnostics_state

def render_veto_heatmap():
    diag = load_runtime_diagnostics_state()
    veto = diag.get("veto_counters", {})
    by_reason = veto.get("by_reason", {})

    # Convert to displayable structure (e.g., table or bar chart)
    # Example: a simple table or small heatmap.
```

Display:

* Each veto reason with count over last window.
* Total signals / vetoes / orders.
* Possibly a “veto rate” = total_vetoes / total_signals.

Example table columns:

* Reason
* Count
* Share of Total Vetoes (%)

### 6.2 Exit Health Widget

In `risk_panel.py` or `intel_panel.py` (whichever is more appropriate):

```python
def render_exit_pipeline_status():
    diag = load_runtime_diagnostics_state()
    exit_status = diag.get("exit_pipeline", {})

    # Show:
    # - open_positions_count
    # - tp_sl_registered_count
    # - tp_sl_missing_count
    # - underwater_without_tp_sl_count
    # - last_exit_scan_ts
    # - last_exit_trigger_ts
```

Add a simple status:

* If `underwater_without_tp_sl_count > 0` → show 🟠/🔴 warning.
* If `open_positions_count > 0` and `last_exit_scan_ts` is very old → show warning.

### 6.3 Wire into Dashboard App

**File:** `dashboard/app.py`

* Add new sub-panel / section in Risk or Advanced tab:

  * “Risk Diagnostics”
  * Contains veto heatmap + exit health.

---

## 7. Incident Playbook Doc

**File:** `docs/v7.5_Incident_Playbook_Positions_Stuck.md` (NEW)

Contents (short):

* **Symptoms:**

  * Positions open, but:

    * No new trades for N hours
    * TP/SL not triggering
    * PnL looks stale/flat

* **Checklist:**

  1. Check `risk_snapshot.json`:

     * `risk_mode` (OK/DEFENSIVE)
     * `dd_state` (normal/warn/panic)
  2. Check `positions_state.json`:

     * `entry_price` / `mark_price` > 0 for open positions
  3. Check `positions_ledger.json` & TP/SL:

     * `open_positions_count`
     * `tp_sl_registered_count` vs `tp_sl_missing_count`
  4. Check runtime diagnostics:

     * `veto_counters.by_reason` for dominant veto
     * `exit_pipeline.underwater_without_tp_sl_count`
  5. Evaluate recovery:

     * If everything is blocked by `max_concurrent` or `symbol_cap`, review caps.
     * If underwater_without_tp_sl > 0, run ledger/registry sync and confirm exit scanner.

---

## 8. Tests

### 8.1 Veto Metrics

**File:** `tests/test_veto_metrics.py` (NEW)

Tests:

1. `record_veto` increments counters and timestamps.
2. Multiple reasons accumulate correctly in `by_reason`.
3. `record_signal_emitted` and `record_order_placed` bump respective counters and timestamps.
4. `build_runtime_diagnostics_snapshot` returns expected structure.

### 8.2 Exit Pipeline Contract

**File:** `tests/test_exit_pipeline_contract.py` (NEW)

Use fixtures:

* Synthetic `open_positions` with:

  * Some with TP/SL registered.
  * Some without.
  * Some underwater beyond threshold.

* A fake ledger + registry + marks interface (mocked or simple dataclasses).

Test:

1. When `scan_tp_sl_exits` runs:

   * `record_exit_scan_run` called (can check via snapshot or monkeypatch).
   * `update_exit_pipeline_status` gets correct counts.
2. When marks cross SL/TP:

   * `record_exit_trigger` is called.
   * Exit events/orders are created.

You can stub routing/exchange and just assert that the exit function’s internal “exit intents” list is non-empty.

### 8.3 State Publish Diagnostics

**File:** `tests/test_state_publish_diagnostics.py` (NEW/extend)

* With some synthetic updates in `diagnostics_metrics`:

  * call `write_runtime_diagnostics_state(state_dict)`.
  * Assert that:

    ```python
    "runtime_diagnostics" in state_dict
    "veto_counters" in state_dict["runtime_diagnostics"]
    "exit_pipeline" in state_dict["runtime_diagnostics"]
    ```

* Check keys: `total_signals`, `by_reason`, `open_positions_count`, etc.

### 8.4 Optional Screener Diagnostics

**File:** `tests/test_signal_screener_diagnostics.py` (NEW)

* Mock screener to emit a known number of signals.
* Confirm `total_signals` increments by that number.

---

## 9. Acceptance Criteria

The patchset is complete when:

1. **Veto Counters:**

   * `execution/risk_limits.py` calls `record_veto(reason)` for all major veto paths.
   * `diagnostics_metrics.VetoCounters` tracks total_signals, total_orders, total_vetoes, and `by_reason`.

2. **Screener & Order Diagnostics:**

   * `signal_screener` calls `record_signal_emitted()` whenever it emits a signal/intents object.
   * The order submission layer calls `record_order_placed()` when an order is actually sent.

3. **Exit Pipeline Diagnostics:**

   * `exit_scanner` calls `record_exit_scan_run()` on each scan.
   * `update_exit_pipeline_status(...)` is invoked with correct counts of:

     * `open_positions_count`
     * `tp_sl_registered_count`
     * `tp_sl_missing_count`
     * `underwater_without_tp_sl_count`
   * `record_exit_trigger()` is called when exits are triggered.

4. **State & Dashboard:**

   * `write_runtime_diagnostics_state()` writes a `runtime_diagnostics` block into the chosen state file.
   * `state_v7.load_runtime_diagnostics_state()` returns this block correctly.
   * Risk/intel dashboard panels display:

     * Veto counts by reason
     * Exit health (TP/SL coverage, underwater-without-TP/SL)

5. **Docs:**

   * `docs/v7.5_Incident_Playbook_Positions_Stuck.md` exists and describes the operational debugging steps.

6. **Tests:**

   * `pytest` on:

     * `tests/test_veto_metrics.py`
     * `tests/test_exit_pipeline_contract.py`
     * `tests/test_state_publish_diagnostics.py`
     * (and optionally `tests/test_signal_screener_diagnostics.py`)
   * All new tests pass.
   * The broader test suite remains green (no regressions).

The runtime behavior of the engine (what trades it takes, what it vetoes) must remain unchanged; only observability and diagnostics are improved.

```
```
