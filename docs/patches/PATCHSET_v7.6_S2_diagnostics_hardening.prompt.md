# PATCHSET v7.6_S2 — Diagnostics Hardening
## (Liveness Semantics, Router Activity, Exit Coverage & Alerts)

## Objective

Harden the diagnostics and liveness layer added in v7.5/v7.6 so that:

1. **Liveness alerts are semantically correct and conservative**  
   - Missing timestamps are treated as *idle / unknown*, not “healthy”.
   - Idle durations are properly computed and surfaced.

2. **Router activity is explicitly tracked**  
   - We have a timestamp for the last router event (not just orders).
   - Liveness includes a real `idle_router` signal.

3. **Exit coverage & mismatch are visible and testable**  
   - Ledger vs TP/SL registry mismatch is quantified.
   - Underwater positions without TP/SL are clearly flagged.

4. **Diagnostics state is structurally stable**  
   - `runtime_diagnostics` includes veto, exit, liveness, and coverage metrics.
   - Dashboard reads these fields reliably.

No trading or risk logic semantics must change.  
All changes are **diagnostics-only**: observe and report.

---

## Files in Scope

You will likely touch:

- **Diagnostics & Liveness**
  - `execution/diagnostics_metrics.py`
  - `execution/state_publish.py`
  - `execution/executor_live.py`
  - `execution/order_router.py` or `execution/router_metrics.py`
  - `execution/exit_scanner.py`

- **State & Loader**
  - `execution/state_v7.py`

- **Dashboard**
  - `dashboard/risk_panel.py`
  - `dashboard/app.py`

- **Config & Docs**
  - `config/strategy_config.json`
  - `docs/v7.6_Runtime_Diagnostics.md`

- **Tests**
  - `tests/integration/test_veto_metrics.py`
  - `tests/integration/test_exit_pipeline_contract.py`
  - `tests/integration/test_state_publish_diagnostics.py`
  - (possibly) `tests/integration/test_state_files_schema.py`

---

## 1. Liveness Semantics: Missing Timestamps & Durations

We already track:

- `last_signal_ts`
- `last_order_ts`
- `last_exit_scan_ts`
- `last_exit_trigger_ts`

But current semantics may treat “missing ts” as “OK”, or not surface durations.

### 1.1 Config Confirm

**File:** `config/strategy_config.json`

Ensure diagnostics.liveness block exists and is used:

```json
"diagnostics": {
  "enabled": true,
  "veto_metrics": { ... },
  "exit_pipeline": { ... },
  "liveness": {
    "enabled": true,
    "max_idle_signals_seconds": 600,
    "max_idle_orders_seconds": 1200,
    "max_idle_exits_seconds": 3600,
    "max_idle_router_events_seconds": 1800
  }
}
````

* If block already exists with different numbers, **do not** change defaults unless necessary.
* Our logic will respect whatever thresholds are present.

### 1.2 Liveness Computation Semantics

**File:** `execution/diagnostics_metrics.py`

You already have a `compute_liveness_alerts(...)` or similar. Harden it with:

#### Rules:

1. If `cfg.enabled` is false → all liveness flags false, no-op.

2. If a timestamp is **missing**:

   * Do **not** assume healthy.
   * Treat duration as `None` but set the corresponding idle flag to **True**
     once the system has been running long enough (or immediately, depending on design).
   * For this patch, we keep it simple: if ts is missing and threshold > 0 → set idle flag True.

3. All durations for existing timestamps must be:

   * Non-negative floats in seconds.
   * Stored in `details` dict for debugging.

#### Implementation sketch:

```python
@dataclass
class LivenessAlerts:
    idle_signals: bool = False
    idle_orders: bool = False
    idle_exits: bool = False
    idle_router: bool = False
    details: Dict[str, float] = field(default_factory=dict)
    missing: Dict[str, bool] = field(default_factory=dict)  # NEW: track missing ts
```

Add or refine helper:

```python
from datetime import datetime, timezone

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def compute_liveness_alerts(cfg_liveness) -> LivenessAlerts:
    alerts = LivenessAlerts()
    if not cfg_liveness.enabled:
        return alerts

    now = datetime.now(timezone.utc)

    def check(ts_str, threshold, key, attr_name):
        if threshold is None or threshold <= 0:
            return

        ts = _parse_iso(ts_str)
        if ts is None:
            # No data → treat as missing and likely idle
            alerts.missing[key] = True
            # We flag idle; duration is not well-defined; optionally set to threshold+1
            alerts.details[key] = float(threshold) + 1.0
            setattr(alerts, attr_name, True)
            return

        delta = (now - ts).total_seconds()
        if delta < 0:
            # clock skew, ignore
            return
        alerts.details[key] = delta
        if delta > threshold:
            setattr(alerts, attr_name, True)

    vc = _veto_counters
    es = _exit_status

    check(vc.last_signal_ts, cfg_liveness.max_idle_signals_seconds, "signals", "idle_signals")
    check(vc.last_order_ts, cfg_liveness.max_idle_orders_seconds, "orders", "idle_orders")
    check(es.last_exit_trigger_ts, cfg_liveness.max_idle_exits_seconds, "exits", "idle_exits")
    check(es.last_router_event_ts, cfg_liveness.max_idle_router_events_seconds, "router", "idle_router")

    return alerts
```

> Note: `ExitPipelineStatus` will gain `last_router_event_ts` in the next step.

---

## 2. Router Activity Tracking

We currently use orders as a proxy for router activity. We want a dedicated field.

### 2.1 Extend ExitPipelineStatus or Dedicated RouterStatus

Simplest: extend `ExitPipelineStatus` to track router events as well.

**File:** `execution/diagnostics_metrics.py`

Extend dataclass:

```python
@dataclass
class ExitPipelineStatus:
    last_exit_scan_ts: Optional[str] = None
    last_exit_trigger_ts: Optional[str] = None
    open_positions_count: int = 0
    tp_sl_registered_count: int = 0
    tp_sl_missing_count: int = 0
    underwater_without_tp_sl_count: int = 0
    last_router_event_ts: Optional[str] = None  # NEW
```

Add helper:

```python
def record_router_event():
    _exit_status.last_router_event_ts = now_iso()
```

### 2.2 Instrument Router

**File:** `execution/order_router.py` (or `execution/router_metrics.py` if you centralize metrics)

After any router event that indicates actual routing activity (e.g., successful route attempt, order placement, TWAP slice):

```python
from execution.diagnostics_metrics import record_router_event

def _route_order(...):
    ...
    # When the router actually does something meaningful (even if order is rejected upstream):
    record_router_event()
    ...
```

You can also instrument TWAP slice routing and fill logging.

---

## 3. Exit Coverage & Ledger/Registry Mismatch

We already update exit pipeline status with:

* open positions count
* TP/SL registered/missing
* underwater_without_tp_sl_count

We now want explicit coverage and mismatch flags.

### 3.1 Extend ExitPipelineStatus

**File:** `execution/diagnostics_metrics.py`

Add fields:

```python
@dataclass
class ExitPipelineStatus:
    ...
    tp_sl_coverage_pct: float = 0.0
    ledger_registry_mismatch: bool = False
```

### 3.2 Compute Coverage & Mismatch

**File:** `execution/exit_scanner.py`

Inside the main scanning function, where we already computed:

* `open_positions`
* `tp_sl_entries`
* `underwater_without_tp_sl_count`

Extend logic:

```python
open_symbols = {p.symbol for p in open_positions}
tp_sl_symbols = {e.symbol for e in tp_sl_entries}

open_positions_count = len(open_positions)
tp_sl_registered_count = len(open_symbols & tp_sl_symbols)
tp_sl_missing_count = len(open_symbols - tp_sl_symbols)

coverage_pct = 0.0
if open_positions_count > 0:
    coverage_pct = tp_sl_registered_count / open_positions_count

ledger_registry_mismatch = tp_sl_missing_count > 0
```

Then call:

```python
update_exit_pipeline_status(
    open_positions_count=open_positions_count,
    tp_sl_registered_count=tp_sl_registered_count,
    tp_sl_missing_count=tp_sl_missing_count,
    underwater_without_tp_sl_count=underwater_without_tp_sl_count,
    tp_sl_coverage_pct=coverage_pct,
    ledger_registry_mismatch=ledger_registry_mismatch,
)
```

Update function signature in `diagnostics_metrics.update_exit_pipeline_status` to accept the new parameters.

Note: keep default values in dataclass so that older callers (if any) still work, or update all callsites.

---

## 4. State Publishing & Dashboard Updates

### 4.1 Diagnostics State

**File:** `execution/state_publish.py`

You already have a `write_runtime_diagnostics_state(state, cfg_liveness)` or similar.

Extend the `runtime_diagnostics` block to include:

* `liveness.missing` map
* `exit_pipeline.tp_sl_coverage_pct`
* `exit_pipeline.ledger_registry_mismatch`
* `exit_pipeline.last_router_event_ts`

Example:

```python
def write_runtime_diagnostics_state(state: dict, cfg_liveness) -> None:
    snapshot = build_runtime_diagnostics_snapshot(cfg_liveness)
    vc = snapshot.veto_counters
    es = snapshot.exit_pipeline_status
    la = snapshot.liveness_alerts

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
            "last_router_event_ts": es.last_router_event_ts,
            "open_positions_count": es.open_positions_count,
            "tp_sl_registered_count": es.tp_sl_registered_count,
            "tp_sl_missing_count": es.tp_sl_missing_count,
            "underwater_without_tp_sl_count": es.underwater_without_tp_sl_count,
            "tp_sl_coverage_pct": es.tp_sl_coverage_pct,
            "ledger_registry_mismatch": es.ledger_registry_mismatch,
        },
        "liveness": {
            "idle_signals": la.idle_signals if la else False,
            "idle_orders": la.idle_orders if la else False,
            "idle_exits": la.idle_exits if la else False,
            "idle_router": la.idle_router if la else False,
            "details": la.details if la else {},
            "missing": la.missing if la else {},
        },
    }
```

### 4.2 State Loader

**File:** `execution/state_v7.py`

Ensure `load_runtime_diagnostics_state()` is robust and returns the new fields when present, falling back gracefully when missing.

### 4.3 Dashboard Risk Panel

**File:** `dashboard/risk_panel.py`

Refine the diagnostics rendering:

1. **Veto Heatmap** — unchanged, still uses `veto_counters`.
2. **Exit Health Widget** — now includes:

   * `tp_sl_coverage_pct`
   * `ledger_registry_mismatch`
   * `underwater_without_tp_sl_count`
3. **Liveness Card** — now includes:

   * idle flags
   * durations from `details`
   * missing ts flags from `missing` (so we can see if blocks never ran)

Example:

* Show liveness as:

  * Signals: ✅ / 🔴 (idle) — “idle for 12m (missing: no)”
  * Orders: ✅ / 🔴 — “idle for 45m (missing: yes, router hasn’t run yet)”
  * Exits: ✅ / 🔴 — “last exit trigger 2h ago”

Be minimal, just enough for ops to diagnose.

---

## 5. Tests

### 5.1 Liveness Semantics

**File:** `tests/integration/test_veto_metrics.py` or a new `tests/integration/test_liveness_diagnostics.py`

Add tests for:

1. Missing timestamps:

   * With `cfg_liveness.enabled = True` and thresholds > 0
   * If `last_signal_ts = None` → `idle_signals == True`, `details["signals"] >= threshold`.
2. Existing timestamps within threshold:

   * `idle_* == False`.
3. Existing timestamps beyond threshold:

   * `idle_* == True` and duration correctly > threshold.
4. `missing` map:

   * For missing ts, `missing["signals"] == True`.

Use small thresholds and monkeypatch `_veto_counters` / `_exit_status` plus `datetime.now`.

### 5.2 Router Activity

**File:** `tests/integration/test_exit_pipeline_contract.py`

Add test:

* Simulate router activity by calling `record_router_event()`.
* After `write_runtime_diagnostics_state`, `exit_pipeline.last_router_event_ts` is not None.
* `idle_router` liveness flag behaves correctly relative to thresholds.

### 5.3 Exit Coverage & Mismatch

**File:** `tests/integration/test_exit_pipeline_contract.py`

Add tests:

1. When all open positions have TP/SL:

   * `tp_sl_coverage_pct == 1.0`, `ledger_registry_mismatch == False`.
2. When some are missing:

   * `tp_sl_coverage_pct` between 0 and 1.
   * `ledger_registry_mismatch == True`.
   * `tp_sl_missing_count` correct.

### 5.4 State Publishing Contract

**File:** `tests/integration/test_state_publish_diagnostics.py`

Extend to verify that:

* `runtime_diagnostics.exit_pipeline` has:

  * `tp_sl_coverage_pct`
  * `ledger_registry_mismatch`
  * `last_router_event_ts` (when simulated)
* `runtime_diagnostics.liveness` has:

  * `missing` dict
  * durations in `details` float-like.

### 5.5 Schema Test (Optional)

**File:** `tests/integration/test_state_files_schema.py`

Optionally add:

* `runtime_diagnostics` schema minimal keys:

  * `veto_counters`, `exit_pipeline`, `liveness`.

Be careful to only assert keys when file exists and diagnostics enabled.

---

## 6. Docs Update

**File:** `docs/v7.6_Runtime_Diagnostics.md`

Update to describe:

* new liveness semantics (missing → idle, not OK),
* router event tracking,
* exit coverage metrics (`tp_sl_coverage_pct`, `ledger_registry_mismatch`),
* how to interpret the dashboard liveness & exit widgets.

---

## 7. Acceptance Criteria

This patchset is complete when:

1. **Liveness Semantics**

   * Missing timestamps are represented in `liveness.missing` and produce `idle_* == True` (when thresholds > 0).
   * Durations in `liveness.details` are computed for present timestamps.
   * No negative durations or crashes on bad timestamps.

2. **Router Activity**

   * `record_router_event()` is called on meaningful router actions.
   * `exit_pipeline.last_router_event_ts` is set when router is active.
   * `liveness.idle_router` reflects router inactivity.

3. **Exit Coverage & Mismatch**

   * `exit_pipeline.tp_sl_coverage_pct` and `exit_pipeline.ledger_registry_mismatch` are updated each exit scan.
   * Underwater-without-TP/SL scenarios are visible through metrics.

4. **State & Dashboard**

   * `runtime_diagnostics` block includes veto, exit, liveness, and coverage fields.
   * Dashboard’s diagnostics panel shows:

     * veto counts,
     * exit health,
     * liveness (including missing-ts info).

5. **Tests**

   * All new/updated tests pass:

     * `tests/integration/test_veto_metrics.py` (or new liveness tests)
     * `tests/integration/test_exit_pipeline_contract.py`
     * `tests/integration/test_state_publish_diagnostics.py`
     * (optional) `tests/integration/test_state_files_schema.py`
   * Full fast suite remains green:

     ```bash
     PYTHONPATH=. pytest tests/unit tests/integration -q
     ```

6. **No Behavior Changes**

   * No change to core risk veto semantics, trade selection, or routing decisions.
   * Only metrics and diagnostics behavior are altered.

This completes **PATCHSET v7.6_S2 — Diagnostics Hardening**, preparing the engine for higher-level execution and factor work in v7.6.

```
```
