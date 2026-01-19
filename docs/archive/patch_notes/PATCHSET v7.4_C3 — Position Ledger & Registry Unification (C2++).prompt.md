# PATCHSET v7.4_C3 — Position Ledger & Registry Unification (C2++)

## Objective

Introduce a **canonical Position Ledger (“P-Ledger”)** that becomes the single source of truth for:

- Open positions
- Per-position TP/SL
- Exit metadata

Registry files become **views** over the ledger, not independent state.

Key goals:

- No more “registry empty but positions exist” states.
- No more divergence between `positions_state.json` and TP/SL registry.
- Clean, testable APIs for:
  - reading current positions
  - reading/writing TP/SL
  - syncing with exchange + executor

This patch must be **backward compatible** with existing state files and exit scanner behaviour.

---

## Files to Touch / Add

- `execution/position_ledger.py` (NEW)
- `execution/positions_state_writer.py` (or where positions_state is written)
- `execution/position_tp_sl_registry.py` (reuse/enhance C2 additions)
- `execution/exit_scanner.py`
- `execution/executor_live.py`
- `execution/state_publish.py`
- `dashboard/risk_panel.py` or `dashboard/positions_panel.py`
- Tests:
  - `tests/test_position_ledger.py` (NEW)
  - `tests/test_exit_scanner_ledger_integration.py` (NEW)
  - `tests/test_state_publish_positions_ledger.py` (NEW or extend)

---

## 1. Position Ledger Core

**File:** `execution/position_ledger.py` (NEW)

### 1.1 Data Model

Define:

```python
from dataclasses import dataclass
from typing import Dict, Optional, Literal
from decimal import Decimal

Side = Literal["LONG", "SHORT"]

@dataclass
class PositionTP_SL:
    tp: Optional[Decimal]
    sl: Optional[Decimal]

@dataclass
class PositionLedgerEntry:
    symbol: str
    side: Side
    entry_price: Decimal
    qty: Decimal
    tp_sl: PositionTP_SL
    created_ts: Optional[float] = None
    updated_ts: Optional[float] = None
````

This is the **canonical** per-position object.

### 1.2 Storage Contract

Underlying files:

* `logs/state/positions_state.json` — existing file, still written.
* `logs/state/position_tp_sl.json` — registry-like file (already present from C2).

The ledger module will:

* **Read** from both and **merge** into `Dict[str, PositionLedgerEntry]`.
* **Write** to the TP/SL file when TP/SL changes.
* Never mutate `positions_state.json` directly; that remains the executor’s “raw positions” snapshot.

### 1.3 Ledger API

Implement:

```python
def load_positions_state() -> Dict[str, dict]:
    """Low-level: read positions_state.json as dict (existing structure)."""

def load_tp_sl_registry() -> Dict[str, dict]:
    """Low-level: read position_tp_sl.json; empty dict if missing."""

def save_tp_sl_registry(registry: Dict[str, dict]) -> None:
    """Write position_tp_sl.json (atomic write)."""


def build_position_ledger() -> Dict[str, PositionLedgerEntry]:
    """
    Merge positions_state.json + position_tp_sl.json into PositionLedgerEntry objects.

    Rules:
    - Only symbols with non-zero qty and valid entry_price are included.
    - If TP/SL exists in registry, attach it.
    - If missing, tp_sl = PositionTP_SL(tp=None, sl=None); can be seeded later.
    """
```

Add helpers:

```python
def upsert_tp_sl(symbol: str, side: Side, entry_price: Decimal, tp: Optional[Decimal], sl: Optional[Decimal]) -> None:
    """
    Update or create TP/SL for a given symbol in the registry file.
    """

def delete_tp_sl(symbol: str) -> None:
    """
    Remove TP/SL when position is fully closed (qty=0/absent in positions_state).
    """

def sync_ledger_with_positions(seed_missing: bool = True, remove_stale: bool = True) -> Dict[str, PositionLedgerEntry]:
    """
    High-level operation:
    - Rebuild ledger from current positions_state.json.
    - Optionally seed TP/SL for missing entries (using existing ATR/percentage logic).
    - Remove TP/SL entries for symbols no longer in positions.
    - Return the in-memory ledger.
    """
```

`sync_ledger_with_positions()` should internally use C2’s ATR-based TP/SL computation.

---

## 2. Integrate Ledger into Executor Startup

**File:** `execution/executor_live.py`

### 2.1 Replace ad-hoc registry seeding

You already added `_sync_tp_sl_registry()` in C2. Replace/upgrade it to use the ledger API:

```python
from execution.position_ledger import sync_ledger_with_positions

def _startup_position_check(...):
    ...
    ledger = sync_ledger_with_positions(seed_missing=True, remove_stale=True)
    logger.info("[startup-ledger] position ledger synced: %s entries", len(ledger))
```

Rules:

* Always run this when `ALLOW_OPEN_POSITIONS=1` or equivalent.
* If no positions exist:

  * `ledger` should be empty.
  * Registry file should be cleaned up / truncated to `{}`.

---

## 3. Exit Scanner Uses Ledger, Not Raw Registry

**File:** `execution/exit_scanner.py`

### 3.1 Replace direct registry reads

Instead of directly reading `position_tp_sl.json`, use the ledger:

```python
from execution.position_ledger import build_position_ledger

def scan_for_exits(...):
    ledger = build_position_ledger()
    for symbol, entry in ledger.items():
        tp = entry.tp_sl.tp
        sl = entry.tp_sl.sl
        ...
```

Behaviour:

* If `tp` or `sl` is `None`, either:

  * Skip, or
  * Call a helper `ensure_tp_sl_for_entry()` that seeds them and persists via `upsert_tp_sl()` (up to you — easiest is to rely on startup sync to ensure they exist).
* The scanner’s exit logic (TP/SL checks) remains the same; it just gets inputs from `PositionLedgerEntry` instead of a raw registry dict.

This enforces **one canonical place** to read TP/SL + position meta.

---

## 4. Positions State Writer: Make It Ledger-Friendly

**File:** `execution/positions_state_writer.py` (or wherever `positions_state.json` is written, often `sync_state.py` / executor)

Ensure that positions are written in a way that ledger can always interpret:

* For each open position entry, ensure:

```json
"symbol": "SOLUSDT",
"side": "LONG",
"qty": 1.234,
"entry_price": 143.36,
"created_ts": 1733310000.123
```

If the current structure differs, ledger should adapt and normalize field names:

* e.g. accept both `side` and `position_side`.
* Accept both `entry_price` and `avg_entry_price`.

Add small normalizer inside `position_ledger.build_position_ledger()` to handle multiple shapes defensively.

---

## 5. State Publishing: Expose Ledger-Derived View

**File:** `execution/state_publish.py`

### 5.1 Expose positions_ledger_summary

Add to state snapshot:

```python
from execution.position_ledger import build_position_ledger

def write_positions_snapshot_state(...):
    ledger = build_position_ledger()
    ...
    state["positions_ledger"] = {
        symbol: {
            "side": entry.side,
            "qty": float(entry.qty),
            "entry_price": float(entry.entry_price),
            "tp": float(entry.tp_sl.tp) if entry.tp_sl.tp is not None else None,
            "sl": float(entry.tp_sl.sl) if entry.tp_sl.sl is not None else None,
            "created_ts": entry.created_ts,
            "updated_ts": entry.updated_ts,
        }
        for symbol, entry in ledger.items()
    }
```

Do **not** remove existing `positions_state` block; this is an additive, higher-level view.

---

## 6. Dashboard: Show Ledger Consistency

**File:** `dashboard/risk_panel.py` or `dashboard/positions_panel.py`

### 6.1 Positions Table from Ledger

* Instead of reading raw `positions_state.json` only, prefer `positions_ledger` where available.
* Display:

  * symbol
  * side
  * qty
  * entry_price
  * tp
  * sl

### 6.2 Consistency Warning (bonus)

Compute:

* `num_positions = len(positions_state.positions)`
* `num_ledger = len(positions_ledger)`
* `num_tp_sl = count of entries with non-null tp/sl`

Show in UI:

* 🟢 if `num_positions == num_ledger` and all have tp/sl (or within expected exception rules).
* 🟡 if some positions missing tp/sl.
* 🔴 if `num_positions > 0` and `num_ledger == 0` (should never happen now; this becomes a loud canary).

---

## 7. Tests

### 7.1 Position Ledger Tests

**File:** `tests/test_position_ledger.py` (NEW)

Cover:

1. **Basic merge**

   * positions_state with 2 symbols
   * tp_sl registry with 1 of them
   * `build_position_ledger()` yields 2 entries; one with tp/sl, one with None.

2. **sync_ledger_with_positions() seeding**

   * positions_state has valid entries
   * no tp_sl registry file
   * `sync_ledger_with_positions(seed_missing=True)` generates TP/SL with expected values (use C2 ATR logic or simplified stubs for tests).

3. **Stale removal**

   * registry contains a symbol not present in positions_state
   * sync removes that symbol from registry.

4. **Edge formats**

   * positions_state with alternative keys (e.g., `position_side`, `avg_entry_price`)
   * ledger normalizes them correctly.

### 7.2 Exit Scanner Integration

**File:** `tests/test_exit_scanner_ledger_integration.py` (NEW)

* Stub `build_position_ledger()` to return:

  * 1 LONG symbol with mark < SL → exit intent emitted.
  * 1 LONG symbol with mark > TP → exit intent emitted.
  * 1 symbol with no tp/sl → assert behaviour (skip or seed) per design.

Ensure scanner **never** directly reads `position_tp_sl.json` in tests.

### 7.3 State Publish Tests

**File:** `tests/test_state_publish_positions_ledger.py` (NEW or extend)

* Ensure `positions_ledger` exists in state snapshot.
* Validate fields for a sample symbol (side, qty, entry_price, tp, sl).

All existing tests (159) must remain green.

---

## 8. Acceptance Criteria

The patch is complete when:

1. `execution/position_ledger.py` exists and:

   * Can merge `positions_state.json` + `position_tp_sl.json` into `PositionLedgerEntry` objects.
   * `sync_ledger_with_positions()` can seed missing TP/SL and remove stale entries.

2. `executor_live.py` uses `sync_ledger_with_positions()` at startup to guarantee a consistent ledger+registry before trading logic runs.

3. `exit_scanner.py` reads from `PositionLedgerEntry` objects instead of a raw registry dict.

4. `state_publish.py` exposes a `positions_ledger` block in state.

5. Dashboard uses `positions_ledger` for its positions view and can show a basic consistency status.

6. New tests:

   * `test_position_ledger.py`
   * `test_exit_scanner_ledger_integration.py`
   * `test_state_publish_positions_ledger.py`
     all pass, and total test suite remains green.

7. It is no longer possible (without breaking tests) to have:

   * non-empty open positions
   * with an empty ledger / registry
     and no warning.

Do not change any other behaviour beyond what is described here.

```
