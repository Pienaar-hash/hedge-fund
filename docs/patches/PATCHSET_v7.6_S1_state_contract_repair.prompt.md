# PATCHSET v7.6_S1 — State Contract Repair
## (positions_state + positions_ledger + kpis_v7)

## Objective

Repair and harden the **core state contract** so that:

1. `logs/state/positions_state.json` is:
   - Written atomically by a single canonical writer (executor),
   - Structurally correct and schema-compliant,
   - Derived from a reliable live positions surface.

2. `logs/state/positions_ledger.json` is:
   - Written every executor loop,
   - Reflects unified position + TP/SL ledger state,
   - Atomically updated and consumable by dashboard + diagnostics.

3. `logs/state/kpis_v7.json` has:
   - A single writer (executor),
   - A consistent schema, no partial/zeroed writes.

This patch **must not** change trading, risk semantics, or execution logic.  
It only makes the state contract correct, atomic, and trustworthy.

---

## Scope & Files

You will likely touch:

- **Positions state & ledger**
  - `execution/executor_live.py`
  - `execution/state_publish.py`
  - `execution/position_ledger.py`
  - `execution/state_v7.py` (loaders/specs)
  - Any helper in `execution/positions_state_utils.py` or similar (if exists)

- **KPIs**
  - `execution/state_publish.py`
  - `execution/sync_state.py` (to remove secondary writer)
  - `execution/state_v7.py` (loader/specs)

- **Config / Docs**
  - `docs/v7.6_State_Contract.md` (update after changes, or stub edits)
  - `v7_manifest.json` (ensure state surfaces reflect reality)

- **Tests**
  - `tests/integration/test_state_positions_contract.py` (or create it)
  - `tests/integration/test_state_positions_ledger_contract.py` (NEW)
  - `tests/integration/test_state_kpis_v7_contract.py` (NEW or extend)
  - `tests/integration/test_state_files_schema.py` (extend)

---

## 1. positions_state.json — Canonical Atomic Writer

### 1.1 Single Writer Rule

**Goal:** only the **executor** writes `logs/state/positions_state.json`.  
No other module (sync_state, dashboard, scripts) may write to this file.

**Actions:**

1. Grep for `positions_state.json` across the repo.  
   - Identify all writers (calls to `_write_json_state`, `_write_json_cache`, custom writers).
2. Remove or refactor any non-executor writer:
   - If `sync_state.py` or any other module currently writes `positions_state.json`, change it to:
     - read-only usage, or
     - write to a different file, e.g. `logs/state/positions_state_sync.json`.

After the patch, **only executor_live/state_publish** should call the writer for `positions_state.json`.

### 1.2 Canonical Writer Helper

We want one helper that always writes this file in the correct shape, atomically.

**File:** `execution/state_publish.py` (preferred central place)

Add:

```python
POSITIONS_STATE_PATH = "logs/state/positions_state.json"

def write_positions_state(positions_rows: list[dict]) -> None:
    """
    Canonical atomic writer for logs/state/positions_state.json.

    positions_rows must contain:
    - symbol (str)
    - side (str)
    - qty (float)
    - entry_price (float)
    - mark_price (float)
    - notional (float)
    - realized_pnl (float)
    - unrealized_pnl (float)
    - leverage/margin fields if available

    This function is the single writer for positions_state.json.
    """
    payload = {
        "updated_at": utc_now_iso(),  # use updated_at, not updated_ts
        "positions": positions_rows,
    }
    _atomic_write_state(POSITIONS_STATE_PATH, payload)
````

Where `_atomic_write_state` is either:

* an existing atomic writer (rename `_write_json_state` / `_write_json_cache` if needed), or
* a new helper that writes to a temp file and `os.replace`s to the final path.

Example atomic pattern:

```python
import json, os, tempfile

def _atomic_write_state(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp_path, path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
```

### 1.3 Executor Integration

**File:** `execution/executor_live.py`

Wherever executor currently publishes positions state, replace ad-hoc calls like:

```python
_write_json_state("logs/state/positions_state.json", positions_payload)
```

with:

```python
from execution.state_publish import write_positions_state

rows = _collect_positions_rows(...)  # see 1.4
write_positions_state(rows)
_last_positions_state = rows  # if you keep in-memory copy for debugging
```

Ensure this is done **once per main loop**, after fresh positions have been fetched and normalized.

### 1.4 Source of Truth: _collect_positions_rows()

We need a single function that constructs `positions_rows` from the exchange positions / internal state.

If it already exists (e.g. `_collect_rows`, `_build_positions_state_rows`), use it; if not, create:

```python
def _collect_positions_rows(exchange_positions: list[dict]) -> list[dict]:
    rows = []
    for p in exchange_positions:
        qty = float(p["qty"])
        entry = float(p["entry_price"])
        mark = float(p["mark_price"])
        notional = abs(qty * mark)  # adjust for contract size if needed

        # compute PnL if not already provided
        # direction = +1 for LONG, -1 for SHORT
        direction = 1 if p["side"] == "LONG" else -1
        unreal = (mark - entry) * qty * direction  # adapt to your contract math

        rows.append({
            "symbol": p["symbol"],
            "side": p["side"],
            "qty": qty,
            "entry_price": entry,
            "mark_price": mark,
            "notional": notional,
            "realized_pnl": float(p.get("realized_pnl", 0.0)),
            "unrealized_pnl": unreal if "unrealized_pnl" not in p else float(p["unrealized_pnl"]),
            # include margin/leverage fields if available
        })
    return rows
```

The exact math can use your real helpers; the key is:

* **no non-zero qty** with zero entry/mark,
* consistent fields per row.

### 1.5 Schema Expectations

`positions_state.json` must always have:

```json
{
  "updated_at": "2025-01-01T00:00:00Z",
  "positions": [
    {
      "symbol": "BTCUSDT",
      "side": "LONG",
      "qty": 0.01,
      "entry_price": 42000.0,
      "mark_price": 43000.0,
      "notional": 430.0,
      "realized_pnl": 0.0,
      "unrealized_pnl": 10.0
      // margin/leverage optional but encouraged
    }
  ]
}
```

Rows with `qty == 0` may be omitted or allowed with zero prices; but they **must not** be the only rows for live positions.

---

## 2. positions_ledger.json — Unified Ledger State

### 2.1 Ensure Writer Exists & Runs Each Loop

**Goal:** `logs/state/positions_ledger.json` must be written by executor **every loop**.

**File:** `execution/state_publish.py`

Add:

```python
POSITIONS_LEDGER_PATH = "logs/state/positions_ledger.json"

def write_positions_ledger_state(ledger_dict: dict) -> None:
    """
    Canonical atomic writer for logs/state/positions_ledger.json.

    ledger_dict is typically:
    {
      "updated_at": "...",
      "entries": [...],  # per symbol+position_id
      "tp_sl_levels": {...},
      "metadata": {...}
    }
    """
    _atomic_write_state(POSITIONS_LEDGER_PATH, ledger_dict)
```

**File:** `execution/executor_live.py`

Where the main loop already has access to the `PositionLedger` object (from C3), add:

```python
from execution.position_ledger import build_positions_ledger_state
from execution.state_publish import write_positions_ledger_state

def _publish_state(...):
    ...
    ledger_state = build_positions_ledger_state(ledger)
    write_positions_ledger_state(ledger_state)
    ...
```

### 2.2 Ledger State Builder

**File:** `execution/position_ledger.py`

Add a helper that transforms the in-memory ledger into a serializable dict:

```python
def build_positions_ledger_state(ledger: PositionLedger) -> dict:
    """
    Build a serializable snapshot of the ledger for state publishing.
    """
    entries = []
    for key, entry in ledger.entries.items():
        entries.append({
            "symbol": entry.symbol,
            "side": entry.side,
            "position_id": entry.position_id,
            "qty": entry.qty,
            "entry_price": entry.entry_price,
            "tp": entry.tp,
            "sl": entry.sl,
            "ttl_ts": entry.ttl_ts,
            # include any extra fields needed by dashboard
        })

    tp_sl_levels = {
        symbol: {
            "tp": levels.tp,
            "sl": levels.sl,
        }
        for symbol, levels in ledger.tp_sl_levels.items()
    }

    return {
        "updated_at": utc_now_iso(),
        "entries": entries,
        "tp_sl_levels": tp_sl_levels,
        "metadata": {
            "version": "v1",
            "entry_count": len(entries),
        },
    }
```

Use your actual ledger structure as needed; the key is:

* `updated_at` at top level,
* `entries` list,
* `tp_sl_levels` map.

### 2.3 Loader & Schema

**File:** `execution/state_v7.py`

Add or update loader:

```python
def load_positions_ledger_state() -> dict:
    path = "logs/state/positions_ledger.json"
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
```

Optionally add a helper that returns a typed view:

```python
def get_ledger_entries() -> list[dict]:
    state = load_positions_ledger_state()
    return state.get("entries", [])
```

Update state contract doc and manifest accordingly.

---

## 3. kpis_v7.json — Single Writer & Schema

### 3.1 Single Writer Rule

**Goal:** only the **executor** writes `logs/state/kpis_v7.json`.

**File:** `execution/sync_state.py`

* Search for any write to `kpis_v7.json`.
* Remove / disable those paths.
* If sync_state needs KPIs, it should read them from `kpis_v7.json` (read-only).

### 3.2 Canonical Writer

**File:** `execution/state_publish.py`

Ensure there is one helper:

```python
KPIS_V7_PATH = "logs/state/kpis_v7.json"

def write_kpis_v7_state(kpis: dict) -> None:
    """
    Canonical atomic writer for logs/state/kpis_v7.json.

    kpis dict should contain:
    - updated_at
    - per-symbol & portfolio KPIs as defined by the risk/telemetry contract.
    """
    if "updated_at" not in kpis:
        kpis["updated_at"] = utc_now_iso()
    _atomic_write_state(KPIS_V7_PATH, kpis)
```

**File:** `execution/executor_live.py`

Call this once per loop, after all KPI calculations:

```python
from execution.state_publish import write_kpis_v7_state

def _publish_state(...):
    ...
    kpis = _build_kpis_snapshot(...)  # whatever function already exists
    write_kpis_v7_state(kpis)
    ...
```

### 3.3 Schema Expectations

Minimal schema for `kpis_v7.json`:

```json
{
  "updated_at": "2025-01-01T00:00:00Z",
  "portfolio": {
    "nav": 4093.43,
    "dd_pct": 0.0326,
    "var_nav_pct": 0.0056,
    "cvar_nav_pct": 0.0070
  },
  "per_symbol": {
    "BTCUSDT": {
      "pnl": 12.34,
      "pnl_pct": 0.0123,
      "exposure_nav_pct": 0.10
    }
  }
}
```

You don’t need to fully enforce this in code, but tests should verify key top-level keys exist.

---

## 4. Tests

### 4.1 positions_state Contract Test

**File:** `tests/integration/test_state_positions_contract.py`

Add/extend tests to verify:

1. `positions_state.json` exists after a simulated executor loop.
2. Top-level keys:

   * `updated_at`
   * `positions`
3. For each row in `positions`:

   * `symbol`, `side`, `qty` present.
   * If `qty != 0`:

     * `entry_price > 0`
     * `mark_price > 0`
     * `notional >= 0`
     * `unrealized_pnl` present.

Pseudo-test:

```python
def test_positions_state_minimal_schema(tmp_path, monkeypatch):
    # arrange: simulate executor writing to tmp_path / logs/state
    # or patch POSITIONS_STATE_PATH to tmp_path
    ...
    data = json.load(open(path))
    assert "updated_at" in data
    assert "positions" in data
    for row in data["positions"]:
        assert "symbol" in row
        assert "side" in row
        assert "qty" in row
        if abs(row["qty"]) > 0:
            assert row["entry_price"] > 0
            assert row["mark_price"] > 0
            assert "unrealized_pnl" in row
```

Use tmp_path and monkeypatch to avoid touching real logs directory.

### 4.2 positions_ledger Contract Test

**File:** `tests/integration/test_state_positions_ledger_contract.py` (NEW)

Tests:

1. If ledger has entries, `positions_ledger.json` exists and has:

   * `updated_at`
   * `entries` list
   * `tp_sl_levels` dict

2. Each entry:

   * `symbol`, `side`, `position_id`, `qty`, `entry_price`.

3. If there are TP/SL levels, they appear in `tp_sl_levels`.

### 4.3 kpis_v7 Contract Test

**File:** `tests/integration/test_state_kpis_v7_contract.py` (NEW)

Tests:

1. `kpis_v7.json` exists after a loop.
2. Contains:

   * `updated_at`
   * `portfolio`
   * `per_symbol`
3. `portfolio` has `nav`, `dd_pct` at minimum.

### 4.4 State Files Schema Test (Extend)

**File:** `tests/integration/test_state_files_schema.py`

Extend the `STATE_FILES` specification (or equivalent) to include:

* `positions_state` → `required_keys = ["updated_at", "positions"]`
* `positions_ledger` → `required_keys = ["updated_at", "entries"]`
* `kpis_v7` → `required_keys = ["updated_at", "portfolio"]`

Verify all schemas when files exist.

---

## 5. Manifest & Doc Updates

### 5.1 v7_manifest.json

Add or update:

```json
"state_files": {
  "positions_state": "logs/state/positions_state.json",
  "positions_ledger": "logs/state/positions_ledger.json",
  "kpis_v7": "logs/state/kpis_v7.json",
  "...": "..."
}
```

Make sure these paths match actual files.

### 5.2 docs/v7.6_State_Contract.md

Update or add a short section:

* Document:

  * `positions_state.json` schema
  * `positions_ledger.json` schema
  * `kpis_v7.json` schema
* Explicitly state:

  * **Canonical writer** (executor/state_publish) for each
  * Atomic write expectation

---

## 6. Acceptance Criteria

This patchset is complete when:

1. **Single Writer Enforcement**

   * Only executor/state_publish writes:

     * `positions_state.json`
     * `positions_ledger.json`
     * `kpis_v7.json`
   * Any previous writes from `sync_state.py` or other modules are removed or redirected.

2. **Atomic Writes**

   * All three files use an atomic write helper (`_atomic_write_state` or equivalent).

3. **Schema & Content**

   * `positions_state.json` always has `updated_at` and `positions` list with non-zero qty rows having non-zero entry/mark.
   * `positions_ledger.json` always has `updated_at`, `entries`, `tp_sl_levels` (possibly empty, but present).
   * `kpis_v7.json` always has `updated_at`, `portfolio`, and `per_symbol`.

4. **Tests**

   * `tests/integration/test_state_positions_contract.py` passes.
   * `tests/integration/test_state_positions_ledger_contract.py` passes.
   * `tests/integration/test_state_kpis_v7_contract.py` passes.
   * `tests/integration/test_state_files_schema.py` passes.
   * Full unit+integration test suite remains green with:

     ```bash
     PYTHONPATH=. pytest tests/unit tests/integration -q
     ```

5. **Docs & Manifest**

   * `v7_manifest.json` lists these state files correctly.
   * `docs/v7.6_State_Contract.md` describes them and their canonical writers.

This completes **S1 — State Contract Repair** and provides a solid base for S2 (Diagnostics Hardening) and subsequent v7.6 tracks.

```
```
