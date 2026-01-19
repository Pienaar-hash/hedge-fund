# PATCHSET v7.6_S3 — Test Suite Renovation
## (markers, speed lanes, unit seeds, fixtures, runtime isolation)

## Objective

Turn the v7.6 test suite into a **clear, fast, predictable** tool by:

1. Clarifying **test lanes** (unit / integration / runtime / legacy),
2. Making the **fast lane** truly quick and stable,
3. Seeding core **unit tests** for critical utilities (diagnostics, state publishing, vol/factor),
4. Isolating **runtime/file-system dependent** tests so they don’t pollute every run,
5. Documenting the test workflow for humans and agents.

This patch must NOT change engine behavior — only the test harness and related docs.

---

## Files in Scope

You will likely touch:

- Test configuration & docs
  - `pytest.ini`
  - `Makefile` (or `run_tests.sh`)
  - `docs/TESTING.md`
  - `tests/conftest.py`

- Test tree
  - `tests/unit/` (currently mostly empty)
  - `tests/integration/`
  - `tests/legacy/` (created in S0-C)

- New unit tests
  - `tests/unit/test_diagnostics_metrics_unit.py`
  - `tests/unit/test_state_publish_unit.py`
  - `tests/unit/test_vol_utils_unit.py`
  - `tests/unit/test_factor_diagnostics_unit.py`

- Integration/runtime tests (extensions)
  - `tests/integration/test_state_files_schema.py`
  - `tests/integration/test_state_publish_diagnostics.py`
  - `tests/integration/test_exit_pipeline_contract.py`
  - (possibly other core integration tests you need to tag)

---

## 1. Markers & Lanes

We already have `legacy` and some runtime-ish behavior. Now we formalize the lanes.

### 1.1 pytest.ini

**File:** `pytest.ini`

Ensure/extend:

```ini
[pytest]
testpaths =
    tests/unit
    tests/integration
    tests/legacy

markers =
    unit: fast, pure in-process tests without filesystem/external side effects
    integration: multi-module tests that may touch filesystem or state files
    runtime: tests that require a running loop, real logs/state, or longer execution
    legacy: v5/v6-era tests kept for reference, not part of v7.6 green bar

addopts = -q
````

> If these markers already exist, extend rather than replace.
> The important part is: **unit**, **integration**, **runtime**, **legacy** all defined.

### 1.2 Default Lanes

We want three canonical commands:

1. **Fast lane (day-to-day):**

   ```bash
   PYTHONPATH=. pytest tests/unit tests/integration -m "not runtime and not legacy" -q
   ```

2. **Runtime slice:**

   ```bash
   PYTHONPATH=. pytest tests/integration -m "runtime" -q
   ```

3. **All tests (including legacy):**

   ```bash
   PYTHONPATH=. pytest -q
   ```

We’ll wire these via Makefile / TESTING doc below.

---

## 2. Makefile / run_tests.sh Targets

### 2.1 Makefile Targets

**File:** `Makefile` (create if missing, or extend)

Add:

```make
test-fast:
\tPYTHONPATH=. pytest tests/unit tests/integration -m "not runtime and not legacy" -q

test-runtime:
\tPYTHONPATH=. pytest tests/integration -m "runtime" -q

test-all:
\tPYTHONPATH=. pytest -q
```

If you’re using `run_tests.sh`, reflect the same splits:

* `./run_tests.sh fast`
* `./run_tests.sh runtime`
* `./run_tests.sh all`

---

## 3. TESTING.md Renovation

**File:** `docs/TESTING.md`

Update/extend to clearly define:

1. **Directories**

   * `tests/unit/` — small, deterministic, no external IO.
   * `tests/integration/` — may touch JSON state files, multiple modules.
   * `tests/legacy/` — old v5/v6 tests, not maintained, can fail.

2. **Markers**

   * `@pytest.mark.unit`
   * `@pytest.mark.integration`
   * `@pytest.mark.runtime`
   * `@pytest.mark.legacy`

3. **Canonical commands**

   * Day-to-day: `make test-fast`
   * Runtime slice: `make test-runtime`
   * Full: `make test-all`

4. **Agent Guidance**

   * Agents should primarily keep `test-fast` green.
   * When patch touches runtime, they must run `test-runtime` at least once.

---

## 4. Seed Core Unit Tests

We want a minimal but meaningful set of **pure unit tests** for:

* diagnostics_metrics helpers,
* state_publish formatting,
* basic vol utilities,
* factor diagnostics math.

These tests should NOT hit filesystem, network, or real state files.

### 4.1 diagnostics_metrics unit tests

**File:** `tests/unit/test_diagnostics_metrics_unit.py` (NEW)

Scope:

* `record_veto`, `record_signal_emitted`, `record_order_placed`
* `record_exit_scan_run`, `record_exit_trigger`
* `compute_liveness_alerts` behavior in isolation (with fake config)

Tests:

1. **Counters increment correctly**

   * Start from clean `_veto_counters`, call record functions, assert totals and timestamps non-None (can monkeypatch `now_iso` if needed).

2. **Liveness with missing timestamps**

   * Provide cfg_liveness with small thresholds (e.g., 10s).
   * Set `last_signal_ts = None`, others None.
   * `compute_liveness_alerts()` → `idle_signals == True`, `missing["signals"] == True`.

3. **Liveness with fresh timestamps**

   * Monkeypatch `datetime.now` to just after `last_signal_ts`.
   * Ensure idle flags false, durations small.

4. **Liveness beyond threshold**

   * Simulate older timestamps.
   * Ensure idle flags true and `details["signals"] > threshold`.

Mark all tests:

```python
import pytest

pytestmark = pytest.mark.unit
```

Use monkeypatch for global state reset (or fixture in conftest).

---

### 4.2 state_publish unit tests

**File:** `tests/unit/test_state_publish_unit.py` (NEW)

Scope:

* `_atomic_write_state` behavior,
* formatting of `write_positions_state`, `write_positions_ledger_state`, `write_kpis_v7_state` without actually writing to disk (monkeypatch underlying writer), or by using tmp_path.

Tests:

1. **Atomic write uses temp file then replace**

   * Monkeypatch `os.replace` to capture calls.
   * Ensure it’s called exactly once.

2. **Positions state payload structure**

   * Call `write_positions_state([{"symbol": "BTCUSDT", ...}])` with monkeypatched `_atomic_write_state` that just inspects payload.
   * Assert payload has keys `updated_at`, `positions`, correct row fields.

3. **Ledger state payload structure**

   * Same pattern for `write_positions_ledger_state`.

4. **KPIs state defaults `updated_at`**

   * If `updated_at` not provided in kpis dict, writer fills it.

Mark as `@pytest.mark.unit`.

---

### 4.3 Vol utilities unit tests

If you have a `vol.py` or similar:

**File:** `tests/unit/test_vol_utils_unit.py` (NEW)

Scope:

* EWMA vol / realized volatility functions used by VaR and vol regimes.

Tests:

1. `compute_ewma_vol` with constant returns → predictable vol.
2. Different lambda values → smaller lambda → more weight on recent obs (behavioural).
3. Edge cases: empty returns → default / 0 behavior documented.

Mark as `@pytest.mark.unit`.

---

### 4.4 Factor diagnostics unit tests

**File:** `tests/unit/test_factor_diagnostics_unit.py` (NEW)

Scope:

* `normalize_factor_vectors`,
* `compute_factor_covariance` / correlation,
* `orthogonalize_factors` (Gram-Schmidt),
* `compute_raw_factor_weights`, `normalize_factor_weights`.

Tests:

1. Normalization:

   * Identity on already-normalized data,
   * z-score vs minmax produce expected ranges.

2. Covariance:

   * For orthogonal inputs, off-diagonal ~0.

3. Orthogonalization:

   * Output vectors pairwise orthogonal within small tolerance.

4. Weights:

   * Negative IR → low or zero weights (depending on design),
   * Sum of normalized weights = 1.

Mark as `@pytest.mark.unit`.

> Keep maths simple; rely on small synthetic vectors.

---

## 5. Runtime & Integration Test Isolation

Now we make sure **runtime-dependent tests** are clearly tagged and don’t slow the fast lane.

### 5.1 Tag runtime tests

**Files (examples, adjust to actual repo):**

* `tests/integration/test_state_files_schema.py`
* `tests/integration/test_exit_pipeline_contract.py`
* `tests/integration/test_state_publish_positions_ledger.py`
* Any that:

  * expect real logs/state under `logs/state/`,
  * depend on long-running loops or actual services.

Add at the top:

```python
import pytest

pytestmark = pytest.mark.integration

@pytest.mark.runtime
def test_something_runtime(...):
    ...
```

Or, if the whole file is runtime-level:

```python
import pytest
pytestmark = [pytest.mark.integration, pytest.mark.runtime]
```

### 5.2 Replace real FS usage with tmp_path where possible

For tests that don’t strictly need real production log directories:

* Use `tmp_path` fixture,
* Monkeypatch state paths in `execution/state_v7` or `state_publish` to point into tmpdir,
* Write/read files there.

Goal:

* Most integration tests should remain **fast** and not depend on pre-existing logs.
* Only a small subset truly need “realistic” layouts; these get `@pytest.mark.runtime`.

---

## 6. Fixtures & Global State Resets

**File:** `tests/conftest.py`

Ensure you have a global fixture to reset diagnostics state and other globals between tests.

Example:

```python
import pytest
from execution import diagnostics_metrics

@pytest.fixture(autouse=True)
def reset_diagnostics_state():
    diagnostics_metrics._veto_counters = diagnostics_metrics.VetoCounters()
    diagnostics_metrics._exit_status = diagnostics_metrics.ExitPipelineStatus()
    yield
    # if needed, clean up again
```

Mark this fixture as `autouse=True` so unit/integration tests don’t bleed state into each other.

You can add tiny fixtures for synthetic positions, TP/SL registry entries, etc., but keep them generic.

---

## 7. Schema Test Extension

We already extended `test_state_files_schema.py` in S1; S3 should just ensure:

* It’s marked `@pytest.mark.integration` and `@pytest.mark.runtime` if it expects real logs/state.
* Minimal required keys for:

  * `positions_state`
  * `positions_ledger`
  * `kpis_v7`
  * `runtime_diagnostics`

And that the test **skips gracefully** if files don’t exist:

```python
if not path.exists():
    pytest.skip(f"{path} not present, skipping schema check")
```

---

## 8. Acceptance Criteria

The patchset is complete when:

1. **Markers & Lanes**

   * pytest markers: `unit`, `integration`, `runtime`, `legacy` are defined in `pytest.ini`.
   * `make test-fast`, `make test-runtime`, `make test-all` (or equivalent) exist and work.

2. **Unit Tests Seeded**

   * `tests/unit/test_diagnostics_metrics_unit.py` exists and passes.
   * `tests/unit/test_state_publish_unit.py` exists and passes.
   * `tests/unit/test_vol_utils_unit.py` exists and passes (if vol utilities exist).
   * `tests/unit/test_factor_diagnostics_unit.py` exists and passes.

3. **Runtime Isolation**

   * Long-running / stateful tests are tagged `@pytest.mark.runtime`.
   * `make test-fast` excludes them via `-m "not runtime and not legacy"`.

4. **Fixtures & Stability**

   * `conftest.py` has a diagnostics reset fixture (and any other necessary state resets).
   * No test ordering dependencies (can run tests multiple times in different orders).

5. **Docs**

   * `docs/TESTING.md` reflects the new lanes, markers, and commands.
   * Agents/humans know that:

     * Engine patches → run `test-fast`
     * Diagnostics/state patches → run `test-fast` + relevant runtime tests.

6. **Green Bar**

   * `PYTHONPATH=. pytest tests/unit tests/integration -m "not runtime and not legacy" -q` passes.
   * `PYTHONPATH=. pytest tests/integration -m "runtime" -q` passes (when logs/state exist).
   * Full suite (`pytest -q`) may have legacy failures, but **unit+integration lanes are green.**

This completes **PATCHSET_v7.6_S3 — Test Suite Renovation**, giving v7.6 a clean, well-labeled test harness ready for deeper execution & factor work.

```
```
