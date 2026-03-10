# Phase 4 — ECS Migration Runbook

> **Trigger:** `[architecture] ecs_ready=true` in executor logs.
> **Principle:** No large atomic rewrite. Each commit keeps the system runnable and tests green.

---

## Current Architecture

```
engines → merge → conviction enrichment → fallback swap → execution
```

## Target Architecture

```
engines → candidate set → conviction enrichment → select executable → execution
```

---

## Commit 1 — Introduce Selector (Additive Only)

**Create:** `execution/candidate_selector.py`

```python
build_candidates(symbol, legacy_intent, hydra_intent)
enrich_candidates_with_conviction(candidates, state)
select_executable_candidate(candidates, config)
```

**Return type:**

```python
{
    "selected": intent | None,
    "candidates": [...],
    "winner_engine": str,
    "loser_engine": str | None,
}
```

**Rules:**
- Do NOT modify executor behavior
- Add unit tests for selector only

**Tests:** `tests/unit/test_candidate_selector.py`
- single candidate
- two candidates, hydra wins
- two candidates, legacy wins
- both fail conviction

**Outcome:** ~4100 tests, system unchanged.

---

## Commit 2 — Shadow Selector

Executor runs selector **in parallel** with current fallback logic.

```python
shadow = candidate_selector(...)
# execution still uses old fallback path
```

**Telemetry:**

```
[selector_shadow] legacy=hydra ecs=legacy match=true
```

**Purpose:** Verify selector decisions match fallback decisions.

**Expected runtime:** 24–48 hours shadow soak.

---

## Commit 3 — Enable ECS Path (Behind Flag)

**Config:** `runtime.yaml`

```yaml
execution:
  use_ecs_selector: false
```

**Executor path:**

```python
if use_ecs_selector:
    intent = selector.selected
else:
    intent = fallback_logic(...)
```

**Outcome:** System supports dual architectures. Unit tests run both modes.

---

## Commit 4 — Telemetry Rebase

Telemetry sources change from fallback fields to selector output.

| Old Source             | New Source                                  |
|------------------------|---------------------------------------------|
| `merge_conflict`       | `len(candidates) > 1`                       |
| `fallback_used`        | `selected_engine != highest_score_engine`    |
| `merge_primary_engine` | `highest_score_engine`                       |
| `merge_legacy_score`   | candidate list                               |
| `merge_hydra_score`    | candidate list                               |

**Metrics preserved:** CEL, HQD, participation, SDD, RSD, RDD, MRI.

No dashboard changes required.

---

## Commit 5 — Remove Fallback Branch

**Delete:**
- `_fallback` field attachment
- fallback swap logic (band-rank comparison)
- fallback attribution rewrite
- fallback telemetry counters

**Expected deletions:** ~800–1200 LOC.

**Executor simplifies to:**

```python
candidates = build_candidates(...)
selected = select_executable_candidate(...)
if selected:
    _send_order(...)
```

---

## Commit 6 — Test Suite Cleanup

**Remove:** `TestFallbackSwap`, `TestFallbackAttribution`, fallback telemetry tests.

**Replace with:** `TestCandidateSelector`, `TestECSExecution`.

Total tests likely shrink slightly but remain >4000.

---

## Commit 7 — Telemetry Simplification

**Remove obsolete metrics:** `fallback_rate`, `fallback_edge_delta`, `fallback_used`.

**MRI updated:** `stable_recovery` condition → replaced with selector agreement rate.

**Dashboard:** Architecture Status shows `ECS Mode: ACTIVE`.

---

## Migration Safety Checklist

Before enabling ECS flag:

- [ ] MRI READY (`ecs_ready=true`)
- [ ] `fallback_rate < 5%`
- [ ] CEL positive
- [ ] SDD stable (`|sdd| <= 0.02`)
- [ ] Shadow selector agreement > 95%

---

## Final Executor Shape

```python
for symbol in symbols:
    candidates = candidate_selector.build(...)
    candidate_selector.enrich(...)
    selected = candidate_selector.select(...)
    if selected:
        _send_order(selected)
```

No branching. No fallback.

---

## Expected Gains

| Dimension           | Before          | After           |
|---------------------|-----------------|-----------------|
| `executor_live.py`  | ~5700 lines     | ~4300 lines     |
| Execution path      | merge→enrich→swap→attribution | candidates→select→execute |
| Engine scalability   | 2 engines       | N engines       |

---

## Resulting Code Structure

```
execution/
├─ executor_live.py          # orchestration only
├─ candidate_selector.py     # selection boundary
├─ conviction_engine.py      # conviction sizing
├─ hydra_integration.py      # engine bridge
├─ risk_limits.py            # risk gate
```

---

## Architectural Milestone

```
Phase 1  single engine
Phase 2  engine competition
Phase 3  competition + recovery + observability  ← current
Phase 4  executable candidate selection          ← this runbook
Phase 5  engine-agnostic executor                ← below
```

Recovery architecture → Deterministic selection architecture.

The only trigger needed: `[architecture] ecs_ready=true`.

---

---

# Phase 5 — Engine-Agnostic Executor (Post-ECS Slice)

> **Trigger:** Phase 4 Commit 5 complete (fallback branch deleted).
> **Goal:** Remove all engine-specific intent plumbing from executor. ~300–500 LOC deletion.

Once ECS provides a single decision boundary, the executor no longer needs to understand how many engines exist or which one produced the selected intent.

---

## What Becomes Deletable

### 1. Engine-Specific Intent Fields (~150–200 LOC)

**Before ECS**, intents carry:

```
intent.legacy_score
intent.hydra_score
intent.fallback
intent.fallback_reason
intent.swap_reason
intent.original_engine
```

**After:** Executor sees only:

```
intent.engine
intent.conviction
intent.size
intent.symbol
```

Everything else is **selector metadata**, not executor state.

### 2. Intent Normalization / Attribution Repair (~150–200 LOC)

**Before:**

```
normalize_intent()
coerce_veto_reasons()
rebuild_intent_after_swap()
patch_engine_attribution()
```

**After:** `intent = selector.selected` — normalization lives in `candidate_selector`.

### 3. Engine-Aware Telemetry Branches (~120–160 LOC)

**Before:** Executor emits `merge_conflict`, `fallback_rate`, `fallback_engine`, `swap_reason`, `merge_primary_engine`, etc.

**After:** Selector result carries `candidates`, `selected_engine`, `candidate_count`. Executor records only `selected_engine`, `candidate_count`, `selected_conviction`.

### 4. Engine-Count Execution Branches (~100–140 LOC)

**Before:**

```python
if hydra_intent and legacy_intent:
    run_merge_logic()
elif hydra_intent:
    use_hydra()
elif legacy_intent:
    use_legacy()
```

**After:**

```python
intent = selector.selected
if intent:
    _send_order(intent)
```

---

## Net Structural Impact (Phase 4 + Phase 5)

| Component            | Before (now) | After Phase 4 | After Phase 5   |
|----------------------|-------------|---------------|-----------------|
| `executor_live.py`   | ~5700 LOC   | ~4300 LOC     | ~3800–4000 LOC  |
| Arbitration logic    | executor    | selector      | selector        |
| Engine awareness     | executor    | executor      | **none**        |
| Fallback logic       | present     | deleted       | deleted         |
| Attribution repair   | present     | deleted       | deleted         |
| Intent normalization | executor    | executor      | **selector**    |

---

## Operational Benefits

**1. Engine-agnostic executor.** Adding a third engine: `engines → selector → executor`. Executor never changes.

**2. Truthful telemetry.** Current telemetry reflects post-swap attribution (a distortion). ECS telemetry reflects actual candidate competition.

**3. Governance clarity.** DLE wants clear authority boundaries. ECS provides: `selector = decision layer`, `executor = execution layer`.

---

## Final Code Structure

```
execution/
├─ executor_live.py          (~3800 LOC, thin orchestrator)
├─ candidate_selector.py     (competition + selection)
├─ conviction_engine.py      (conviction sizing)
├─ order_dispatch.py         (exchange dispatch)
├─ fill_tracker.py           (fill polling)
├─ risk_limits.py            (risk gate)
```

---

---

# `_loop_once()` Collapse — The Architectural Proof

> The single function where Phases 4 + 5 converge.

## Before (~320 LOC)

```python
def _loop_once(i):

    _sync_dry_run()
    _refresh_risk_config()

    account = _account_snapshot()
    positions = get_positions()

    check_for_testnet_reset()

    # EXIT PATH
    exits = scan_all_exits(positions)
    for exit_signal in exits:
        _send_order(exit_signal)

    # ENTRY PATH
    legacy_intents = generate_legacy_intents()
    hydra_intents  = generate_hydra_intents()

    for symbol in universe:

        legacy = legacy_intents.get(symbol)
        hydra  = hydra_intents.get(symbol)

        if legacy and hydra:
            intent = merge_logic(legacy, hydra)
        elif hydra:
            intent = hydra
        elif legacy:
            intent = legacy
        else:
            intent = None

        if intent:
            intent = conviction_enrichment(intent)

            if intent.fallback:
                intent = fallback_swap(intent)

            _send_order(intent)

    _pub_tick()
    _maybe_run_telegram_alerts()
    _maybe_run_pipeline_v6_compare()
```

**Problems:** Two engine pipelines, merge logic, fallback swap, attribution repair,
conviction enrichment split across layers, executor knows engine internals.
The executor is effectively a strategy engine, not just an executor.

## After Phase 4 + Phase 5 (~90 LOC)

```python
def _loop_once(cycle):

    state = refresh_runtime_state()

    # exits always run first
    exits = exit_scanner.scan(state.positions)
    for exit_intent in exits:
        executor.send(exit_intent)

    # entry candidates
    for symbol in universe:
        candidates = selector.build_candidates(symbol, state)
        selector.enrich_with_conviction(candidates, state)
        selected = selector.select(candidates)

        if selected is None:
            continue

        executor.send(selected)

    telemetry.publish_cycle(state)
```

**Four responsibilities:** refresh state, process exits, select entry candidate, send order. Nothing else.

## Further Stage Split (~35–50 LOC)

```python
def _loop_once():
    state = collect_cycle_state()
    run_exit_pipeline(state)
    run_entry_pipeline(state)
    publish_cycle_state(state)
```

## What Disappears

| Removed Surface           | Reason                       |
|---------------------------|------------------------------|
| `legacy_intent`           | engine abstraction removed   |
| `hydra_intent`            | engine abstraction removed   |
| merge logic               | replaced by candidate set    |
| fallback swap             | recovery no longer needed    |
| attribution repair        | selector handles attribution |
| swap telemetry            | obsolete                     |
| engine-specific telemetry | selector metadata replaces   |

---

---

# Phase 6 — `_send_order()` Decomposition (~1200–1400 LOC Removal)

> **Trigger:** Phase 5 complete (executor is engine-agnostic).
> **Goal:** Break `_send_order()` monolith into clean subsystems.

`_send_order()` is the single largest function in the codebase. Once the executor
is a thin orchestrator, this becomes the obvious next target.

## Current `_send_order()` Shape

A ~1500 LOC function carrying 12 inline gates, position flip detection,
risk checks, order dispatch, fill tracking, and post-fill hooks:

```
doctrine_gate
churn_guard
strategy_attribution
conviction_band_check
kill_switch
risk_check
fee_edge_check
notional_guard
position_flip_handler
order_dispatch
fill_tracking
post_fill_hooks
```

## Target Decomposition

```
execution/
├─ pre_order_checks.py      # doctrine, churn, conviction, kill_switch
├─ flip_handler.py          # close-then-open for direction changes
├─ risk_gate.py             # risk_check, fee_edge, notional_guard
├─ order_dispatch.py        # maker-first POST_ONLY + taker fallback (exists)
├─ fill_tracker.py          # fill polling, PnL close (exists)
├─ post_fill_hooks.py       # ledger update, telemetry, alerts
```

## Executor Call Site

```python
def send_order(state, intent):
    intent = pre_order_checks.run(state, intent)
    if intent is None:
        return  # vetoed

    if flip_handler.needs_flip(state, intent):
        flip_handler.execute_flip(state, intent)
        return

    risk_gate.check(state, intent)

    result = order_dispatch.send(state, intent)

    fill_tracker.track(result)

    post_fill_hooks.run(state, intent, result)
```

## Expected Impact

| Component          | Before    | After           |
|--------------------|-----------|-----------------|
| `_send_order()`    | ~1500 LOC | deleted (6 modules) |
| `executor_live.py` | ~3800 LOC | ~2400–2600 LOC  |
| Gate logic         | inline    | `pre_order_checks.py` |
| Flip logic         | inline    | `flip_handler.py` |
| Risk enforcement   | inline    | `risk_gate.py` |

## Why This Matters

Each gate becomes independently testable. Currently the 12-gate chain is
tested through `_send_order()` integration tests. After decomposition:

```
test_pre_order_checks.py   — doctrine, churn, conviction, kill_switch
test_flip_handler.py       — close+open sequencing
test_risk_gate.py          — risk, fee_edge, notional
test_post_fill_hooks.py    — ledger, telemetry, alerts
```

---

---

# Full Migration Arc

```
v7.x executor:             ~5700 LOC
Phase 4  ECS:              ~4300 LOC  (remove fallback)
Phase 5  engine purge:     ~3800 LOC  (remove engine awareness)
         stage split:      ~3400 LOC  (_loop_once → 4 stages)
Phase 6  _send_order split:~2400 LOC  (6 focused modules)
```

```
Before:  executor = strategy engine + risk engine + telemetry engine
After:   executor = deterministic orchestrator calling focused subsystems
```

Aligns with DLE constitution: execution performs permitted actions, does not make decisions.
