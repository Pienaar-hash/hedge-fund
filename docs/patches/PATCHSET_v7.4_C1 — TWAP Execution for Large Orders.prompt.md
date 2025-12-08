# PATCHSET v7.4_C1 — TWAP Execution for Large Orders

## Objective

Add a **simple TWAP execution style** for large orders that:

- Splits large notional into multiple slices.
- Uses **existing maker-first routing** for each slice.
- Spaces slices over a short time window (intra-bar).
- Tags and measures TWAP executions for router metrics and telemetry.

This must:

- Fully respect the existing **risk pipeline** (no bypasses).
- Remain backward-compatible for all non-TWAP orders.
- Plug into the current router + events + metrics architecture.

---

## Files to Touch

- `config/runtime.yaml`
- `execution/runtime_config.py`
- `execution/order_router.py`
- `execution/events.py`
- `execution/router_metrics.py`
- `tests/test_order_router_twap.py` (new)

No changes to executor entrypoints, screener, or risk_limits.

---

## 1. Config: TWAP Settings

**File:** `config/runtime.yaml`

### Add a `twap` section under router/runtime settings

Example:

```yaml
router:
  twap:
    enabled: true
    min_notional_usd: 500.0
    slices: 4
    interval_seconds: 10
````

Rules:

* `enabled`: global feature flag; when `false`, router **never** TWAPs.
* `min_notional_usd`: minimum **gross USD** notional to consider TWAP.
* `slices`: number of child orders per TWAP execution.
* `interval_seconds`: sleep time between slices (can be 0 for tight environments, but default 5–10s).

---

## 2. Runtime Config Helper

**File:** `execution/runtime_config.py`

### Add a TWAP config accessor

Create a small config dataclass:

```python
from dataclasses import dataclass

@dataclass
class TWAPConfig:
    enabled: bool
    min_notional_usd: float
    slices: int
    interval_seconds: float
```

Add loader:

```python
def get_twap_config() -> TWAPConfig:
    cfg = _load_runtime_config()  # whatever existing loader is
    twap_cfg = cfg.get("router", {}).get("twap", {}) or {}
    return TWAPConfig(
        enabled=bool(twap_cfg.get("enabled", False)),
        min_notional_usd=float(twap_cfg.get("min_notional_usd", 0.0)),
        slices=int(twap_cfg.get("slices", 1)),
        interval_seconds=float(twap_cfg.get("interval_seconds", 0.0)),
    )
```

Safety:

* Clamp `slices` to at least `1`.
* If `enabled=False` or `slices <= 1`, router should behave like pre-TWAP.

---

## 3. Router: TWAP Path

**File:** `execution/order_router.py`

### 3.1 Decide when to TWAP

Inside the main routing function (e.g. `route_intent` / `route_order`):

* Compute `gross_usd` as you already do for risk / min_notional.
* Get `twap_cfg = runtime_config.get_twap_config()`.

Then:

```python
use_twap = (
    twap_cfg.enabled
    and gross_usd >= twap_cfg.min_notional_usd
    and twap_cfg.slices > 1
)
```

If `use_twap`:

* Call a new helper, e.g. `_route_twap(...)`.
* Else: keep existing single-shot routing path untouched.

### 3.2 Implement `_route_twap(...)`

Add:

```python
def _route_twap(
    intent: OrderIntent,
    twap_cfg: TWAPConfig,
    *,
    client: ExchangeClient,
    logger: Logger,
    # plus any runtime/router context you already pass around
) -> List[ChildOrderResult]:
    """
    Execute a TWAP for the given intent:
    - Split qty into N slices.
    - For each slice, call existing maker-first routing.
    - Sleep between slices as per config.
    - Collect child results and return.
    """
```

Implementation guidelines:

1. **Split quantity**:

   * Let `total_qty = intent.qty`.
   * Compute `slice_qty = total_qty / twap_cfg.slices`.
   * For last slice, adjust to fix rounding error so sum(children) == total.

2. **Reuse existing routing**:

   * You should already have a function like `_route_single_order(...)` that does maker-first + taker fallback.

   * Call it per slice:

     ```python
     for i in range(twap_cfg.slices):
         child_intent = intent.with_qty(slice_qty_i)
         result = _route_single_order(child_intent, client=client, logger=logger, execution_style="twap")
         results.append(result)
         if i < twap_cfg.slices - 1 and twap_cfg.interval_seconds > 0:
             time.sleep(twap_cfg.interval_seconds)
     ```

   * **Do not** re-run risk_limits here; the executor should have already passed the full intent through risk. We are only slicing execution.

3. **Min notional sanity**:

   * Ensure each `slice_qty` is above the exchange **min notional** for this symbol.
   * If `slice_qty` would drop below min notional:

     * Either:

       * Reduce `slices` dynamically so each slice >= min_notional, or
       * Fallback to single-shot path with a warning log.
     * Keep behaviour deterministic and tested.

4. **Return semantics**:

   * If the router previously returned a single `OrderResult`, now return a structure that is either:

     * A list of `ChildOrderResult` combined into a summary, or
     * A consistent aggregate result with underlying children logged.

   * Respect existing API; if you need to extend a struct, do so **additively** (e.g. add `children` field).

---

## 4. Events: Tag TWAP Executions

**File:** `execution/events.py`

Where you emit structured router events (e.g. to `router_events.jsonl`):

* Add `execution_style` field:

  * `"single"` for normal orders (default).
  * `"twap"` for TWAP child orders.

For TWAP:

* For each child order event, log:

```json
{
  "execution_style": "twap",
  "twap": {
    "slice_index": i,
    "slice_count": N,
    "slice_qty": ...,
    "parent_gross_usd": ...,
    "twap_cfg": {
      "min_notional_usd": ...,
      "slices": ...,
      "interval_seconds": ...
    }
  }
}
```

No need to be excessively verbose, but enough to analyse later.

---

## 5. Router Metrics: TWAP vs Non-TWAP

**File:** `execution/router_metrics.py`

Extend the metrics aggregation to separate TWAP vs non-TWAP executions:

* Maintain metrics like:

  * `twap_trades_count`
  * `twap_avg_slippage_bps`
  * `single_trades_count`
  * `single_avg_slippage_bps`

Compute slippage the same way as existing metrics (if slippage already exists; otherwise stub with neutral values for now but keep structure).

These metrics will be useful both in dashboards and investor packs.

---

## 6. Tests

**File:** `tests/test_order_router_twap.py` (new)

Add tests with a **fake exchange client** (no real HTTP):

1. **Below threshold → no TWAP**

   * `twap.enabled = True`, `min_notional_usd = 500`.
   * `gross_usd = 300`.
   * Ensure:

     * `use_twap` is `False`.
     * Single-shot routing path used.
     * `execution_style` logged as `"single"`.

2. **Above threshold → TWAP**

   * `gross_usd = 1000`, `min_notional_usd = 500`, `slices = 4`.
   * Verify:

     * `_route_twap` is called.
     * Exactly 4 child orders created.
     * `sum(child_qty) == total_qty` (within float tolerance).
     * All child events tagged `execution_style = "twap"`.

3. **Min notional sanity**

   * Configure symbol min notional so that `total_qty / slices` falls below min.
   * Ensure router:

     * Either reduces slices, or falls back to single route **as per your chosen behaviour**, and
     * Tests assert that behaviour.

4. **Config off → no TWAP**

   * `enabled = False` → all orders are `"single"`.

5. **Interval logic**

   * You don’t need real `time.sleep` in tests; patch or mock it so:

     * `_route_twap` calls sleep N-1 times for N slices when interval > 0.
     * Zero times when interval = 0.

All existing tests must remain green.

---

## 7. Acceptance Criteria

The patch is complete when:

1. `config/runtime.yaml` contains a `router.twap` config with `enabled`, `min_notional_usd`, `slices`, and `interval_seconds`.

2. `runtime_config.get_twap_config()` returns a sane `TWAPConfig` struct with defaults when config is missing/partial.

3. `order_router`:

   * Uses TWAP only when:

     * `enabled=True`
     * `gross_usd >= min_notional_usd`
     * `slices > 1`
   * Slices orders correctly.
   * Reuses existing maker-first path for each slice.
   * Honors min-notional constraints deterministically.

4. `events`:

   * Tag TWAP vs single executions using `execution_style`.
   * Log basic TWAP metadata for each child.

5. `router_metrics`:

   * Track at least counts for TWAP vs single executions.
   * (Optional) track average slippage for TWAP vs single.

6. New tests in `test_order_router_twap.py` pass, and all existing tests (A1, A2, B1, B2) remain green.

No contracts (risk, state, screener, executor entrypoints) are broken in the process.

```