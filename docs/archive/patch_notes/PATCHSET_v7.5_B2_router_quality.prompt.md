# PATCHSET v7.5_B2 — Dynamic Router Quality Score & Hybrid Integration

## Objective

Turn the router into a **first-class signal component** by:

1. Computing a **per-symbol router_quality_score ∈ [0, 1]** from:
   - EWMA expected slippage (bps)
   - EWMA realized slippage (bps)
   - Slippage drift (realized - expected)
   - Liquidity bucket (A/B/C)
   - TWAP skip vs execute behaviour

2. Feeding router_quality_score into:
   - **Hybrid scoring** (symbol_score_v6.py)
   - **Signal screener** ranking/filtering
   - **Telemetry & dashboard** (router quality panel)

3. Remaining:
   - **Backwards compatible** (no breaking of existing contracts)
   - **Config-driven** (weights + scaling in strategy_config.json)
   - **Runtime safe** (no blocking, no hard veto yet)

---

## Files to Touch / Add

- `config/strategy_config.json` (extend)
- `execution/router_metrics.py` (extend)
- `execution/slippage_model.py` (extend small helpers if needed)
- `execution/state_publish.py` (extend)
- `execution/state_v7.py` (extend loader)
- `execution/symbol_score_v6.py` (extend hybrid scoring)
- `execution/signal_screener.py` (extend ranking/filtering)
- `dashboard/execution_panel.py` (extend: router quality visualization)
- Tests:
  - `tests/test_router_quality_score.py` (NEW)
  - `tests/test_hybrid_score_router_quality.py` (NEW)
  - `tests/test_signal_screener_router_quality.py` (NEW)
  - `tests/test_state_publish_router_quality.py` (NEW/extend)

---

## 1. Config Additions

**File:** `config/strategy_config.json`

Add:

```json
"router_quality": {
  "enabled": true,
  "base_score": 0.8,
  "min_score": 0.2,
  "max_score": 1.0,
  "slippage_drift_bps_thresholds": {
    "green": 2.0,
    "yellow": 6.0
  },
  "bucket_penalties": {
    "A_HIGH": 0.0,
    "B_MEDIUM": -0.05,
    "C_LOW": -0.15
  },
  "twap_skip_penalty": 0.10,
  "low_quality_hybrid_multiplier": 0.5,
  "high_quality_hybrid_multiplier": 1.05
}
````

Interpretation:

* `base_score`: starting point before penalties/bonuses.
* `bucket_penalties`: permanent haircut per bucket.
* `slippage_drift_bps_thresholds`:

  * `green`: drift <= green → little/no penalty
  * `yellow`: between green & yellow → medium penalty
  * > yellow → strong penalty
* `twap_skip_penalty`: applied when a high fraction of TWAP slices are skipped.
* `low_quality_hybrid_multiplier`: factor applied to hybrid_score when router_quality is poor.
* `high_quality_hybrid_multiplier`: slight boost when router_quality is excellent.

---

## 2. Router Quality Score Computation

**File:** `execution/router_metrics.py`

We already have EWMA slippage metrics & liquidity buckets from B1.

### 2.1 Data Model

Add:

```python
from dataclasses import dataclass

@dataclass
class RouterQualitySnapshot:
    score: float
    bucket: str
    ewma_expected_bps: float
    ewma_realized_bps: float
    slippage_drift_bps: float
    twap_skip_ratio: float
    trade_count: int
```

### 2.2 Score Function

Implement:

```python
def compute_router_quality_score(
    *,
    bucket: str,
    ewma_expected_bps: float,
    ewma_realized_bps: float,
    twap_skip_ratio: float,
    cfg: RouterQualityConfig
) -> float:
    """
    Computes a router_quality_score ∈ [min_score, max_score].
    """
```

Logic (v1, simple & monotonic):

1. Start with `score = cfg.base_score`.

2. Apply bucket penalty:

   ```python
   score += cfg.bucket_penalties.get(bucket, 0.0)
   ```

3. Compute drift:

   ```python
   drift = ewma_realized_bps - ewma_expected_bps
   ```

4. Slippage drift adjustment:

   * If `drift <= green` → small negative or zero adjustment.
   * If `green < drift <= yellow` → moderate negative adjustment.
   * If `drift > yellow` → strong negative adjustment.

   Example:

   ```python
   if drift <= cfg.slippage_drift_bps_thresholds["green"]:
       score -= 0.02
   elif drift <= cfg.slippage_drift_bps_thresholds["yellow"]:
       score -= 0.08
   else:
       score -= 0.18
   ```

5. TWAP skip penalty:

   * Consider `twap_skip_ratio` = skipped_slices / total_slices (0–1).
   * Apply penalty `score -= cfg.twap_skip_penalty * twap_skip_ratio`.

6. Clamp:

   ```python
   score = max(cfg.min_score, min(cfg.max_score, score))
   ```

### 2.3 Snapshot Builder

In `router_metrics.py`:

```python
def build_router_quality_snapshot(
    symbol: str,
    slippage_stats: SymbolSlippageSnapshot,
    bucket: str,
    twap_skip_ratio: float,
    cfg: RouterQualityConfig
) -> RouterQualitySnapshot:
    score = compute_router_quality_score(
        bucket=bucket,
        ewma_expected_bps=slippage_stats.ewma_expected_bps,
        ewma_realized_bps=slippage_stats.ewma_realized_bps,
        twap_skip_ratio=twap_skip_ratio,
        cfg=cfg,
    )
    return RouterQualitySnapshot(
        score=score,
        bucket=bucket,
        ewma_expected_bps=slippage_stats.ewma_expected_bps,
        ewma_realized_bps=slippage_stats.ewma_realized_bps,
        slippage_drift_bps=slippage_stats.ewma_realized_bps - slippage_stats.ewma_expected_bps,
        twap_skip_ratio=twap_skip_ratio,
        trade_count=slippage_stats.trade_count,
    )
```

---

## 3. State Publishing & Loading

### 3.1 State Publishing

**File:** `execution/state_publish.py`

Extend router state:

```python
# in write_router_snapshot_state or equivalent:
state["router_quality"] = {
    symbol: {
        "score": snapshot.score,
        "bucket": snapshot.bucket,
        "ewma_expected_bps": snapshot.ewma_expected_bps,
        "ewma_realized_bps": snapshot.ewma_realized_bps,
        "slippage_drift_bps": snapshot.slippage_drift_bps,
        "twap_skip_ratio": snapshot.twap_skip_ratio,
        "trade_count": snapshot.trade_count,
    }
    for symbol, snapshot in router_quality_snapshots.items()
}
```

### 3.2 State Loader

**File:** `execution/state_v7.py`

Add:

```python
def load_router_quality() -> Dict[str, dict]:
    """
    Loads router_quality block from router state file.
    Returns an empty dict if not present.
    """
```

This will be used by:

* symbol_score_v6 (for hybrid scoring)
* dashboards (execution panel)
* screener (optional)

---

## 4. Hybrid Scoring Integration

**File:** `execution/symbol_score_v6.py`

We already have:

* hybrid_score composition
* decay
* vol regime modifiers
* carry / trend / expectancy / router components

### 4.1 Config Dataclass

Add:

```python
@dataclass
class RouterQualityHybridConfig:
    enabled: bool
    low_quality_multiplier: float
    high_quality_multiplier: float
    low_score_threshold: float
    high_score_threshold: float
```

Derived from `strategy_config["router_quality"]`.

### 4.2 Applying Router Quality

In `hybrid_score()` where final score is composed:

1. Retrieve router_quality_score for symbol (if present) via loader / passed-in context.

2. If `router_quality.enabled`:

   ```python
   if rq_score <= cfg.low_score_threshold:
       hybrid *= cfg.low_quality_hybrid_multiplier
   elif rq_score >= cfg.high_score_threshold:
       hybrid *= cfg.high_quality_hybrid_multiplier
   # else: leave hybrid unchanged
   ```

Suggested defaults:

* `low_score_threshold`: 0.5
* `high_score_threshold`: 0.9
* `low_quality_hybrid_multiplier`: 0.5
* `high_quality_hybrid_multiplier`: 1.05

3. Ensure final hybrid score still clamped to [-1, 1].

---

## 5. Signal Screener Integration

**File:** `execution/signal_screener.py`

Add optional **router-quality-aware gating**:

1. When building candidate list:

   * Attach `router_quality_score` per symbol (if available).
   * If score missing, treat as neutral (e.g., 0.8) or skip gating.

2. Before emitting intents:

   * Drop candidates with extremely poor router quality:

     ```python
     if rq_score < router_quality_min_for_emission:
         continue
     ```

   `router_quality_min_for_emission` can be derived from config (e.g., 0.35–0.4).

3. For ranking:

   * Keep primary sort by `hybrid_score`.
   * Secondary sort by router_quality_score (higher first) to prefer liquid, well-executing names when hybrid scores are similar.

This is a **soft prioritisation**, not a veto.

---

## 6. Dashboard: Router Quality Visualization

**File:** `dashboard/execution_panel.py`

Add a **Router Quality** section:

### 6.1 Table

Columns:

* Symbol
* Bucket (A/B/C)
* Router Score (0–1)
* EWMA Exp. (bps)
* EWMA Real. (bps)
* Drift (bps)
* TWAP Skip Ratio

Coloring:

* Router Score:

  * ≥ 0.9 → green
  * 0.7–0.9 → yellow
  * < 0.7 → red

* Drift sign:

  * Negative or small positive → green
  * Large positive (> yellow threshold) → red

### 6.2 Badge in Existing Tables

Optionally:

* Show a small router “health dot” per symbol in existing execution/slippage tables:

  * 🟢 good (score ≥ 0.9)
  * 🟡 ok
  * 🔴 poor

---

## 7. Tests

### 7.1 `tests/test_router_quality_score.py` (NEW)

Cases:

1. **Ideal conditions:**

   * Low drift, bucket A_HIGH, low twap_skip_ratio → score near max_score.

2. **Mediocre conditions:**

   * Medium drift, bucket B_MEDIUM → score ~ base_score - small penalties.

3. **Bad conditions:**

   * High drift, bucket C_LOW, high twap_skip_ratio → score near min_score.

4. Clamp behaviour:

   * Ensure score never < min_score or > max_score.

### 7.2 `tests/test_hybrid_score_router_quality.py` (NEW)

* With router_quality disabled → hybrid unchanged.
* With enabled:

  * rq_score < low_threshold → hybrid scaled by low_quality_multiplier.
  * rq_score > high_threshold → hybrid scaled by high_quality_multiplier.
* Combined with decay/regime logic still results in clamped [-1, 1].

### 7.3 `tests/test_signal_screener_router_quality.py` (NEW)

* Candidates with low router_quality_score are filtered out if below min_for_emission.
* Among similar hybrid scores, higher router_quality is preferred.

### 7.4 `tests/test_state_publish_router_quality.py` (NEW or extend)

* `router_quality` block exists with correct keys.
* Dashboard loader consumes the block without error.

All new tests must pass; existing suites remain green (or with pre-existing, unchanged failures).

---

## 8. Acceptance Criteria

The patchset is complete when:

1. `router_quality` config exists and is used to compute per-symbol router_quality_score.
2. `router_metrics` computes a meaningful score from:

   * EWMA expected/realized slippage
   * Liquidity bucket
   * TWAP skip ratio
3. State publishing exposes a `router_quality` block in router state.
4. `symbol_score_v6` uses router_quality to modulate hybrid_score when enabled.
5. `signal_screener`:

   * can filter out worst router_quality symbols
   * prefers higher router_quality for tie-breaking.
6. Dashboard’s execution panel shows a clear router quality view.
7. Tests in:

   * `test_router_quality_score.py`
   * `test_hybrid_score_router_quality.py`
   * `test_signal_screener_router_quality.py`
   * `test_state_publish_router_quality.py`
     all pass and do not break existing behaviour.

No changes may be made to:

* Risk veto contracts (DD, VaR, CVaR, correlation caps)
* Position ledger semantics
* TWAP config contracts

beyond what is described above.

```
