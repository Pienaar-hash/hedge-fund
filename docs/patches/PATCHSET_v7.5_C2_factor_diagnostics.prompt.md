# PATCHSET v7.5_C2 — Factor Diagnostics, Covariance & Attribution

## Objective

Turn GPT-Hedge v7.5 into a **true multi-factor model** with:

1. **Factor Diagnostics**
   - Per-symbol factor vector: [trend, carry, rv_momo, router_quality, expectancy, vol_regime, etc.]
   - Factor normalization (z-scoring / range-scaling)
   - Per-symbol factor fingerprint

2. **Factor Covariance & Correlations**
   - Cross-factor covariance & correlation matrix
   - Factor volatilities
   - Factor “overlap” detection (which factors are redundant / highly correlated)

3. **PnL Attribution by Factor (Daily Snapshot)**
   - PnL slices per factor
   - Factor contributions per symbol (approximated via hybrid weights)
   - Factor contribution to portfolio PnL

These must be:

- Read-only from a trading perspective (no immediate sizing changes)
- Testable & isolated
- Exposed to the dashboard for visual analysis
- Backwards-compatible with existing state contracts

---

## Files to Touch / Add

- `config/strategy_config.json` (extend)
- `execution/factor_diagnostics.py` (NEW)
- `execution/symbol_score_v6.py` (small, to expose raw factor vector)
- `execution/pnl_attribution.py` (NEW or extend if file exists)
- `execution/state_publish.py` (extend)
- `execution/state_v7.py` (extend loaders)
- `dashboard/intel_panel.py` (extend)
- `dashboard/factor_panel.py` (NEW)
- Tests:
  - `tests/test_factor_vector_extraction.py` (NEW)
  - `tests/test_factor_normalization.py` (NEW)
  - `tests/test_factor_covariance.py` (NEW)
  - `tests/test_factor_pnl_attribution.py` (NEW)
  - `tests/test_state_publish_factor_diagnostics.py` (NEW/extend)

---

## 1. Config Additions

**File:** `config/strategy_config.json`

Add:

```json
"factor_diagnostics": {
  "enabled": true,
  "factors": [
    "trend",
    "carry",
    "rv_momentum",
    "router_quality",
    "expectancy",
    "vol_regime"
  ],
  "normalization_mode": "zscore",        // or "minmax"
  "covariance_lookback_days": 30,
  "pnl_attribution_lookback_days": 14,
  "max_abs_zscore": 3.0
}
````

Notes:

* `factors`: which factor channels to track (must match names exposed by symbol_score_v6).
* `normalization_mode`:

  * `"zscore"`: factor-wise z-scoring across symbols
  * `"minmax"`: [0,1] scaling per factor across symbols
* `covariance_lookback_days`: window over which factor covariance is computed.
* `pnl_attribution_lookback_days`: window for PnL attribution aggregation.

---

## 2. Factor Vector Extraction

We need a consistent factor vector per symbol *at the time hybrid_score is computed*.

**File:** `execution/symbol_score_v6.py`

### 2.1 Dataclass for Factor Vector

Add:

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class FactorVector:
    symbol: str
    factors: Dict[str, float]   # e.g. {"trend": 0.8, "carry": 0.2, ...}
    hybrid_score: float
```

### 2.2 Expose Factor Components

Where hybrid_score is computed (trend + carry + rv + router + expectancy + vol-regime adjustments), make sure each component is *individually available* and then combined.

Something like:

```python
components = {
    "trend": trend_component,
    "carry": carry_component,
    "rv_momentum": rv_component,
    "router_quality": router_component,
    "expectancy": expectancy_component,
    "vol_regime": vol_regime_component,
}
hybrid = sum(components.values())
# ... apply decay, regimes, clamps etc.
```

Add an accessor:

```python
def build_factor_vector(symbol: str, components: Dict[str, float], hybrid: float) -> FactorVector:
    return FactorVector(symbol=symbol, factors=components, hybrid_score=hybrid)
```

The diagnostics engine will consume these FactorVectors.

---

## 3. Factor Diagnostics Engine

**File:** `execution/factor_diagnostics.py` (NEW)

### 3.1 Dataclasses

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class NormalizedFactorVector:
    symbol: str
    factors: Dict[str, float]   # normalized values

@dataclass
class FactorCovarianceSnapshot:
    factors: List[str]
    covariance: np.ndarray      # shape (F, F)
    correlation: np.ndarray     # shape (F, F)
    factor_vols: Dict[str, float]

@dataclass
class FactorDiagnosticsSnapshot:
    per_symbol: Dict[str, NormalizedFactorVector]
    covariance: FactorCovarianceSnapshot
```

### 3.2 Normalization

Implement:

```python
def normalize_factor_vectors(
    vectors: List[FactorVector],
    factor_names: List[str],
    mode: str,
    max_abs_zscore: float
) -> List[NormalizedFactorVector]:
    """
    Normalizes factor values per factor across symbols.
    mode: "zscore" or "minmax"
    """
```

Behavior:

* For each factor in `factor_names`:

  * Collect its values across all symbols.
  * Compute:

    * z-score: (x - mean) / std; clamp to ±max_abs_zscore.
    * minmax: (x - min) / (max - min); rescale to [0, 1] or [-1, 1] depending on taste.
* If std == 0 or max == min: map all to 0 (for zscore) or 0.0 (for minmax) for that factor.

### 3.3 Covariance & Correlation

Implement:

```python
def compute_factor_covariance(
    vectors: List[FactorVector],
    factor_names: List[str]
) -> FactorCovarianceSnapshot:
    """
    Builds factor covariance and correlation matrices across symbols.
    """
```

Steps:

1. Build matrix `X` of shape (N_symbols, F) where columns are factor values.
2. Center each column (subtract mean).
3. Covariance: `cov = (Xᵀ X) / (N_symbols - 1)` (or np.cov).
4. Vols: `factor_vols[name] = sqrt(cov[i, i])`.
5. Correlation: `corr[i, j] = cov[i,j] / (vol_i * vol_j)`, handle zero vols gracefully (set corr=0).

Return FactorCovarianceSnapshot.

### 3.4 Diagnostic Snapshot Builder

Implement:

```python
def build_factor_diagnostics_snapshot(
    factor_vectors: List[FactorVector],
    cfg: FactorDiagnosticsConfig
) -> FactorDiagnosticsSnapshot:
    """
    Produces normalized per-symbol factor vectors + covariance snapshot.
    """
```

Use `cfg.factors` as the factor list.

---

## 4. PnL Attribution by Factor

We’ll approximate factor PnL contribution using:

* Per-trade hybrid breakdown by weight share of each factor component.
* Daily PnL aggregated by factor share.

**File:** `execution/pnl_attribution.py` (NEW or extend existing attribution module)

### 4.1 Dataclasses

```python
@dataclass
class FactorPnlSlice:
    factor: str
    pnl_usd: float

@dataclass
class FactorPnlSnapshot:
    by_factor: Dict[str, float]        # factor -> pnl_usd over window
    total_pnl_usd: float
    window_days: int
```

### 4.2 Approximate Attribution Logic

Assume each trade record includes:

* `hybrid_score_at_entry`
* Factor vector at entry: e.g. `{"trend": 0.8, "carry": 0.2, ...}` (if not currently stored, we can approximate using normalized factors at entry time or drop to a simpler form).

For C2 v1, we keep it simple and approximate as:

For each closed trade in the window:

1. Let `pnl = trade.realized_pnl_usd`.

2. Let `factor_components = trade.factor_components` or reconstruct from logs if present.

3. Compute weights:

   ```python
   total_abs = sum(abs(v) for v in factor_components.values()) or 1.0
   weight[f] = abs(factor_components[f]) / total_abs
   ```

4. Allocate PnL:

   ```python
   for f in factors:
       factor_pnl[f] += pnl * weight[f]
   ```

Aggregate across trades in the lookback window.

Implement:

```python
def compute_factor_pnl_snapshot(
    trades: List[TradeRecord],
    factor_names: List[str],
    window_days: int
) -> FactorPnlSnapshot:
    ...
```

We can later refine this, but v1 is good enough for “which factor is driving PnL”.

---

## 5. State Publishing & Loading

**File:** `execution/state_publish.py`

### 5.1 Factor Diagnostics State

Add block to intel/diagnostics state (e.g. `intel.json` or `factor_state.json` if you prefer a new file):

```python
def write_factor_diagnostics_state(snapshot: FactorDiagnosticsSnapshot, state: dict) -> None:
    state["factor_diagnostics"] = {
        "per_symbol": {
            vec.symbol: vec.factors
            for vec in snapshot.per_symbol.values()
        },
        "covariance": {
            "factors": snapshot.covariance.factors,
            "covariance_matrix": snapshot.covariance.covariance.tolist(),
            "correlation_matrix": snapshot.covariance.correlation.tolist(),
            "factor_vols": snapshot.covariance.factor_vols,
        }
    }
```

### 5.2 Factor PnL Attribution State

```python
def write_factor_pnl_state(pnl_snapshot: FactorPnlSnapshot, state: dict) -> None:
    state["factor_pnl"] = {
        "by_factor": pnl_snapshot.by_factor,
        "total_pnl_usd": pnl_snapshot.total_pnl_usd,
        "window_days": pnl_snapshot.window_days,
    }
```

Ensure these are invoked from the standard intel/diagnostics publishing flows.

### 5.3 Loaders

**File:** `execution/state_v7.py`

Add:

```python
def load_factor_diagnostics_state() -> dict: ...
def load_factor_pnl_state() -> dict: ...
```

Return `{}` if missing.

---

## 6. Dashboard Factor Panel

**File:** `dashboard/factor_panel.py` (NEW)

### 6.1 Per-Symbol Factor Fingerprint

Show a table:

* Symbol
* Hybrid Score
* Trend factor
* Carry factor
* RV-MOMO factor
* Router factor
* Expectancy factor
* Vol-Regime factor

Optional: radar / spider chart per symbol if your dashboard lib supports it (or stacked bar).

### 6.2 Factor Covariance Matrix

* Render a **heatmap** of the factor correlation matrix.
* Show per-factor vol in a sidebar.

Interpretation:

* Corr near 1.0 → highly redundant factors.
* Corr near 0.0 → independent.
* Negative corr → hedging factors.

### 6.3 Factor PnL Attribution

Table:

| Factor         | PnL (USD) | % of Total |
| -------------- | --------- | ---------- |
| trend          | x         | y%         |
| carry          | x         | y%         |
| rv_momentum    | x         | y%         |
| router_quality | x         | y%         |
| expectancy     | x         | y%         |
| vol_regime     | x         | y%         |

Color PnL positive in green, negative in red.

---

## 7. Tests

### 7.1 `tests/test_factor_vector_extraction.py`

* Ensure symbol_score_v6 exposes consistent factor vectors:

  * All configured factors present.
  * Missing factors default to 0.0.
* Hybrid score equals sum of components (before external modifiers) where expected.

### 7.2 `tests/test_factor_normalization.py`

* z-score mode:

  * mean ~ 0, std ~ 1 for a simple synthetic case.
  * clamp to ±max_abs_zscore enforced.
* minmax mode:

  * factors scaled into [0,1] or [-1,1] consistently.

### 7.3 `tests/test_factor_covariance.py`

* Synthetic data with known relationships:

  * e.g. factor B = 2 × factor A → correlation ≈ 1.
  * factor C independent noise → corr ≈ 0.
* Vols computed from diagonal of covariance.

### 7.4 `tests/test_factor_pnl_attribution.py`

* Simple trades with known factor weights and PnL.
* Verify that factor-level PnL sums to total PnL (within floating-point tolerance).
* Check that factors with zero weight receive zero PnL.

### 7.5 `tests/test_state_publish_factor_diagnostics.py`

* State objects structured with expected keys.
* Dashboard loaders consume state without errors.

All new tests must pass; existing test suite remains green.

---

## 8. Acceptance Criteria

The patchset is complete when:

1. symbol_score_v6 exposes factor vectors used in hybrid_score composition.
2. factor_diagnostics module:

   * Normalizes factor vectors.
   * Computes factor covariance & correlation matrix.
   * Produces FactorDiagnosticsSnapshot.
3. pnl_attribution module:

   * Computes FactorPnlSnapshot from a set of trades.
4. state_publish:

   * Writes factor_diagnostics and factor_pnl blocks.
5. state_v7:

   * Loads factor_diagnostics and factor_pnl for dashboard.
6. Dashboard:

   * Displays per-symbol factor fingerprints.
   * Shows factor correlation matrix and vols.
   * Shows factor PnL attribution.
7. All new tests (factor vector, normalization, covariance, PnL attribution, state publishing) pass, and overall test suite remains green.

This patchset must **not** alter trade decisions, sizing, or risk veto logic.
It is purely analytical and diagnostic, preparing for future v7.6 factor-weight optimization.

```
