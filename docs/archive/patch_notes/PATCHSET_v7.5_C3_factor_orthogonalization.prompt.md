# PATCHSET v7.5_C3 — Factor Orthogonalization & Auto-Weighting

## Objective

Upgrade GPT-Hedge v7.5 from a “multi-factor engine” to an **optimized multi-factor model** by:

1. **Orthogonalizing factors** (remove redundancy and overlap).
2. **Auto-weighting factors** based on:
   - factor volatility (risk),
   - factor IR/Sharpe (performance),
   - covariance structure.
3. **Feeding optimized factor weights into hybrid_score** in a **configurable, safe** manner.

This patch must:
- Be *backward compatible* (can be disabled via config).
- Not break existing factor diagnostics / PnL attribution.
- Preserve all current risk vetoes and execution logic.

---

## Files to Touch / Add

- `config/strategy_config.json` (extend)
- `execution/factor_diagnostics.py` (extend with orthogonalization & weights)
- `execution/symbol_score_v6.py` (extend hybrid_score to use factor weights)
- `execution/state_publish.py` (extend factor_diagnostics publishing)
- `execution/state_v7.py` (extend loaders)
- `dashboard/factor_panel.py` (extend UI)
- Tests:
  - `tests/test_factor_orthogonalization.py` (NEW)
  - `tests/test_factor_auto_weighting.py` (NEW)
  - `tests/test_hybrid_score_factor_weights.py` (NEW)
  - `tests/test_state_publish_factor_weights.py` (NEW/extend)

---

## 1. Config Additions

**File:** `config/strategy_config.json`

Add under existing `factor_diagnostics` block:

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
  "normalization_mode": "zscore",
  "covariance_lookback_days": 30,
  "pnl_attribution_lookback_days": 14,
  "max_abs_zscore": 3.0,

  "orthogonalization": {
    "enabled": true,
    "method": "gram_schmidt"    // or "none"
  },
  "auto_weighting": {
    "enabled": true,
    "mode": "vol_inverse_ir",   // or "equal", "vol_inverse", "ir_only"
    "min_weight": 0.05,
    "max_weight": 0.40,
    "normalize_to_one": true,
    "smoothing_alpha": 0.2      // EWMA smoothing for weights across days
  }
}
````

Interpretation:

* `orthogonalization.enabled`: toggles orthogonalization; if false, keep raw factors.
* `auto_weighting.mode`:

  * `"equal"`: all factors same weight.
  * `"vol_inverse"`: weights ∝ 1 / factor_vol.
  * `"ir_only"`: weights ∝ factor_IR (Sharpe-like).
  * `"vol_inverse_ir"`: weights ∝ (factor_IR / factor_vol), our recommended default.
* `min_weight` / `max_weight`: per-factor clamps.
* `normalize_to_one`: whether weights sum to 1.
* `smoothing_alpha`: how quickly factor weights adapt over time.

---

## 2. Factor Orthogonalization

**File:** `execution/factor_diagnostics.py`

We already compute covariance & normalized vectors.

### 2.1 New Dataclasses

Add:

```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class FactorWeights:
    weights: Dict[str, float]    # factor_name -> weight

@dataclass
class OrthogonalizedFactorVectors:
    per_symbol: Dict[str, Dict[str, float]]  # symbol -> factor_name -> ortho_value
```

### 2.2 Gram-Schmidt Orthogonalization (Factors Across Symbols)

We want to orthogonalize factors across symbols. Think of:

* For each factor `f`, we have a column vector `X_f` over symbols.
* Combine all into matrix `X` of shape (N_symbols, F).
* Apply Gram-Schmidt on columns to produce orthogonal columns `Q`.

Implement:

```python
def orthogonalize_factors(
    factor_vectors: List[FactorVector],
    factor_names: List[str]
) -> OrthogonalizedFactorVectors:
    """
    Applies Gram-Schmidt to the factor columns across symbols, returning
    orthogonalized factor values per symbol and factor.
    """
```

Steps:

1. Build matrix `X` with shape (N_symbols, F):

   * Row i: symbol_i
   * Column j: factor_j value from FactorVector.factors

2. Apply Gram-Schmidt on columns:

   ```python
   # pseudo
   Q = []
   for j in range(F):
       v = X[:, j].copy()
       for k in range(j):
           proj = np.dot(v, Q[k]) / np.dot(Q[k], Q[k]) * Q[k]
           v = v - proj
       Q.append(v)
   ```

3. Stack Q to shape (N_symbols, F). Handle degenerate factors (norm == 0) by leaving column as zeros.

4. Map back to per-symbol dict:

   ```python
   result = {}
   for i, symbol in enumerate(symbols):
       result[symbol] = {}
       for j, factor_name in enumerate(factor_names):
           result[symbol][factor_name] = Q_matrix[i, j]
   ```

This gives orthogonalized factor values per symbol.

### 2.3 Integration into Diagnostics Snapshot

Extend `build_factor_diagnostics_snapshot`:

* After computing covariance:

  * If `cfg.orthogonalization.enabled`:

    * Compute `ortho_vectors = orthogonalize_factors(...)`.
    * Optionally recompute covariance on orthogonalized values (for debug / inspection).
  * Else:

    * `ortho_vectors = {symbol: factors}` (identity mapping).

* Provide both:

  ```python
  @dataclass
  class FactorDiagnosticsSnapshot:
      per_symbol: Dict[str, NormalizedFactorVector]
      covariance: FactorCovarianceSnapshot
      orthogonalized: OrthogonalizedFactorVectors
  ```

Update state publishing accordingly.

---

## 3. Factor Auto-Weighting

**File:** `execution/factor_diagnostics.py`

We now compute factor weights from:

* Factor vol
* Factor PnL attribution

We already have:

* `FactorCovarianceSnapshot.factor_vols`
* Factor PnL per factor from `factor_pnl_attribution.py` / state.

### 3.1 Compute Factor IR / Sharpe-Like Metric

Implement:

```python
def compute_factor_ir(
    factor_pnl: Dict[str, float],
    factor_vols: Dict[str, float],
    eps: float = 1e-9
) -> Dict[str, float]:
    """
    Computes per-factor 'information ratio':
    IR_f = factor_pnl[f] / (factor_vols[f] + eps)
    """
```

We can use absolute or signed pnl; for v1, use **signed** IR to preserve sign.

### 3.2 Compute Raw Weights

Implement:

```python
def compute_raw_factor_weights(
    mode: str,
    factor_names: List[str],
    factor_vols: Dict[str, float],
    factor_ir: Dict[str, float],
    eps: float = 1e-9
) -> Dict[str, float]:
    """
    mode: 'equal', 'vol_inverse', 'ir_only', 'vol_inverse_ir'
    Returns unnormalized weights (may be negative if IR negative).
    """
```

Modes:

* `"equal"`:

  * `w_f = 1.0`
* `"vol_inverse"`:

  * `w_f = 1.0 / (factor_vols[f] + eps)`
* `"ir_only"`:

  * `w_f = factor_ir[f]`
* `"vol_inverse_ir"`:

  * `w_f = factor_ir[f] / (factor_vols[f] + eps)`

### 3.3 Normalize & Clamp Weights

Implement:

```python
def normalize_factor_weights(
    raw_weights: Dict[str, float],
    min_weight: float,
    max_weight: float,
    normalize_to_one: bool
) -> FactorWeights:
    """
    Applies abs-normalization and clamping:
    - take absolute values
    - normalize sum to 1 if requested
    - clamp each weight to [min_weight, max_weight]
    """
```

Steps:

1. Convert `raw` → `abs_raw = {f: abs(w)}`.
2. If all zeros: assign equal weights.
3. Normalize: `w_f = abs_raw[f] / sum(abs_raw.values())` if `normalize_to_one=True`.
4. Clamp each `w_f` to `[min_weight, max_weight]`.
5. Renormalize again if `normalize_to_one=True` (optional but nicer).

### 3.4 EWMA Smoothing for Stability

We want weights to evolve slowly.

Implement:

```python
def smooth_factor_weights(
    prev: Optional[FactorWeights],
    current: FactorWeights,
    alpha: float
) -> FactorWeights:
    """
    EWMA smoothing of weights:
    w_new = alpha * current + (1 - alpha) * prev
    """
```

If `prev is None`: return `current`.

This function will be called using previously persisted weights (from state) to smooth across days.

### 3.5 Build Final Weights Snapshot

Add helper:

```python
@dataclass
class FactorWeightsSnapshot:
    weights: Dict[str, float]

def build_factor_weights_snapshot(
    cfg: FactorDiagnosticsConfig,
    factor_cov: FactorCovarianceSnapshot,
    factor_pnl: Dict[str, float],
    prev_weights: Optional[FactorWeights]
) -> FactorWeightsSnapshot:
    """
    Computes final per-factor weights with:
    - IR / vol-based raw weights
    - normalization & clamping
    - EWMA smoothing
    """
```

---

## 4. Feeding Factor Weights into Hybrid Score

**File:** `execution/symbol_score_v6.py`

We’ve already got:

* `FactorVector` per symbol with factor components
* `FactorDiagnosticsSnapshot` from C2

Now we want to:

1. Replace the “flat” combination for configured factors with:

   [
   \text{hybrid_raw}(symbol) = \sum_f w_f \cdot \text{factor_value}_f(symbol)
   ]

2. Then still apply:

   * decay
   * regimes
   * clamps

### 4.1 Config for Hybrid Weight Usage

Add:

```python
@dataclass
class FactorWeightHybridConfig:
    enabled: bool
    use_orthogonalized: bool    # if True, use orthogonalized factor values
```

Derived from `factor_diagnostics.auto_weighting.enabled` and `orthogonalization.enabled`.

### 4.2 Hybrid Composition Logic

Where we previously did:

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
```

We now do:

```python
if factor_weight_cfg.enabled and factor_weights is not None:
    # choose raw or orthogonalized factors
    factor_values = get_factor_values_for_symbol(symbol, factor_vectors, ortho_vectors, factor_weight_cfg)
    hybrid_raw = 0.0
    for f_name, f_value in factor_values.items():
        w = factor_weights.weights.get(f_name, 0.0)
        hybrid_raw += w * f_value
else:
    # fallback: legacy sum of components
    hybrid_raw = sum(components.values())
```

Then:

* Apply router quality modulation (existing v7.5_B2 logic).
* Apply alpha decay.
* Apply vol regime modifiers.
* Clamp final `hybrid` to [-1, 1].

Important: **do not change downstream risk or sizing logic**.

---

## 5. State Publishing

**File:** `execution/state_publish.py`

Add factor weights to diagnostics state:

```python
def write_factor_weights_state(weights_snapshot: FactorWeightsSnapshot, state: dict) -> None:
    state["factor_weights"] = weights_snapshot.weights
```

Update `compute_and_write_factor_diagnostics_state()` to:

* include `factor_weights` in the state.
* store whether orthogonalization is active.

**File:** `execution/state_v7.py`

Add:

```python
def load_factor_weights_state() -> Dict[str, float]:
    """
    Returns factor_weights from diagnostics state or {} if missing.
    """
```

---

## 6. Dashboard: Factor Weights & Orthogonalization

**File:** `dashboard/factor_panel.py`

Extend:

### 6.1 Factor Weights Table

Columns:

* Factor
* Weight
* Vol (from factor_vols)
* PnL (from factor_pnl)
* IR (from factor_pnl / vol)

Visual:

* Bar chart of weights.
* Ensure sum ~1 when `normalize_to_one=True`.

### 6.2 Orthogonalization Indicator

Add a small indicator:

* Show whether **orthogonalization is enabled**.
* Optionally show correlation matrix before and after orthogonalization (if you publish both).

This makes it clear to the human (and investors) that factors are not double-counted.

---

## 7. Tests

### 7.1 `tests/test_factor_orthogonalization.py`

* Synthetic example where:

  * Factor B = Factor A (perfectly correlated).
  * Factor C = independent noise.

* After orthogonalization:

  * A remains as-is.
  * B is nearly zero (or orthogonal residual).
  * C remains independent.

* Verify that dot-products between orthogonalized columns are ~0 (within tolerance).

### 7.2 `tests/test_factor_auto_weighting.py`

* Known factor_vol & factor_pnl values.
* Check:

  * `"equal"` mode → equal weights.
  * `"vol_inverse"` → lower vol → higher weight.
  * `"ir_only"` → higher IR → higher weight.
  * `"vol_inverse_ir"` → combination (IR/vol).
* Check min_weight / max_weight clamps.
* Test EWMA smoothing moves weights gradually toward new weights.

### 7.3 `tests/test_hybrid_score_factor_weights.py`

* With auto_weighting disabled → hybrid same as legacy sum.
* With enabled:

  * using simple factor vectors and weights → hybrid equals Σ w_f * factor_f.
* Ensure final hybrid respects [-1, 1] and still passes through decay / regime logic.

### 7.4 `tests/test_state_publish_factor_weights.py`

* Factor diagnostics state contains `factor_weights`.
* Dashboard loaders can consume `factor_weights` without error.

All new tests must pass; overall suite remains green.

---

## 8. Acceptance Criteria

The patchset is complete when:

1. `factor_diagnostics` config has `orthogonalization` and `auto_weighting` sections and they are respected.
2. `orthogonalize_factors()` produces orthogonal factor columns across symbols.
3. `compute_factor_ir()` and `compute_raw_factor_weights()` produce intuitive weights given synthetic factor PnL and vol inputs.
4. `normalize_factor_weights()` and `smooth_factor_weights()` yield stable, bounded weights.
5. `symbol_score_v6.hybrid_score()`:

   * Uses Σ (w_f × factor_f(symbol)) when auto_weighting is enabled.
   * Falls back to legacy behaviour when disabled or weights missing.
6. `state_publish` and `state_v7`:

   * Expose `factor_weights` and (optionally) orthogonalization status in diagnostics state.
7. The dashboard factor panel:

   * Shows factor weights + IR + vol.
   * Indicates whether orthogonalization is active.
8. All new tests (orthogonalization, auto-weighting, hybrid integration, state publish) pass and the overall test suite remains green.

This patch **must not change** risk veto logic, position sizing contracts, or execution policies.
It only changes **how hybrid alpha is composed**, in a controlled and reversible way.

```
