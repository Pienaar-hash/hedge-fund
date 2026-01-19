# PATCHSET v7.4_B2 — Volatility Regime Model (EWMA 7/30) + Sizing & Hybrid Modulation

## Objective

Turn the informal “regime modulation” (already used to downweight carry in high/crisis regimes) into a **first-class, explicit volatility regime model** that:

- Computes **realized volatility** using EWMA over short and long horizons (e.g. 7 vs 30 days).
- Classifies each symbol (or base asset) into one of:  
  `low`, `normal`, `high`, `crisis`.
- Modulates:
  - **Position sizing** (`per_trade_nav_pct` multiplier).
  - **Hybrid score weighting** (e.g. downweight carry / expectancy in crisis).
- Publishes a **regime snapshot** to state and surfaces it in the dashboard.

This must **not** break A1/A2/B1 and should plug into the existing hybrid/regime hooks you just added.

---

## Files to Touch

- `config/strategy_config.json`
- `execution/utils/vol.py` (or create if not present)
- `execution/intel/symbol_score_v6.py` (or wherever scores are assembled)
- `execution/signal_screener.py`
- `execution/state_publish.py`
- `execution/state_v7.py`
- `dashboard/intel_panel.py` and/or `dashboard/overview_panel.py`
- Tests:
  - `tests/test_vol_regimes.py` (new)
  - `tests/test_signal_screener_vol_regimes.py` (new)
  - `tests/test_state_publish_vol_regimes.py` (new or extend)

---

## 1. Config: Vol Regime Parameters

**File:** `config/strategy_config.json`

### Add a `vol_regimes` block

Example:

```json
"vol_regimes": {
  "defaults": {
    "short_window_bars": 168,
    "long_window_bars": 720,
    "short_halflife_bars": 84,
    "long_halflife_bars": 360,
    "ratio_low_high": {
      "low": 0.6,
      "normal": 1.2,
      "high": 1.8
    }
  },
  "sizing_multipliers": {
    "CORE": {
      "low": 1.15,
      "normal": 1.0,
      "high": 0.75,
      "crisis": 0.5
    },
    "SATELLITE": {
      "low": 1.1,
      "normal": 1.0,
      "high": 0.7,
      "crisis": 0.4
    },
    "TACTICAL": {
      "low": 1.0,
      "normal": 0.9,
      "high": 0.6,
      "crisis": 0.3
    },
    "ALT-EXT": {
      "low": 0.9,
      "normal": 0.8,
      "high": 0.5,
      "crisis": 0.2
    }
  },
  "hybrid_weight_modifiers": {
    "default": {
      "low":   { "carry": 1.0,  "expectancy": 1.0,  "router": 1.0  },
      "normal":{ "carry": 1.0,  "expectancy": 1.0,  "router": 1.0  },
      "high":  { "carry": 0.7,  "expectancy": 0.9,  "router": 1.0  },
      "crisis":{ "carry": 0.4,  "expectancy": 0.8,  "router": 1.1  }
    }
  }
}
````

Rules:

* `short_window_bars` / `long_window_bars`: number of bars used to compute vol (e.g., on the screener timeframe).
* `short_halflife_bars` / `long_halflife_bars`: EWMA halflives.
* `ratio_low_high` defines boundaries for `short_vol / long_vol`:

  * If ratio < `low` → `low` regime.
  * If between `low` and `normal` → `normal`.
  * If between `normal` and `high` → `high`.
  * If > `high` → `crisis`.
* `sizing_multipliers` are tier-specific; if tier missing, fallback to a CORE-like default.
* `hybrid_weight_modifiers` tweak B1’s hybrid component weights by regime (you already have regime modulation; this formalizes it).

---

## 2. Vol Utilities: EWMA & Regime Classification

**File:** `execution/utils/vol.py` (create if missing or extend if exists)

### 2.1 EWMA Realized Vol

Add:

```python
from dataclasses import dataclass
from typing import Literal, Sequence
import numpy as np

@dataclass
class VolInputs:
    returns: np.ndarray  # 1D array of log returns

@dataclass
class VolRegime:
    label: Literal["low", "normal", "high", "crisis"]
    vol_short: float
    vol_long: float
    ratio: float
```

Helper functions:

```python
def compute_ewma_vol(
    returns: np.ndarray,
    halflife_bars: int
) -> float:
    """
    Compute EWMA volatility (std dev) of log returns.
    returns: 1D numpy array.
    """
```

Implementation:

* Use standard EWMA with decay `lambda = 0.5 ** (1 / halflife_bars)`.
* Weighted variance → sqrt → annualise or keep at bar-level (be consistent; bar-level is fine as long as thresholds are tuned accordingly).

```python
def classify_vol_regime(
    vol_inputs: VolInputs,
    cfg: VolRegimeConfig
) -> VolRegime:
    """
    Use short vs long EWMA vol to classify into low/normal/high/crisis.
    """
```

Where `VolRegimeConfig` is a small config dataclass built from `strategy_config["vol_regimes"]["defaults"]`.

Logic:

1. Compute `vol_short` using `short_halflife_bars` and last `short_window_bars` returns.
2. Compute `vol_long` using `long_halflife_bars` and last `long_window_bars` returns.
3. If `vol_long <= 0` or insufficient data → default regime `"normal"` with zeros.
4. `ratio = vol_short / vol_long`.
5. Classify:

```python
if ratio < cfg.ratio_low:
    label = "low"
elif ratio < cfg.ratio_normal:
    label = "normal"
elif ratio < cfg.ratio_high:
    label = "high"
else:
    label = "crisis"
```

Return `VolRegime(label, vol_short, vol_long, ratio)`.

---

## 3. Integration: Symbol Scores & Screener

### 3.1 Symbol Scoring: attach VolRegime

**File:** `execution/intel/symbol_score_v6.py`

Extend `SymbolScore` dataclass to include:

```python
@dataclass
class SymbolScore:
    ...
    vol_regime: str  # "low" | "normal" | "high" | "crisis"
    vol_short: float
    vol_long: float
    vol_ratio: float
```

In the scoring pipeline:

* Fetch price history (or returns) for the symbol over the needed window.
* Build `VolInputs` and call `classify_vol_regime(...)`.
* Attach `vol_regime.label`, `vol_short`, `vol_long`, `ratio` to the score.

If history missing:

* Default to `normal`, vols 0, ratio 1, and mark clearly in logs if desired.

### 3.2 Hybrid Weight Modulation

You indicated B1 already has **regime modulation**. Now we make it explicit:

* Before combining trend/carry/expectancy/router into final hybrid score:

  * Read `hybrid_weight_modifiers` for current regime (and optionally tier).
  * Multiply the **base weights** from B1 (e.g. trend 0.40, carry 0.25, expectancy 0.20, router 0.15) by regime-specific multipliers.

Pseudo:

```python
base = HybridBaseWeights(trend=0.4, carry=0.25, expectancy=0.2, router=0.15)
mods = get_hybrid_modifiers_for_regime(regime_label)  # from config

w_trend     = base.trend
w_carry     = base.carry * mods.carry
w_expect    = base.expectancy * mods.expectancy
w_router    = base.router * mods.router

# normalise optionally:
total = w_trend + w_carry + w_expect + w_router
if total > 0:
    w_trend  /= total
    w_carry  /= total
    w_expect /= total
    w_router /= total

hybrid_score = (
    w_trend * trend_score
    + w_carry * carry_score
    + w_expect * expectancy_score
    + w_router * router_quality_score
)
```

If you already have a `HybridScoreConfig`, extend it with regime-aware modifiers instead of re-inventing.

### 3.3 Sizing Modulation in Screener

**File:** `execution/signal_screener.py`

You already compute a base `per_trade_nav_pct` and hybrid score. Now:

* For each candidate, read:

  * Tier (CORE/SAT/TACTICAL/ALT-EXT).
  * `vol_regime` from `SymbolScore`.
* Look up `sizing_multipliers[tier][vol_regime]` from config.
* Apply:

```python
effective_per_trade_nav_pct = base_per_trade_nav_pct * sizing_multiplier
```

Before:

* Risk limits enforcement (`risk_limits.check_order`) remains unchanged.
* All existing caps (per-symbol, portfolio, group) still apply.

If config missing:

* Default `sizing_multiplier = 1.0`.

---

## 4. State & Dashboard

### 4.1 State Publish

**File:** `execution/state_publish.py`

In the intel or symbol state section (where you already publish hybrid and carry scores):

Add, for each symbol:

```json
"vol_regime": "high",
"vol": {
  "short": 0.0123,
  "long": 0.0085,
  "ratio": 1.45
}
```

This should re-use the `SymbolScore` values; **do not recompute** vol here.

Also consider adding a summary in `risk.json` or `intel.json`:

```json
"vol_regime_summary": {
  "low": 3,
  "normal": 7,
  "high": 2,
  "crisis": 0
}
```

(Counts of symbols per regime, optional but useful.)

### 4.2 State Loader

**File:** `execution/state_v7.py`

Add helpers:

* `load_vol_regime_snapshot()`
* `load_symbol_vol_regimes()`

or extend existing intel loaders to include:

* `vol_regime`, `vol_short`, `vol_long`, `vol_ratio`.

### 4.3 Dashboard

**File(s):** `dashboard/intel_panel.py`, maybe `dashboard/overview_panel.py`, and `dashboard/app.py`

1. **Intel Panel:**

   * Add a **Volatility Regime** column or small badge next to each symbol:

     * “L”, “N”, “H”, “C” with colour coding:

       * L: blue
       * N: grey/green
       * H: orange
       * C: red

   * Optionally show `short / long` vol in a tooltip.

2. **Overview Panel:**

   * Add a small bar/chart showing number of symbols in each regime, using `vol_regime_summary`.

3. **Advanced Tab:**

   * Since you already wired hybrid + carry panels in B1, add a small “Vol Regimes” block in the Advanced tab:

     * Table or chips with regime counts and maybe average volatility.

All UI changes should be additive and safe with missing fields (render “—” when absent).

---

## 5. Tests

### 5.1 Vol Utilities

**File:** `tests/test_vol_regimes.py` (new)

Scenarios:

1. **Stable low-vol series**:

   * returns with small variance.
   * Expect `vol_short ≈ vol_long`, `ratio ≈ 1`, classification `"normal"` (based on thresholds).

2. **Sudden spike (crisis)**:

   * returns where last segment has large variance.
   * Expect `ratio` > `high` threshold, classification `"crisis"`.

3. **Low regime**:

   * synthetic where short vol significantly lower than long vol (vol has been compressing).
   * classification `"low"`.

4. **Insufficient data**:

   * fewer points than `long_window_bars`.
   * classification `"normal"` with zeros, or a well-defined fallback.

### 5.2 Screener Integration

**File:** `tests/test_signal_screener_vol_regimes.py` (new)

Scenarios:

1. **CORE sizing & regimes**:

   * Base `per_trade_nav_pct = 0.02`.
   * For vol regimes `low`, `normal`, `high`, `crisis` with CORE multipliers 1.15, 1.0, 0.75, 0.5:

     * Confirm effective size is correctly scaled.

2. **Different tiers**:

   * Use SATELLITE and TACTICAL with different multipliers; ensure mapping is tier-aware.

3. **Missing config**:

   * When `vol_regimes` block absent or incomplete, fallback to 1.0.

4. **Hybrid weight modulation**:

   * Build synthetic scores (trend, carry, expectancy, router) with regime-specific modifiers.
   * Assert that hybrid score changes according to regime (e.g., carry weight reduced in `crisis` vs `normal`).

### 5.3 State & Dashboard

**File:** `tests/test_state_publish_vol_regimes.py` (new or extend existing)

* Assert that:

  * `vol_regime`, `vol.short`, `vol.long`, `vol.ratio` appear for a known symbol in intel state.
  * `vol_regime_summary` matches constructed sample.

You may not have dashboard tests yet; if you do, extend them minimally to verify the panel uses the new fields without crashing.

All existing tests (A1, A2, B1) must remain green.

---

## 6. Acceptance Criteria

The patch is complete when:

1. `strategy_config.json` contains a `vol_regimes` block with defaults, sizing multipliers, and hybrid weight modifiers.
2. `execution/utils/vol.py` can compute EWMA vol and classify each symbol into a regime.
3. `symbol_score_v6` attaches regime information (`vol_regime`, `vol_short`, `vol_long`, `vol_ratio`) to symbol scores.
4. `signal_screener`:

   * Applies tier + regime-based sizing multipliers to base `per_trade_nav_pct`.
   * Uses regime-aware hybrid weight modulation (trend/carry/expectancy/router).
5. `state_publish`:

   * Exposes regime info per symbol in intel state.
   * Optionally publishes `vol_regime_summary`.
6. Dashboard:

   * Displays per-symbol regime badges.
   * Shows some aggregate regime summary (even minimal).
7. New tests pass:

   * `test_vol_regimes.py`
   * `test_signal_screener_vol_regimes.py`
   * `test_state_publish_vol_regimes.py`
8. All A1/A2/B1 tests still pass, with no runtime regression.

Do not change any other behaviour or contracts beyond what is described here.

```
