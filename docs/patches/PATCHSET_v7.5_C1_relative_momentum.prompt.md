# PATCHSET v7.5_C1 — Cross-Asset Relative Momentum (RV-MOMO) Factor

## Objective

Add a **relative momentum / cross-asset strength factor** to the alpha stack so that:

- Capital prefers **strong coins vs weak coins**, not just “good setups in isolation”.
- Hybrid score is informed by:
  - BTC vs ETH strength
  - L1s vs ALTs
  - Memecoins vs rest
- The factor is:
  - Configurable
  - Purely additive (no breaking changes)
  - Integrated into state + dashboard + screener

End goal:

> `hybrid_score = trend + carry + expectancy + router + relative_momentum (+ decay, regimes, tiers)`

---

## Files to Touch / Add

- `config/strategy_config.json` (extend)
- `config/rv_momo_baskets.json` (NEW)
- `execution/rv_momentum.py` (NEW)
- `execution/symbol_score_v6.py` (extend)
- `execution/state_publish.py` (extend)
- `execution/state_v7.py` (extend loader)
- `execution/signal_screener.py` (extend optional filtering/sorting)
- `dashboard/intel_panel.py` (extend)
- Tests:
  - `tests/test_rv_momentum_baskets.py` (NEW)
  - `tests/test_rv_momentum_factor.py` (NEW)
  - `tests/test_symbol_score_rv_momentum.py` (NEW)
  - `tests/test_state_publish_rv_momentum.py` (NEW/extend)
  - `tests/test_signal_screener_rv_momentum.py` (NEW)

---

## 1. Config Additions

### 1.1 Strategy Config Block

**File:** `config/strategy_config.json`

Add:

```json
"rv_momentum": {
  "enabled": true,
  "lookback_bars": 48,
  "half_life_bars": 24,
  "normalize_mode": "zscore",     // or "rank"
  "btc_vs_eth_weight": 0.35,
  "l1_vs_alt_weight": 0.35,
  "meme_vs_rest_weight": 0.20,
  "per_symbol_weight": 0.10,
  "max_abs_score": 1.0
}
````

Notes:

* `lookback_bars`: on the signal timeframe (e.g. 1h) for relative momentum.
* `half_life_bars`: for EWMA smoothing of relative returns.
* `normalize_mode`:

  * `"zscore"`: convert raw relative returns into z-scores within each basket.
  * `"rank"`: assign scores based on rank within basket.
* `*_weight`: factor weights in combined rv_score.
* `per_symbol_weight`: optional direct symbol momentum contribution.
* `max_abs_score`: clamp for final rv_score ∈ [-1, 1].

### 1.2 Basket Definitions

**File:** `config/rv_momo_baskets.json` (NEW)

```json
{
  "pairs": {
    "btc_vs_eth": {
      "long": "BTCUSDT",
      "short": "ETHUSDT"
    }
  },
  "baskets": {
    "l1": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "alts": ["LTCUSDT", "LINKUSDT", "SUIUSDT"],
    "meme": ["DOGEUSDT", "WIFUSDT"]
  }
}
```

Interpretation:

* `btc_vs_eth`: simple pair; momentum of BTC vs ETH.
* `l1` vs `alts`: group-average comparisons.
* `meme`: special high-beta basket.

---

## 2. RV-Momentum Engine

**File:** `execution/rv_momentum.py` (NEW)

### 2.1 Data Structures

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class RvConfig:
    enabled: bool
    lookback_bars: int
    half_life_bars: int
    normalize_mode: str
    btc_vs_eth_weight: float
    l1_vs_alt_weight: float
    meme_vs_rest_weight: float
    per_symbol_weight: float
    max_abs_score: float

@dataclass
class RvSymbolScore:
    symbol: str
    score: float

@dataclass
class RvSnapshot:
    per_symbol: Dict[str, RvSymbolScore]
    btc_vs_eth_spread: float
    l1_vs_alt_spread: float
    meme_vs_rest_spread: float
```

### 2.2 Returns Loading

Assume we already have a returns/ohlc loader used by trend/carry. Use it here:

```python
def load_returns_for_symbols(symbols: list[str], lookback_bars: int) -> Dict[str, np.ndarray]:
    """
    Returns a dict mapping symbol -> np.array of log/percent returns over lookback.
    If data missing, symbol may be skipped or filled with zeros.
    """
```

### 2.3 Pair Relative Momentum (BTC vs ETH)

```python
def compute_pair_relative_momentum(
    long_symbol: str,
    short_symbol: str,
    returns: Dict[str, np.ndarray]
) -> float:
    """
    Returns relative momentum of long vs short. E.g. mean(long - short).
    Positive means long_symbol outperforming short_symbol.
    """
```

### 2.4 Basket Relative Momentum (L1 vs ALTS, MEME vs REST)

For two groups A and B:

* Compute average return profile of each basket.
* Relative momentum = mean(returns_A - returns_B).

```python
def compute_basket_relative_momentum(
    group_a: list[str],
    group_b: list[str],
    returns: Dict[str, np.ndarray]
) -> float:
    """
    Returns relative momentum of group A vs group B.
    Uses equal-weight average of available symbols in each group.
    """
```

For `meme_vs_rest`:

* `group_a = meme basket`
* `group_b = all_symbols - meme`

---

## 3. Per-symbol RV Factor

### 3.1 Normalization

Implement:

```python
def normalize_scores(
    raw_scores: Dict[str, float],
    mode: str,
    max_abs: float
) -> Dict[str, float]:
    """
    mode: "zscore" or "rank"
    Returns per-symbol scores scaled to [-max_abs, max_abs].
    """
```

* `"zscore"`:

  * compute mean & std across symbols
  * z = (raw - mean) / std
  * rescale z to [-max_abs, max_abs] (e.g. using tanh or linear scaling)
* `"rank"`:

  * rank symbols by raw score
  * map ranks → linearly spaced in [-max_abs, max_abs].

### 3.2 Combined Per-Symbol RV Score

Implement:

```python
def build_rv_snapshot(
    cfg: RvConfig,
    returns: Dict[str, np.ndarray],
    baskets_cfg: dict
) -> RvSnapshot:
    """
    Computes:
    - btc_vs_eth_relative_momo
    - l1_vs_alt_relative_momo
    - meme_vs_rest_relative_momo
    - per_symbol final rv_score
    """
```

Logic:

1. Compute:

   * pair spread: btc_vs_eth
   * group spreads: l1_vs_alt, meme_vs_rest
2. For each symbol:

   * Start from 0.
   * If symbol is BTC or ETH → add btc_vs_eth contribution (sign depends on side in pair).
   * If symbol in L1 basket:

     * add + l1_vs_alt_weight × l1_vs_alt_spread
   * If symbol in ALTS basket:

     * add - l1_vs_alt_weight × l1_vs_alt_spread
   * If symbol is MEME:

     * add + meme_vs_rest_weight × meme_vs_rest_spread
   * If symbol in REST (non-meme), apply opposite sign for that component.
   * Optionally add its own raw momentum (e.g. cumulative return) × per_symbol_weight.
3. Normalize resulting per-symbol scores to [-max_abs, max_abs].

`RvSnapshot.per_symbol[sym].score` is the final `rv_score` for that symbol.

---

## 4. Hybrid Scoring Integration

**File:** `execution/symbol_score_v6.py`

### 4.1 Config Dataclass

Add:

```python
@dataclass
class RvHybridConfig:
    enabled: bool
    weight: float      # how much rv_score contributes within hybrid composition
    max_abs_score: float
```

Map from `strategy_config["rv_momentum"]`.

### 4.2 In hybrid_score()

Where we compose:

```python
hybrid = trend_component + carry_component + expectancy_component + router_component
# + maybe others (e.g. vol/regime adjustments)
```

Integrate:

```python
if rv_cfg.enabled:
    rv_score = rv_snapshot.per_symbol.get(symbol).score if rv_snapshot else 0.0
    rv_score = max(-rv_cfg.max_abs_score, min(rv_cfg.max_abs_score, rv_score))
    hybrid += rv_cfg.weight * rv_score
```

Then, as usual:

* apply alpha decay
* apply regimes
* clamp final hybrid to [-1, 1].

---

## 5. State Publishing & Loading

### 5.1 State Publish

**File:** `execution/state_publish.py`

Add:

```python
def write_rv_momentum_state(rv_snapshot: RvSnapshot, state: dict) -> None:
    state["rv_momentum"] = {
        "per_symbol": {
            sym: {
                "score": entry.score
            }
            for sym, entry in rv_snapshot.per_symbol.items()
        },
        "spreads": {
            "btc_vs_eth": rv_snapshot.btc_vs_eth_spread,
            "l1_vs_alt": rv_snapshot.l1_vs_alt_spread,
            "meme_vs_rest": rv_snapshot.meme_vs_rest_spread
        }
    }
```

Ensure this is called alongside other intel publishing (e.g. hybrid/carry/vol regimes).

### 5.2 Loader

**File:** `execution/state_v7.py`

Add:

```python
def load_rv_momentum_state() -> dict:
    """
    Returns rv_momentum block from intel state, or {}.
    """
```

This will be used by:

* hybrid scorer (optional if running in multi-pass mode)
* dashboard intel panel

---

## 6. Signal Screener Integration

**File:** `execution/signal_screener.py`

Enhance ranking:

* Primary sort: hybrid_score (already in place).
* Secondary sort: rv_score (higher first) when hybrid scores are close.

Optionally add a mild RV filter:

* If `rv_score < rv_min_for_long` and signal is LONG, down-rank or drop.
* If `rv_score > rv_max_for_short` and signal is SHORT, down-rank or drop.

This keeps the screener favouring **relative leaders** on the long side and **relative laggards** on the short side (if shorts used).

Keep the filter config-driven (e.g. add small block under `rv_momentum` if needed).

---

## 7. Dashboard Integration

**File:** `dashboard/intel_panel.py`

Add “Relative Momentum” view:

### 7.1 Symbol Table

Columns:

* Symbol
* RV Score (–1 to 1)
* Bucket(s) (L1 / ALT / MEME / REST)
* Hybrid Score (for context)
* Tier (CORE / SAT / TACT / ALT-EXT)

Sort default by RV score descending.

### 7.2 Spreads View

Small summary:

* BTC vs ETH spread (positive = BTC stronger)
* L1 vs ALTS spread
* MEME vs REST spread

Optionally color them:

* Green if spread supports risk-on rotation,
* Yellow for mild,
* Red for extreme one-sided regime.

---

## 8. Tests

### 8.1 `tests/test_rv_momentum_baskets.py` (NEW)

* Correct loading of `rv_momo_baskets.json`.
* Correct classification of symbols into:

  * L1 / ALTS / MEME / REST sets.
* Edge cases: symbol in multiple baskets (last wins or defined behaviour).

### 8.2 `tests/test_rv_momentum_factor.py` (NEW)

* Synthetic returns where:

  * BTC strongly outperforms ETH → btc_vs_eth spread positive as expected.
  * L1 basket outperforms ALTS → positive spread.
  * MEME underperforms rest → negative spread.
* Check per-symbol raw rv_score before normalization is directionally correct.

### 8.3 `tests/test_symbol_score_rv_momentum.py` (NEW)

* With rv_momentum disabled → hybrid unchanged.
* With enabled:

  * positive rv_score increases hybrid (for long-weighted factor).
  * negative rv_score decreases hybrid.
* Combined with router/decay/regimes still yields clamped [-1, 1].

### 8.4 `tests/test_state_publish_rv_momentum.py` (NEW/extend)

* `rv_momentum` block present in state.
* Contains per_symbol scores and spreads.

### 8.5 `tests/test_signal_screener_rv_momentum.py` (NEW)

* When hybrid scores are equal, higher rv_score ranks first.
* If rv filters enabled:

  * Very negative rv_score prunes some long signals (or downgrades them).

All new tests must pass; total suite remains green (existing unrelated failures remain as-is).

---

## 9. Acceptance Criteria

The patchset is complete when:

1. `rv_momentum` config and baskets file exist and load correctly.
2. `execution/rv_momentum.py`:

   * Computes pair and basket relative momentum.
   * Produces per-symbol rv_score in [-1, 1].
3. `symbol_score_v6.py`:

   * Incorporates rv_score into hybrid_score when enabled.
4. `state_publish.py`:

   * Writes rv_momentum state with per_symbol scores and spreads.
5. `state_v7.py`:

   * Loads rv_momentum state for downstream use.
6. `signal_screener.py`:

   * Uses rv_score as secondary sort key, and optionally as a mild filter.
7. Dashboard:

   * Shows RV scores per symbol and spreads across baskets.
8. Tests:

   * `test_rv_momentum_baskets.py`
   * `test_rv_momentum_factor.py`
   * `test_symbol_score_rv_momentum.py`
   * `test_state_publish_rv_momentum.py`
   * `test_signal_screener_rv_momentum.py`
     all pass, and the overall test suite remains green (aside from pre-existing failures).

No changes may be made to drawdown, VaR/CVaR, ledger, router quality, or slippage contracts beyond what is explicitly described here.

```
