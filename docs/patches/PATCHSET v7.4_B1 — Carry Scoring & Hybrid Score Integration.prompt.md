# PATCHSET v7.4_B1 — Carry Scoring & Hybrid Score Integration

## Objective

Turn the existing **carry inputs** (funding rate, basis) into a **live alpha factor** and integrate it into:

- Symbol scoring (intel layer)
- Screener ranking and gating
- Telemetry and dashboard

Key outputs:

- `carry_score` in `[-1.0, +1.0]`
- `hybrid_score = w_trend * trend_score + w_carry * carry_score`
- Screener uses `hybrid_score` instead of bare trend
- Intel state + dashboard show these scores clearly

Must **not** break existing contracts or behaviour for symbols where carry data is missing.

---

## Files to Touch

- `config/strategy_config.json`
- `execution/intel/symbol_score_v6.py` (or the main symbol scoring module)
- `execution/signal_screener.py`
- `execution/state_publish.py`
- `dashboard/intel_panel.py` or `dashboard/overview_panel.py` (where symbol scores are rendered)
- Tests:
  - `tests/test_symbol_score_carry.py` (new)
  - `tests/test_signal_screener_hybrid_score.py` (new or extend existing)
  - `tests/test_state_publish_intel_scores.py` (new or extend)

No changes to executor entrypoints, router, or risk limits.

---

## 1. Config: Weights & Thresholds

**File:** `config/strategy_config.json`

### Change

Add a **hybrid scoring** block that is:

- Compatible with the Strategy Tiering model (CORE/SATELLITE/TACTICAL/ALT-EXT)
- Simple to start, tier-aware later

Example:

```json
"hybrid_scoring": {
  "default": {
    "w_trend": 0.7,
    "w_carry": 0.3,
    "min_hybrid_score_long": 0.15,
    "min_hybrid_score_short": 0.15
  },
  "tiers": {
    "CORE": {
      "w_trend": 0.6,
      "w_carry": 0.4
    },
    "SATELLITE": {
      "w_trend": 0.7,
      "w_carry": 0.3
    },
    "TACTICAL": {
      "w_trend": 0.8,
      "w_carry": 0.2
    },
    "ALT-EXT": {
      "w_trend": 0.5,
      "w_carry": 0.5
    }
  }
}
````

Rules:

* `w_trend + w_carry` **do not need** to equal 1.0, but in practice should be close.
* If tier-specific weights are missing for a tier, fall back to `"default"`.
* `min_hybrid_score_long` / `short` are global thresholds the screener can use to decide if an intent is viable.

No existing config keys should be removed or renamed.

---

## 2. Symbol Scoring: Add Carry & Hybrid Score

**File:** `execution/intel/symbol_score_v6.py`
(or whatever file computes per-symbol trend scores used by the screener)

### 2.1 Inputs

Assume we already have access to:

* `funding_rate_8h` (e.g. decimal, 0.0005 = 0.05% per 8h)
* `basis_bps` (annualized or rolling basis, in bps or decimal – treat as “carry signal”)
* Trend metrics: `trend_score` or equivalent in `[-1, +1]`

If inputs are not yet plumbed, stub them as optional and handle `None` gracefully.

### 2.2 Dataclasses

Extend the main symbol score dataclass to include carry + hybrid fields:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class SymbolScore:
    symbol: str
    trend_score: float
    # NEW:
    carry_score: float
    hybrid_score: float
    # keep existing fields: rsi, ema_slope, etc.
```

If there is already a `SymbolScore` dataclass, **extend** it rather than creating a new one.

Define a helper input dataclass if useful:

```python
@dataclass
class CarryInputs:
    funding_rate_8h: Optional[float] = None   # raw funding rate per 8h
    basis_bps: Optional[float] = None         # e.g. annualized basis in bps
```

### 2.3 Carry Scoring Function

Add:

```python
def score_carry(inputs: CarryInputs) -> float:
    """
    Map funding + basis into a carry score in [-1.0, +1.0].

    Positive = favorable carry for holding the position in the intended direction.
    Negative = expensive carry.
    0.0 = neutral or insufficient data.
    """
```

Implementation rules:

* If both `funding_rate_8h` and `basis_bps` are None → return 0.0.

* Normalize funding to a small range:

  * E.g., clamp to [-0.01, +0.01] (±1% per 8h), then scale to [-0.7, +0.7].

* Normalize basis similarly:

  * Convert bps to decimal: `basis = basis_bps / 10000.0`.
  * Clamp to [-0.20, +0.20] (±20% annualized) then scale to [-0.3, +0.3].

* Combine:

  ```python
  score = funding_component + basis_component
  score = max(-1.0, min(1.0, score))
  ```

* If only one of funding/basis is present, use that component alone.

* This score should be **direction-neutral** for now (favourable carry in absolute sense). The screener can later interpret direction (long vs short).

### 2.4 Hybrid Score

In the symbol scoring pipeline:

1. Load hybrid config:

   * Determine tier for this symbol/strategy (if available); otherwise use `"default"`.
   * Extract `w_trend` and `w_carry`.

2. Compute:

```python
carry_score = score_carry(carry_inputs)
hybrid_score = w_trend * trend_score + w_carry * carry_score
# optional clamp:
hybrid_score = max(-1.0, min(1.0, hybrid_score))
```

3. Populate `SymbolScore` with these fields.

4. Ensure:

* If `carry_score` is 0.0 (e.g., no data), hybrid behavior should be close to previous trend-only behavior.
* No exceptions thrown if funding or basis are missing.

---

## 3. Screener: Use Hybrid Score for Ranking and Gating

**File:** `execution/signal_screener.py`

### 3.1 Ranking

Wherever we currently sort/rank candidates by `trend_score` (or similar), switch to `hybrid_score`:

```python
candidates.sort(key=lambda s: s.hybrid_score, reverse=True)
```

If there are existing tie-breakers (e.g. volatility, liquidity), **preserve them**.

### 3.2 Gating

Use the config thresholds defined in `strategy_config["hybrid_scoring"]["default"]`:

* For **long** candidates: require `hybrid_score >= min_hybrid_score_long`
* For **short** candidates: require `hybrid_score <= -min_hybrid_score_short`

If tier-specific thresholds are introduced later, design with that in mind, but for now:

```python
if direction == "LONG":
    if hybrid_score < min_hybrid_score_long:
        # do not emit long intent
elif direction == "SHORT":
    if hybrid_score > -min_hybrid_score_short:
        # do not emit short intent
```

Ensure:

* Existing filters (liquidity, ATR, tier constraints, etc.) remain intact.
* If config missing, fall back to conservative defaults (e.g. 0.0 threshold, meaning “no gating”).

### 3.3 Telemetry (Screener)

If the screener logs candidate selection, extend its JSON with:

```json
{
  "trend_score": ...,
  "carry_score": ...,
  "hybrid_score": ...
}
```

This helps debugging “why this symbol was picked / skipped”.

---

## 4. Intel State & Telemetry

**File:** `execution/state_publish.py`

Wherever we publish intel state (usually to `logs/state/intel.json` or similar):

### Change

For each symbol, include:

```json
"trend_score": ...,
"carry_score": ...,
"hybrid_score": ...
```

Rules:

* Add fields, do **not** rename existing ones.
* If carry data is missing, `carry_score` should be `0.0` (or `null` if that’s more consistent with current style, but prefer 0.0 if the scoring function already handles missing data).

Example state fragment:

```json
"BTCUSDT": {
  "trend_score": 0.22,
  "carry_score": 0.35,
  "hybrid_score": 0.27,
  "...": "..."
}
```

This state will be consumed by the dashboard intel panel.

---

## 5. Dashboard: Show Carry & Hybrid Score

**File:** `dashboard/intel_panel.py` (or `dashboard/overview_panel.py`)

### Change

1. Extend the per-symbol intel table to include:

* `trend_score`
* `carry_score`
* `hybrid_score`

If space is limited, at least show:

* `hybrid_score` (primary)
* `carry_score` in a smaller column or tooltip

2. Visual hints:

* Optional: Colour or icon scaling by hybrid score (e.g. strong green > 0.3, red < -0.3).
* Optional: Show “T/C/H” triple in a single cell.

3. Make sure dashboard does **not** assume these fields are absent; handle missing/null gracefully (display `—`).

---

## 6. Tests

### 6.1 Carry Scoring Tests

**File:** `tests/test_symbol_score_carry.py` (new)

Test `score_carry()`:

1. **Neutral**:

   * No inputs (`funding_rate_8h=None`, `basis_bps=None`) → `carry_score == 0.0`.

2. **Positive funding + basis**:

   * E.g. `funding_rate_8h = 0.0005`, `basis_bps = 200`.
   * Expect `carry_score > 0.0`.

3. **Negative funding + basis**:

   * E.g. `funding_rate_8h = -0.0005`, `basis_bps = -200`.
   * Expect `carry_score < 0.0`.

4. **Clamping**:

   * Use extreme inputs (e.g. `funding_rate_8h = 0.05`) and ensure score is capped at `+1.0`.
   * Similar for negative extreme.

### 6.2 Hybrid Score Integration Tests

**File:** `tests/test_symbol_score_hybrid.py` (new or extend symbol score tests)

Scenarios:

1. **Carry neutral**:

   * `carry_score = 0.0`, `trend_score = 0.4`, `w_trend = 0.7`, `w_carry = 0.3`.
   * Expect `hybrid_score ≈ 0.28` (or identical to scaled trend).

2. **Positive carry boosts hybrid**:

   * `trend_score = 0.2`, `carry_score = 0.5`.
   * Expect `hybrid_score > trend_score`.

3. **Negative carry drags hybrid**:

   * `trend_score = 0.3`, `carry_score = -0.5`.
   * Expect `hybrid_score < trend_score`.

4. **Clamp**:

   * When combination overshoots, `hybrid_score` should remain within `[-1.0, +1.0]`.

### 6.3 Screener Hybrid Behaviour Tests

**File:** `tests/test_signal_screener_hybrid_score.py` (new or extend screener tests)

Scenarios:

1. **Ranking by hybrid_score**:

   * Two symbols with different (trend, carry) combos.
   * Ensure symbol with higher hybrid_score is ranked first.

2. **Long gating**:

   * `min_hybrid_score_long = 0.15`.
   * For a candidate long with `hybrid_score = 0.10`, ensure **no long intent** emitted.
   * For `hybrid_score = 0.20`, intent is emitted.

3. **Short gating**:

   * `min_hybrid_score_short = 0.15`.
   * For `hybrid_score = -0.10`, ensure no short.
   * For `hybrid_score = -0.25`, short allowed.

### 6.4 State Publish Tests

**File:** `tests/test_state_publish_intel_scores.py` (new or extend intel state tests)

* Verify that for a sample symbol:

  * `trend_score`, `carry_score`, and `hybrid_score` appear in the state JSON.
  * Values match what symbol scoring produced.

All existing tests must remain passing.

---

## 7. Acceptance Criteria

The patch is complete when:

1. `strategy_config.json` defines `hybrid_scoring` with weights and thresholds.

2. `symbol_score_v6` (or equivalent) computes:

   * `carry_score` in `[-1.0, +1.0]`
   * `hybrid_score` using config weights

3. Screener uses `hybrid_score` for:

   * Ranking symbols.
   * Gating long/short intents via `min_hybrid_score_long/short`.

4. `state_publish` exposes `trend_score`, `carry_score`, `hybrid_score` in intel state.

5. Dashboard intel panel displays carry + hybrid score (at least hybrid, ideally trend + carry + hybrid).

6. All new tests pass:

   * `test_symbol_score_carry.py`
   * `test_symbol_score_hybrid.py`
   * `test_signal_screener_hybrid_score.py`
   * `test_state_publish_intel_scores.py`

7. No regressions in existing tests or runtime behaviour for symbols lacking carry data.

Do not change any other behaviour or contracts beyond what is described here.

```
