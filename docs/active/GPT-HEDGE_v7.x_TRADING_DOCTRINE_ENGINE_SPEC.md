Good. Then let’s do this **properly** — not as a config tweak, not as a research note, but as a **doctrine**.

What follows is **the trading constitution** for the system you already built.
If this fails, it fails honestly.

---

# GPT-HEDGE v7.x

## TRADING DOCTRINE (ENGINE SPEC)

This doctrine defines **when the system is allowed to exist in the market**, **when it must refuse to trade**, and **how positions are allowed to die**.

No toggles. No experiments. This is the engine.

---

## I. FIRST PRINCIPLES (Non-Negotiable)

1. **Markets are noisy at short horizons**

   * The system does **not** attempt to predict the next candle.
   * It participates only when **regime persistence** exists.

2. **Survival precedes profit**

   * The system’s primary goal is to **remain positioned through noise** when right.
   * Exits are based on **regime invalidation**, not pain thresholds.

3. **Capital is allocated to regimes, not signals**

   * Signals only decide *direction*.
   * Regimes decide *permission* and *size*.

4. **Churn is the enemy**

   * Any behaviour that increases turnover without increasing expectancy is a defect.

---

## II. REGIME AUTHORITY (Sentinel-X is Supreme)

Sentinel-X is the **sole gatekeeper** of market participation.

### Allowed Regimes

| Sentinel-X Regime | Allowed Action                        |
| ----------------- | ------------------------------------- |
| `TREND_UP`        | Long only                             |
| `TREND_DOWN`      | Short only                            |
| `MEAN_REVERT`     | Mean reversion head only              |
| `CHOPPY`          | Flat or micro-size only               |
| `BREAKOUT`        | Entry allowed only after confirmation |
| `CRISIS`          | Forced capital contraction            |

If Sentinel-X says **CRISIS**:

* All heads are throttled to near-zero
* No new positions unless **explicitly hedging**

This is not negotiable.

---

## III. WHEN THE SYSTEM TRADES

### A. Entry Permission (ALL must be true)

1. **Regime Persistence**

   * Sentinel-X regime unchanged for ≥ *N* cycles
   * No single-bar regime flips

2. **Directional Alignment**

   * TREND_UP → Long signals only
   * TREND_DOWN → Short signals only
   * MEAN_REVERT → Z-score based entries only

3. **Execution Feasibility**

   * Minotaur execution regime ≠ `CRUNCH`
   * Spread and depth within tolerances

4. **Portfolio Head Budget Available**

   * Hydra head has non-zero remaining budget
   * Alpha Router allocation > floor

If **any** of the above fails:

> **No trade is allowed.**

---

### B. What a “Signal” Actually Means

A signal is **not** a command to trade.

A signal is:

> “If we are allowed to be in the market, this is the direction.”

Signals **never override regime**.

---

## IV. WHEN THE SYSTEM REFUSES TO TRADE

The system **must refuse to trade** under the following conditions:

### 1. Regime Ambiguity

* Sentinel-X confidence below threshold
* Competing regime probabilities close together

### 2. Volatility without Direction

* HIGH volatility + NEUTRAL trend
* This is where most retail systems die

### 3. Execution Hostility

* Minotaur classifies regime as `WIDE_SPREAD` or `CRUNCH`
* Liquidity is unreliable

### 4. Alpha Mortality

* Alpha Decay survival probability < threshold
* Strategy is statistically dying → step aside

Refusal is **strength**, not weakness.

---

## V. POSITION SIZING (Capital as Ammunition)

Position size is the product of:

```
Base Risk
× Regime Multiplier
× Strategy Health
× Alpha Survival
× Execution Quality
```

No single component can force size up.

This is already implemented across:

* Alpha Router
* Hydra PnL throttling
* Cerberus head multipliers

**Fixed % NAV sizing is banned.**

---

## VI. HOW POSITIONS EXIT (THIS IS CRITICAL)

### A. What We DO NOT USE

❌ Symmetric ATR TP/SL
❌ Fixed reward:risk exits
❌ Candle-based stopouts

These create:

* Small wins
* Large losses
* Fee bleed

---

### B. Valid Exit Reasons (Only These)

#### 1. **Regime Invalidation (Primary Exit)**

* Sentinel-X flips regime **against** the position
* Or confidence collapses

This is the main exit.
Most trades should die here.

---

#### 2. **Structural Failure (Secondary Exit)**

* Trend strength decays below threshold
* Carry edge disappears
* Relative value spread mean-reverts fully

Exit slowly, not violently.

---

#### 3. **Time Stop (Safety Valve)**

* Position has not progressed after *T* bars
* Market is ignoring the thesis

This prevents capital imprisonment.

---

#### 4. **Crisis Override**

* Sentinel-X enters CRISIS
* Forced partial or full liquidation

Survival > correctness.

---

### C. Trailing Profit Logic (Optional, Not Mandatory)

* Trailing is **volatility-aware**
* Never tightens during HIGH volatility
* Only trails when regime remains supportive

No profit target is required.

Let winners exist.

---

## VII. ROLE OF ADVANCED MODULES (WHY THEY EXIST)

Your “fancy” modules finally have a purpose:

| Module          | Actual Role               |
| --------------- | ------------------------- |
| Sentinel-X      | Market permission         |
| Alpha Decay     | Strategy mortality        |
| Hydra           | Intent arbitration        |
| Cerberus        | Regime capital routing    |
| Minotaur        | Execution survivability   |
| Execution Alpha | Post-trade truth          |
| Crossfire       | Structural relative value |
| Alpha Miner     | Emergent edge discovery   |

None of these generate *entries*.
They govern **whether the system is allowed to act**.

---

## VIII. WHAT FAILURE LOOKS LIKE (HONESTLY)

If this system fails:

* It will fail **slowly**
* Drawdowns will be explainable
* Logs will tell a coherent story
* You will know *why* it failed

That is the difference between:

* gambling
* and running a real system

---

## IX. FINAL LINE IN THE SAND

> **The system is not here to trade often.
> It is here to trade correctly when the market allows it.**

