# Research Program Status

**Date:** 2026-06-04
**Branch:** `claude/evaluate-bot-performance-7k0VA`
**Status:** Phase 1 complete — pivoting to on-chain signals

---

## Executive summary

Three distinct economic mechanisms have been tested at weekly resolution in BTC,
ETH, and SOL. None pass the pre-registered gate (ρ > 0.15, p < 0.05). The live
signal (`hybrid_score`) is statistically anti-predictive.

- **Hypotheses A and D** are cleanly falsified — effects absent, not underpowered.
- **Hypothesis E** at 20 assets: ρ collapsed from +0.064 (3-asset) to +0.0098. Falsified.
- **Phase 2 target:** on-chain/flow signals — Hypothesis F (funding rate momentum).

---

## Full results

| Test | Signal | Universe | Timescale | Best ρ | p | Verdict |
|------|--------|----------|-----------|--------|---|---------|
| Audit — `hybrid_score` | Composite live | BTC/ETH/SOL | Live episodes | −0.213 (BTC) | 0.027 | **ANTI-PREDICTIVE** |
| Hypothesis A — Funding rate extremes | Level: −(fr/mean_abs) | BTC/ETH/SOL | Hours | −0.04 | — | **FALSIFIED** |
| Hypothesis D — Time-series momentum | Past-N-week return | BTC/ETH/SOL | Weeks | +0.09 | 0.38 | **FALSIFIED** |
| Hypothesis E — Cross-sectional momentum | Relative rank (3-asset) | BTC/ETH/SOL | Weeks | +0.064 | 0.11 | **UNDERPOWERED → FALSIFIED** |
| Hypothesis E — Cross-sectional momentum | Relative rank (20-asset) | 20 perps | Weeks | +0.0098 | 0.77 | **FALSIFIED** |

### Why each failure mode is different

**`hybrid_score` audit:** Active anti-prediction. BTC ρ=−0.213 (p=0.027), ETH ρ=−0.172 (p=0.049).
Sign inversion yields pooled ρ=+0.082, still below gate. Not a fixable signal.

**Hypothesis A (funding rate level):** Effect absent and directionally wrong. High positive
funding predicts positive return, not the mean-reversion we hypothesised. Clean falsification.

**Hypothesis D (time-series momentum):** ρ reverses sign with lookback parameter (1w negative,
4w/8w weakly positive) — noise signature. Sign stability with more data (1500d): SOL flipped
from −0.152 to +0.071. Clean falsification.

**Hypothesis E (cross-sectional momentum):** Direction consistent across all 4 configs, but
ρ collapsed from +0.064 to +0.0098 when universe expanded from 3 to 20 assets. Wrong
direction — more assets should amplify a real signal. Falsified.

---

## Next hypothesis: Hypothesis F — Funding Rate Momentum

**Mechanism:** Unlike Hypothesis A (level: extreme funding → mean reversion), Hypothesis F
tests *trend*: when the funding rate is accelerating upward, it signals growing bullish
positioning → trend continuation over the next 1–4 days.

**Signal:**
```
signal = mean(fr[-N:]) - mean(fr[-2N:-N])   # short-window mean minus prior-window mean
```
Positive signal = funding trending up (longs increasingly willing to pay premium).

**Difference from Hypothesis A:**
- A: `signal = -(fr_t / rolling_mean_abs)` — level, mean-reverting expectation
- F: `signal = mean(fr_recent) - mean(fr_prior)` — slope, trend-continuation expectation

**Data:** Existing Binance Futures `/fapi/v1/fundingRate` endpoint — no new data infrastructure.

**Script:** `research/funding_rate_momentum.py` (to be built)

**Pre-registered gate:** same framework (ρ > 0.15, p < 0.05, Q5−Q1 > 0, mono ≥ 0.75, fee-adj ρ > 0)

---

## Document index

| Document | Scope |
|----------|-------|
| [PHASE1A_SIGNAL_AUDIT_SUMMARY.md](PHASE1A_SIGNAL_AUDIT_SUMMARY.md) | Live `hybrid_score` audit + Hypothesis A |
| [PHASE1B_HYPOTHESIS_D_SUMMARY.md](PHASE1B_HYPOTHESIS_D_SUMMARY.md) | Weekly time-series momentum |
| [PHASE1C_HYPOTHESIS_E_SUMMARY.md](PHASE1C_HYPOTHESIS_E_SUMMARY.md) | Cross-sectional momentum, 3-asset + 20-asset |

All research scripts in `research/`. All testable without touching production code.
