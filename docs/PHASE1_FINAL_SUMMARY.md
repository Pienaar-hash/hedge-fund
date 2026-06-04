# Phase 1 Research Program — Final Summary

**Date:** 2026-06-04
**Branch:** `claude/evaluate-bot-performance-7k0VA`
**Disposition:** HALTED

---

## Verdict

The research program found no causal signal. Zero of five tested hypotheses
cleared the pre-registered falsification gate. The live signal (`hybrid_score`)
is statistically anti-predictive. The program halts.

This is not a temporary result. The pre-committed exit condition was written
before the final test ran precisely to prevent "one more cycle" drift. It holds.

---

## What was tested

| | Hypothesis | Signal | Mechanism | Result |
|-|------------|--------|-----------|--------|
| Audit | `hybrid_score` (live) | Composite | Unknown | ρ=−0.21 BTC, −0.17 ETH — **ANTI-PREDICTIVE** |
| A | Funding rate extremes | −(fr / rolling_mean_abs) | Carry unwind / mean-reversion | **FAIL** |
| D | Time-series momentum | past-N-week price return | Price autocorrelation | **FAIL** |
| E | Cross-sectional momentum | rank(past return) across assets | Relative mispricing | **FAIL** |
| F | Funding rate momentum | mean(fr_recent) − mean(fr_prior) | Positioning trend continuation | **FAIL** |
| G | OI change rate | oi_return × sign(price_dir) | Position sizing momentum | **BLOCKED** |

Three distinct economic mechanisms: carry/mean-reversion, price autocorrelation,
and positioning. All fail. G is blocked on data (Binance retains 30 days of OI
history; 16 pairs max vs 50-pair minimum). OI tracks the same underlying
positioning as funding; the Coinalyze alternative was considered and declined —
extending the timeline to test a signal correlated with an already-falsified
signal class is not a mechanistic reason to continue.

---

## What the failure pattern says

**Funding signals (A, F): anti-predictive, not just absent.**
Both the level and slope of the funding rate have consistently negative ρ on
BTC and ETH. This is the same direction as the live signal audit. The market
appears to price funding information quickly enough that acting on it produces
the wrong side of the trade.

**Price signals (D, E): near-zero, no consistent direction.**
Time-series and cross-sectional momentum are consistent with efficient pricing
of public price information at daily-to-weekly resolution in this 3-asset
universe. The academic result (Moskowitz 2012) operates at 6–12 month lookbacks
across 50+ uncorrelated assets; testing 4-week momentum on BTC/ETH/SOL is not
that strategy.

**Common cause:** The system was built before the signal was validated. The audit
and research program ran in the right order relative to each other, but both
ran after live capital was deployed. The $1,124 loss (−11.2% on $10K inception
NAV) is the cost of that sequencing error.

---

## What was built

The infrastructure is intact and correct. Nothing needs to be discarded.

| Component | Status | Notes |
|-----------|--------|-------|
| Causality audit framework | Ready | `research/signal_causality_audit.py` |
| Spearman ρ + p-value stack | Ready | Tested, clean Lentz impl, 110 unit tests |
| Falsification pipeline | Ready | Pre-registered gates, consistent across all scripts |
| `funding_rate_extremes.py` | Ready | Hypothesis A |
| `weekly_momentum.py` | Ready | Hypothesis D |
| `cross_sectional_momentum.py` | Ready | Hypothesis E |
| `funding_rate_momentum.py` | Ready | Hypothesis F |
| `oi_momentum.py` | Ready | Hypothesis G (pending data) |
| `oi_collector.py` | Running | Passive daily OI snapshot to `data/oi_history.jsonl` |

If the program restarts, the correct sequence is: hypothesis written → historical
data test passes gate → paper trade → live capital. Not the reverse.

---

## Conditions for restart

1. A signal hypothesis with a plausible causal mechanism *not* derived from:
   - Funding rate (level or slope) — falsified
   - Price autocorrelation — falsified
   - Cross-sectional price momentum — falsified
   - OI change (correlated with funding) — not yet tested but mechanistically similar

   Candidates not yet evaluated: net liquidation flow, realised vol regime,
   on-chain exchange inflow/outflow, options implied vol vs realised vol spread.
   Each requires either a paid data feed or a different venue.

2. The hypothesis must clear the historical data gate *before* a single line
   of execution code is written or modified.

3. If the capital base is not replenished, systematic trading at $10K is
   unlikely to generate net-positive returns after fees regardless of signal
   quality. The fee drag (~0.1% per trade) on a signal with ρ=0.15 leaves
   very thin margin. The program should not restart at this NAV unless the
   expected edge is materially larger than anything found in Phase 1.

---

## Branch state

All research scripts in `research/`. All unit tests in `tests/research/`.
110 tests, all pass. No production code was modified.

The branch `claude/evaluate-bot-performance-7k0VA` is clean and complete.
PR #36 documents the full research record.
