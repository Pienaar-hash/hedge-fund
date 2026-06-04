# Phase 1c — Hypothesis E: Cross-Sectional Momentum

**Date:** 2026-06-04
**Branch:** `claude/evaluate-bot-performance-7k0VA`
**Script:** `research/cross_sectional_momentum.py`

---

## Verdict: HYPOTHESIS NOT SUPPORTED

Cross-sectional momentum is not detectable in the BTC/ETH/SOL universe at any
of the four lookback/horizon combinations tested. All configurations fail on
four of five pre-registered criteria.

---

## Results

| Config | n_total | ρ | p | T3−T1 | Mono | Verdict |
|--------|---------|---------|-------|---------|------|---------|
| 4w lookback / 1w fwd | 642 | +0.064 | 0.106 | +0.006 | 0.50 | FAIL 2/5 |
| 1w lookback / 1w fwd | 651 | +0.047 | 0.236 | +0.001 | 0.50 | FAIL 2/5 |
| 8w lookback / 2w fwd | 315 | +0.020 | 0.725 | −0.000 | 0.50 | FAIL 1/5 |
| 12w lookback / 4w fwd | 153 | +0.053 | 0.521 | −0.001 | 0.50 | FAIL 1/5 |

---

## What the data shows

**Sign is consistently positive.** Unlike Hypothesis D where ρ reversed sign across
lookback lengths (noise signature), Hypothesis E produces positive ρ in all four
configurations. This is the weakest possible evidence of a directional effect, but it
is consistent with the hypothesis direction.

**Magnitude is noise-level.** Best result is ρ=+0.064 (p=0.106). Detecting
ρ=0.064 with 80% power requires n≈1,900 pairs; the test has n=642. The
hypothesis is not falsified by its own strength — it is underpowered.

**Monotonicity locks at 0.50 universally.** The middle tercile (T2) never sits
reliably between T1 and T3. This is structurally inevitable with only 3 assets:
at each time point the ranks are exactly {0, 0.5, 1.0}. The middle asset is
always T2 by definition, and whether T2's forward return falls between T1 and T3
is a coin flip independent of any signal.

**Spread flips negative at longer horizons.** At 8w/2w and 12w/4w, T3−T1 goes
slightly negative — winners at medium-term horizons slightly underperform losers,
which is consistent with eventual mean-reversion crowding out the momentum effect.

---

## Why the test is underpowered (structural, not fixable by more data)

Cross-sectional momentum requires a cross-section. With 3 assets:
- Each period produces exactly 3 signal values: {0, 0.5, 1.0}
- The signal is effectively a ternary variable (short, neutral, long)
- The middle asset contributes zero information (it is always "neutral")
- Effective degrees of freedom per period ≈ 2

A viable cross-sectional test requires 10–20 assets minimum. With 20 assets:
- 20 distinct rank values per period, resolving signal to ±5th percentile
- 1-week sampling over 4 years → n ≈ 200 periods × 20 assets = 4,000 pairs
- 80% power to detect ρ=0.10 — a realistic cross-sectional effect size

This is a data/universe problem, not a hypothesis problem.

---

## Research program scorecard (to date)

| Hypothesis | Signal class | Timescale | Best ρ | Verdict |
|------------|-------------|-----------|--------|---------|
| A — Funding rate extremes | Carry / mean-reversion | Hours | −0.04 | FAIL |
| D — Time-series momentum | Price autocorrelation | Weeks | +0.09 | FAIL |
| E — Cross-sectional momentum | Relative performance | Weeks | +0.06 | FAIL / UNDERPOWERED |
| (audit) — hybrid_score | Composite, live signal | Live episodes | −0.21 (BTC) | ANTI-PREDICTIVE |

Three distinct economic mechanisms tested. Zero signal detected above noise in a
3-asset, weekly-resolution framework.

---

## 20-asset result (2026-06-04)

| Config | n_total | ρ | p | T3−T1 | Mono | Verdict |
|--------|---------|---------|-------|---------|------|---------|
| 20 assets / 4w lookback / 1w fwd | 920 | +0.0098 | 0.766 | +0.009 | 1.00 | CONDITIONAL 3/5 |

ρ collapsed from +0.064 (3-asset) to +0.0098 (20-asset). This is the wrong
direction for a pure underpowering diagnosis — more assets should have amplified
the signal, not reduced it. The test window was constrained to ~46 periods per
asset because newer tokens required all 20 symbols to be concurrently available.

**Conclusion: HYPOTHESIS FALSIFIED.** Weekly cross-sectional price return rank
does not predict next-week return in perpetual futures. The program pivots to
on-chain signals.

---

## Next valid paths

**Path 1 — Expand the cross-section (extends Hypothesis E)**
~~Run Hypothesis E with a 20-asset universe before concluding.~~

*Completed 2026-06-04 — see 20-asset result above. Pivot to on-chain signals.*

**Path 2 — On-chain/flow signals**
Weekly-resolution price return signals do not predict future returns in
perpetuals. Next signal class rotation:
- **Hypothesis F — Funding rate momentum** (trending fr predicts continuation)
- Open interest change rate (OI growth predicts price continuation)
- Net liquidation flow (cascade risk when liq volume spikes)

---

## Reproducibility

```bash
python3 -m research.cross_sectional_momentum --days 1500 --lookback-weeks 4 --horizon-weeks 1
python3 -m research.cross_sectional_momentum --days 1500 --lookback-weeks 1 --horizon-weeks 1
python3 -m research.cross_sectional_momentum --days 1500 --lookback-weeks 8 --horizon-weeks 2
python3 -m research.cross_sectional_momentum --days 1500 --lookback-weeks 12 --horizon-weeks 4

# 20-asset run
python3 -m research.cross_sectional_momentum \
    --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT ADAUSDT DOGEUSDT \
              AVAXUSDT LINKUSDT DOTUSDT MATICUSDT UNIUSDT LTCUSDT \
              XRPUSDT ATOMUSDT NEARUSDT APTUSDT ARBUSDT OPUSDT \
              FILUSDT INJUSDT \
    --days 1095 --lookback-weeks 4 --horizon-weeks 1
```

| Artifact | Path |
|----------|------|
| Script | `research/cross_sectional_momentum.py` |
| Unit tests (22, all pass) | `tests/research/test_cross_sectional_momentum.py` |
| Results (gitignored) | `data/hypothesis_e_*.json` |
