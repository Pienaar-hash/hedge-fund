# Phase 1b — Hypothesis D: Weekly Time-Series Momentum

**Date:** 2026-06-03
**Branch:** `claude/evaluate-bot-performance-7k0VA`
**Script:** `research/weekly_momentum.py`

---

## Verdict: HYPOTHESIS NOT SUPPORTED

Time-series momentum at weekly resolution is not present in BTC, ETH, or SOL
over the 2024–2026 period. All four configurations fail all three symbols
on the primary gate (ρ > 0.15). No configuration achieves statistical
significance on any symbol.

---

## Results

### Config 1 — 4-week lookback, 1-week forward (primary)

| Symbol | ρ | p | Q5−Q1 | Mono | Verdict |
|--------|---------|--------|---------|------|---------|
| BTCUSDT | +0.0245 | 0.807 | +0.024 | 0.50 | FAIL 2/5 |
| ETHUSDT | +0.0318 | 0.751 | +0.034 | 0.50 | FAIL 2/5 |
| SOLUSDT | −0.0611 | 0.542 | −0.026 | 0.50 | FAIL 0/5 |

### Config 2 — 1-week lookback, 1-week forward (short lookback)

| Symbol | ρ | p | Q5−Q1 | Mono | Verdict |
|--------|---------|--------|---------|------|---------|
| BTCUSDT | −0.0872 | 0.377 | −0.032 | 0.25 | FAIL 0/5 |
| ETHUSDT | −0.1250 | 0.204 | −0.016 | 0.50 | FAIL 0/5 |
| SOLUSDT | −0.1522 | 0.121 | +0.012 | 0.50 | FAIL 1/5 |

### Config 3 — 8-week lookback, 2-week forward (longer horizon)

| Symbol | ρ | p | Q5−Q1 | Mono | Verdict |
|--------|---------|--------|---------|------|---------|
| BTCUSDT | +0.0880 | 0.552 | +0.033 | 0.50 | FAIL 2/5 |
| ETHUSDT | +0.0508 | 0.732 | +0.010 | 0.50 | FAIL 2/5 |
| SOLUSDT | +0.0126 | 0.932 | +0.005 | 0.50 | FAIL 2/5 |

### Config 4 — 4-week lookback, 1-week forward, vol-normalized

| Symbol | ρ | p | Q5−Q1 | Mono | Verdict |
|--------|---------|--------|---------|------|---------|
| BTCUSDT | +0.0218 | 0.828 | +0.016 | 0.50 | FAIL 2/5 |
| ETHUSDT | +0.0341 | 0.734 | +0.034 | 0.50 | FAIL 2/5 |
| SOLUSDT | −0.0659 | 0.511 | −0.020 | 0.75 | FAIL 1/5 |

---

## What the data shows

**No momentum effect at any horizon tested.** ρ values range from −0.152 to
+0.088 across all 12 symbol × configuration combinations. All p-values are far
above the 0.05 gate (range 0.12–0.93), consistent with no systematic
predictability.

**Lookback length reverses sign.** The 1-week lookback shows negative ρ on all
three symbols (mean-reversion tendency), while 4-week and 8-week lookbacks show
weakly positive ρ on BTC and ETH. This sign flip is itself evidence of noise:
the signal direction depends on parameter choice, which is characteristic of
data-mined rather than structural effects.

**SOL is incoherent across configurations.** SOL shows negative ρ at 1w and
4w lookbacks but turns positive at 8w. There is no consistent directional
signal.

**Best candidate: SOL 1-week mean-reversion** (ρ=−0.152, p=0.121 for 1w/1w).
This is the closest result to the gate in absolute terms — as an anti-signal it
would imply ρ=+0.152 — but it still misses significance and falls below the ρ >
0.15 threshold. With n=107 it would need ρ > 0.19 to reach p < 0.05.
Not a foundation for a trading decision.

---

## Why momentum may be absent here

Academic momentum (Moskowitz, Ooi, Pedersen 2012) operates across long
lookback windows (12 months) and multi-asset portfolios where diversification
sharpens the signal. At the single-asset, weekly level on highly correlated
crypto assets during a single 2-year bull-market cycle:

- All three assets move together — cross-sectional sorting collapses to noise.
- A 2-year window (n≈104 non-overlapping 1-week periods) provides limited
  power. Detecting ρ=0.15 with 80% power requires approximately n=175 pairs.
- The 2024–2026 period contains a strong bull trend that masks any
  autocorrelation in the residual.

---

## Scorecard: hypotheses tested to date

| Hypothesis | Signal | Result | Best ρ |
|------------|--------|--------|--------|
| A — Funding rate extremes | anti_signal = −(fr / mean_abs) | FAIL all horizons | −0.04 |
| D — Weekly time-series momentum | (close_t − close_{t−Nw}) / close_{t−Nw} | FAIL all configs | +0.09 |

Both hypotheses, across both directions of the market (mean-reversion and
trend-following), return ρ values indistinguishable from zero.

---

## Next valid work

The two most tractable paths from here:

**Hypothesis E — Cross-sectional momentum (relative, not absolute)**
Sort assets by prior-period return and go long the top tercile, short the
bottom. This uses the 3-asset universe as a cross-section. Effect size may be
larger because it removes the common trend. Testable with the same framework;
will need a pooled Spearman across (signal_rank, return) pairs.

**Increase sample size: extend history or add assets**
n=104 weekly non-overlapping pairs is underpowered for detecting ρ=0.15.
Options: (a) use Binance OHLCV from 2020 onwards (~4 years, n≈200), or
(b) add BNBUSDT, ADAUSDT, DOGEUSDT to the universe (n multiplies by asset
count). Neither changes the hypothesis; both improve power.

Both can be built as extensions of `research/weekly_momentum.py` without
touching any production code.

---

## Reproducibility

```bash
python3 -m research.weekly_momentum --symbols BTCUSDT ETHUSDT SOLUSDT \
    --days 730 --lookback-weeks 4 --horizon-weeks 1

python3 -m research.weekly_momentum --lookback-weeks 1 --horizon-weeks 1
python3 -m research.weekly_momentum --lookback-weeks 8 --horizon-weeks 2
python3 -m research.weekly_momentum --vol-normalize --lookback-weeks 4 --horizon-weeks 1
```

| Artifact | Path |
|----------|------|
| Script | `research/weekly_momentum.py` |
| Unit tests (28, all pass) | `tests/research/test_weekly_momentum.py` |
| Config 1 results | `data/hypothesis_d_4w1w.json` (gitignored) |
| Config 2 results | `data/hypothesis_d_1w1w.json` (gitignored) |
| Config 3 results | `data/hypothesis_d_8w2w.json` (gitignored) |
| Config 4 results | `data/hypothesis_d_4w1w_volnorm.json` (gitignored) |
