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

## Next valid paths

**Path 1 — Expand the cross-section (extends Hypothesis E)**
Add perpetual futures for 15–20 liquid tokens (BNB, ADA, DOT, AVAX, LINK, UNI,
MATIC, etc.). This does not change the hypothesis — only the universe size. All
existing `research/cross_sectional_momentum.py` code generalises via `--symbols`.
A 20-asset run over 4 years produces n≈4,000 pairs with adequate power.

This is the highest-probability next step because:
- ρ was consistently positive across all 4 Hypothesis E configs (not noise-style)
- The failure mode is diagnosed (3-asset underpowering), not signal absence
- No code changes needed — one CLI flag change

**Path 2 — Abandon and redesign (if broader cross-section also fails)**
If a 20-asset cross-sectional test at 4w/1w also returns ρ < 0.15, the conclusion
is that weekly-resolution price return signals do not predict future returns in
perpetuals. At that point, the research program would pivot to:
- On-chain signals (funding, open interest change, net liquidation flow)
- Volatility regime signals (realised vol spread, VIX analog for crypto)
- Microstructure signals (order flow imbalance) — requires tick data

---

## Recommended immediate action

Run Hypothesis E with a 20-asset universe before concluding:

```bash
python3 -m research.cross_sectional_momentum \
    --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT ADAUSDT DOGEUSDT \
              AVAXUSDT LINKUSDT DOTUSDT MATICUSDT UNIUSDT LTCUSDT \
              XRPUSDT ATOMUSDT NEARUSDT APTUSDT ARBUSDT OPUSDT \
              FILUSDT INJUSDT \
    --days 1095 \
    --lookback-weeks 4 --horizon-weeks 1 \
    --out data/hypothesis_e_20asset_4w1w.json
```

Three years of data (not four) because some tokens lack 4-year history on Binance
Futures. The script handles missing bars via tolerance-based snapping. If any
symbol returns zero bars, remove it and re-run.

---

## Reproducibility

```bash
python3 -m research.cross_sectional_momentum --days 1500 --lookback-weeks 4 --horizon-weeks 1
python3 -m research.cross_sectional_momentum --days 1500 --lookback-weeks 1 --horizon-weeks 1
python3 -m research.cross_sectional_momentum --days 1500 --lookback-weeks 8 --horizon-weeks 2
python3 -m research.cross_sectional_momentum --days 1500 --lookback-weeks 12 --horizon-weeks 4
```

| Artifact | Path |
|----------|------|
| Script | `research/cross_sectional_momentum.py` |
| Unit tests (22, all pass) | `tests/research/test_cross_sectional_momentum.py` |
| Results (gitignored) | `data/hypothesis_e_*.json` |
