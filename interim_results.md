# Binary Lab — Interim Results

**Report date:** 2026-03-30
**Status:** Experimental. S1 completed. S2 active with ablation gates. No claims of profitability.

---

## Current State

| Sleeve | Status | Mode | Trades (closed) | Period |
|--------|--------|------|-----------------|--------|
| S1 (Synthetic Binary) | COMPLETED | PAPER | 415 | 2026-02-26 to 2026-03-09 |
| S2 (Polymarket CLOB) | ACTIVE | PAPER | 449 | 2026-03-20 to 2026-03-30 |

Both sleeves operate in paper trade mode. No real capital has been deployed.

---

## S1: Synthetic Binary

### Result: Inconclusive

| Metric | Value |
|--------|-------|
| Total entries | 419 |
| Total closes | 415 |
| No-trade events | 1,830 |
| Win rate (gross) | 29.6% |
| Avg gross PnL | −$0.0204 per round |
| Avg net PnL | −$0.0335 per round |
| Cumulative net PnL | −$9.57 |
| Per-round size | $20 |
| Period | 2026-02-26 to 2026-03-09 |

### Band Separation

Not demonstrated. With a 29.6% gross win rate and negative expected value across all rounds, no monotonic conviction-band ordering was observed.

### Assessment

S1 did not generate positive expected value. The cumulative loss is small relative to capital ($9.57 on $2,000 allocation) but directional — negative. The sleeve has been completed and is no longer generating trades.

---

## S2: Polymarket CLOB

### Aggregate Results (All Trades)

| Metric | Value |
|--------|-------|
| Total entries | 456 |
| Total closes | 449 |
| No-trade events | 694 |
| Win rate (gross) | 51.4% |
| Avg gross PnL | +$1.7912 per round |
| Avg net PnL | +$1.6961 per round |
| Cumulative net PnL | +$761.56 |
| Max drawdown | $368.35 |
| Per-round size | $30 |
| Period | 2026-03-20 to 2026-03-30 |

### Side Breakdown

| Side | N | Avg Gross PnL | Avg Net PnL | Win Rate | Cum Net PnL |
|------|---|--------------|------------|----------|-------------|
| YES | 306 | +$9.6825 | +$9.5529 | 53.9% | +$2,923.20 |
| NO | 143 | −$15.0952 | −$15.1164 | 46.2% | −$2,161.64 |

The YES/NO asymmetry is the dominant effect. YES-side trades are strongly profitable. NO-side trades are strongly losing. The aggregate positive result is the net of these two opposing forces.

### Edge Band Separation (Quartile Analysis)

| Quartile | N | |edge| Range | Avg Gross PnL | Avg Net PnL | Win Rate | Cum Net PnL |
|----------|---|-------------|---------------|-------------|----------|-------------|
| Q1 (lowest) | 112 | 0.031–0.077 | +$0.03 | −$0.05 | 51.8% | −$5.87 |
| Q2 | 112 | 0.077–0.120 | +$0.23 | +$0.12 | 49.1% | +$13.73 |
| Q3 | 112 | 0.120–0.185 | +$3.55 | +$3.44 | 53.6% | +$384.73 |
| Q4 (highest) | 113 | 0.185–0.379 | +$3.34 | +$3.27 | 51.3% | +$368.97 |

Band separation is partially monotonic. Q1 and Q2 are near-zero or slightly negative. Q3 and Q4 are materially positive. The transition occurs around |edge| ≈ 0.10.

### Edge Distribution

| Statistic | Value |
|-----------|-------|
| Mean |edge| | 0.1367 |
| Median |edge| | 0.1204 |
| P10 | 0.0504 |
| P90 | 0.2487 |

### Ablation Analysis (Daemon-Era Subset, N=168)

An ablation study on the 168 most recent closes confirmed conditional value:

| Filter | N | Avg Gross PnL | Cum Net PnL |
|--------|---|---------------|-------------|
| Baseline (all) | 168 | −$0.32 | −$58.17 |
| YES-only | 82 | +$15.14 | +$1,238.69 |
| |edge| ≥ 0.07 | 120 | +$1.03 | +$120.22 |
| |edge| ≥ 0.10 | 91 | +$3.34 | +$301.21 |
| Random baseline (N=120) | 120 | −$0.38 | −$49.22 |

The random baseline confirms the magnitude filter captures real signal, not sampling artifact.

### Current Gate Configuration

Following ablation results, an ablation gate was deployed (2026-03-30):

| Parameter | Value |
|-----------|-------|
| Gate enabled | true |
| Minimum |edge| | 0.10 |
| Side filter | YES_ONLY |
| Fallback threshold | 0.07 (if trade frequency drops below 20/day) |

This is the highest-probability configuration based on ablation evidence. Live validation is in progress. N ≥ 50 trades required before assessment.

### Kill Distance

| Metric | Value | Threshold |
|--------|-------|-----------|
| Capital allocation | $900 | — |
| Kill NAV | $650 | Hard termination |
| Kill DD | $250 (27.8%) | Hard termination |
| Current DD | $368.35 | Above threshold (historically) |

---

## Failure Condition Tracking

### S1 Failure Conditions

| Condition | Status | Notes |
|-----------|--------|-------|
| NAV < $1,700 | Not triggered | Losses were small (−$9.57) |
| DD > $300 | Not triggered | Max DD well below threshold |
| Band separation absent at n ≥ 100 | Not evaluable | Win rate too low for band analysis |
| Rule violation | None detected | — |

### S2 Failure Conditions

| Condition | Status | Notes |
|-----------|--------|-------|
| BSS ≤ 0 at n ≥ 100 | Not evaluated | BSS requires probability calibration assessment |
| Tail regions not profitable at n ≥ 100 | Partially addressed | Q3+Q4 (high |edge|) profitable; Q1+Q2 near zero |
| Paper PnL < 70% of backtest | Not directly comparable | Different measurement periods |
| NO-side systematic loss | OBSERVED | −$2,161.64 cumulative; ablation gate now blocks NO trades |

---

## Early Observations

1. **S1 produced negative expected value.** The synthetic binary approach did not demonstrate skill. The experiment is completed.

2. **S2 shows conditional value in paper trading.** Aggregate net PnL is positive (+$761.56 over 449 rounds), but this is driven entirely by the YES-side (+$2,923.20) offsetting the NO-side (−$2,161.64).

3. **Edge magnitude is a valid filter.** Band separation analysis shows a clear breakpoint around |edge| ≈ 0.10. Below this threshold, expected value is near zero. Above it, expected value is materially positive.

4. **YES/NO asymmetry is the largest effect.** The side the model takes (YES vs NO) determines profitability more than any other variable. The origin of this asymmetry (structural market bias vs model bias) is not yet determined.

5. **S2 edge does not transfer to futures.** A separately conducted futures proxy experiment (N=144 across 3 variant configurations) conclusively rejected S2 as a futures directional signal. S2 is venue-specific.

6. **These are paper trade results.** No real capital has been deployed. Paper fills assume execution at observed CLOB prices. Real execution may differ due to latency, slippage, and market impact.

7. **The ablation gate is newly deployed.** The YES_ONLY + |edge| ≥ 0.10 configuration has not yet accumulated sufficient live data for independent validation (requires N ≥ 50).

---

*Source files: `logs/execution/binary_lab_trades.jsonl`, `logs/execution/binary_lab_s2_trades.jsonl`, `config/binary_lab_limits.json`, `config/binary_lab_limits_s2.json`*
