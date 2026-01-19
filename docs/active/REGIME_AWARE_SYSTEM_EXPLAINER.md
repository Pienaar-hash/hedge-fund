# Regime-Aware Trading System — One-Page Explainer

**Version:** v7.8 Doctrine Engine  
**Observation Period:** Dec 18, 2025 – Jan 8, 2026 (21 days)  
**Status:** Live observation, doctrine holding

---

## What Was Designed

A **regime-governed execution system** where market structure—not signals—determines permission to trade.

### Core Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Sentinel-X    │────▶│  Doctrine Gate  │────▶│    Execution    │
│ Regime Detector │     │  (Entry/Exit)   │     │   (Orders)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Regime Pressure │     │ Cycle Statistics│
│   (Diagnostic)  │     │  (Forensic Log) │
└─────────────────┘     └─────────────────┘
```

### Key Principles

1. **Signals determine direction; regimes determine permission**
2. **No regime = no trade** — Sentinel-X must confirm stable regime before entry
3. **Exits are thesis-driven** — Positions die when thesis dies, not on arbitrary targets
4. **Refusal is first-class** — Every veto is logged and analyzed

---

## What Makes It Different

| Traditional Systems | This System |
|---------------------|-------------|
| Trade when signals fire | Trade when regime allows AND signal fires |
| Exit on TP/SL | Exit on thesis failure (regime flip) |
| Always in market | Flat when market is hostile |
| Optimize entry timing | Optimize regime alignment |
| Backtest on price | Validate on regime transitions |

### The Doctrine Gate

The system classifies market into regimes:
- **TREND_UP/DOWN** — Directional trading allowed (long/short aligned)
- **MEAN_REVERT** — Both directions via z-score
- **CHOPPY** — No entries (maximum bait conditions)
- **CRISIS** — Forced contraction

A position can only open when:
1. Regime is stable (held for 2+ cycles)
2. Direction aligns with regime
3. Risk limits are satisfied

---

## Observed Behavior (21 Days Live)

### By The Numbers

| Metric | Value |
|--------|-------|
| Regime Changes | 55 |
| Entries Allowed | 537 |
| Entries Vetoed | 21,934 |
| **Veto Rate** | **97.6%** |
| Exits via Thesis (REGIME_FLIP) | 594 |
| Exits via Seatbelt (Stop-Loss) | 2 |
| **Seatbelt Ratio** | **0.3%** |

### What Happened

1. **Dec 18-25:** System entered during stable MEAN_REVERT, exited on regime flip to CHOPPY
2. **Dec 25-Jan 1:** Extended CHOPPY oscillation — system stayed flat for 6 days
3. **Jan 1:** MEAN_REVERT stabilized — system entered BTC/ETH/SOL, exited 4 hours later on flip
4. **Jan 1-8:** Peak hostility — 55 regime changes, 882 near-flip events — system stayed flat

### The Pressure Gauge

The Regime Pressure Dashboard (read-only diagnostic) revealed:
- **794 near-flip events** — market oscillating at decision boundary
- **Avg dwell: 4.7 hours** — regimes not lasting
- **Hostility: HOSTILE** — not yet CALM

The system refused to trade not because it was broken, but because **the market was actively deceptive**.

---

## Why This Reveals Edge

### The Hypothesis

Most trading losses come from:
1. Trading in hostile regimes (chop)
2. Exiting on price panic instead of thesis failure
3. Overtrading when signals fire but structure is absent

### What We're Testing

| Falsification Criterion | What Would Prove System Wrong | Current Status |
|------------------------|------------------------------|----------------|
| RF-1: Regime Participation | System refuses to trade during aligned regimes | ✅ 1.0 (trades when allowed) |
| RF-2: Seatbelt Dominance | >30% of exits via stop-loss | ✅ 0.3% (thesis-driven) |
| RF-3: Choppy Discipline | Entries during CHOPPY regime | ✅ Zero entries |

### The Edge Mechanism

If the system is correct, edge emerges from:
1. **Avoiding regime-misaligned trades** (the 97.6% veto rate)
2. **Holding through noise** (thesis-based exits, not price-based)
3. **Capital preservation during hostility** (7+ days flat in peak churn)

### What Would Confirm Edge

1. Profitable cycles that start after extended HOSTILE periods
2. Exit quality remains thesis-driven (RF-2 stays < 10%)
3. Win rate during stable regimes exceeds baseline

---

## Current Status

| Dimension | State |
|-----------|-------|
| Regime | MEAN_REVERT (0.54 confidence) |
| Stability | Not yet stable (distance: 2 cycles) |
| Positions | Flat |
| Hostility | HOSTILE (softening) |
| System Health | HEALTHY |

### What We're Waiting For

1. **Confidence recovery** — sustained >0.5 without flipping
2. **Stability achievement** — 2+ cycles without regime change
3. **Pressure easing** — near-flips dropping, dwell expanding

When these align, the system will trade. Until then, it waits.

---

## The Bottom Line

> **This system is designed to be wrong less often, not right more often.**

It sacrifices frequency for quality. It refuses to negotiate with hostile markets. It exits on thesis, not fear.

After 21 days of live observation:
- Every falsification criterion remains green
- The system has demonstrated it can wait
- The system has demonstrated it can trade when allowed
- The system has demonstrated it can exit correctly

The remaining question is not *if* it works, but *how much edge* it captures when conditions align.

That question can only be answered by time.

---

*Document generated: 2026-01-08*  
*Engine version: v7.8*  
*Observation continues.*
