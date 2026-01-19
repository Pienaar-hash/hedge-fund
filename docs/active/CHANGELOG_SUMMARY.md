# Changelog Summary — v7.8 (Current)

**Doctrine Version:** v7.8  
**Cycle:** CYCLE_003 (Active)  
**Last Material Change:** January 19, 2026  

---

## Current Doctrine State

The trading system operates under **frozen doctrine v7.8**. No parameter changes, threshold adjustments, or execution logic modifications are permitted during CYCLE_003.

### Active Components

| Component | Purpose |
|-----------|---------|
| Doctrine Kernel | Regime-gated entry/exit permission |
| Sentinel-X | Regime classification (TREND_UP/DOWN, MEAN_REVERT, CHOPPY, CRISIS) |
| Episode Ledger | Completed trade cycle tracking (read-only observability) |
| Regime Pressure | Permission dynamics monitoring |

---

## Recent Material Changes

### v7.8 (January 2026)
- **Episode Ledger** — Derived view of completed trade cycles
- **CYCLE_002 Archive** — 443 episodes preserved with full provenance
- **Regime Pressure Dashboard** — Permission dynamics observability
- **Doctrine Falsification Criteria** — Explicit rules for doctrine invalidation

### v7.6 (December 2025)
- **Sentinel-X** — Regime classifier with stability requirements
- **Doctrine Kernel** — Entry/exit gate with regime governance
- **Position Ledger** — Unified TP/SL tracking
- **Router Health** — Execution quality monitoring

### v7.4 (November 2025)
- **Risk Engine v6** — Typed decisions, exposure caps
- **Hybrid Alpha v2** — Multi-factor scoring
- **TWAP Router** — Slippage-aware execution

---

## Archived Documentation

Full patch history and release notes are preserved in:

```
docs/archive/
├── patch_notes/     # Individual patch specifications
├── v7_releases/     # v7.4 - v7.7 release docs
├── legacy/          # Pre-v7 documentation
└── drafts/          # Exploratory analysis (pre-CYCLE_002)
```

> **Note:** Changes prior to v7.8 are non-operative. Historical docs preserved for audit and provenance only.

---

## Cycle History

| Cycle | Period | Episodes | Net PnL | Status |
|-------|--------|----------|---------|--------|
| CYCLE_002 | Dec 13, 2025 - Jan 19, 2026 | 443 | -$217 | Archived |
| CYCLE_003 | Jan 19, 2026 - present | TBD | TBD | Active |

---

## Success Criteria (CYCLE_003)

| Metric | Threshold | CYCLE_002 Baseline |
|--------|-----------|-------------------|
| Avg Duration | > 2h | 1.6h |
| Exit:Entry Ratio | < 2:1 | 3.9:1 |
| Fee Share | < 30% | ~50% |
| Confidence Persistence | > 0.65 for 48h+ | Not achieved |

---

*Document updated: January 19, 2026*
