# GPT-Hedge v7.6 — Institutional Execution & Telemetry Release

**Release Date:** December 2025  
**Version:** v7.6  
**Branch:** v7.6-dev → main

---

## Overview

v7.6 is a **stability and observability release** focused on institutional-grade execution quality, robust factor governance, hardened state contracts, and comprehensive operational visibility. No new alpha strategies are introduced; the emphasis is on reliability, transparency, and operational safety.

---

## Feature Summary

### Router Microstructure (P1)
- Per-symbol slippage, latency, and TWAP usage tracking via RouterStats accumulator
- Quality scoring with slippage drift buckets (GREEN/YELLOW/RED) and latency buckets (FAST/NORMAL/SLOW)
- router_health.json v2 with global + per-symbol metrics

### Hybrid Scoring & Vol Regime (P2)
- Vol regime factor integrated into hybrid alpha scoring
- Regime-aware sizing adjustments based on volatility environment
- Router quality factor penalizes symbols with degraded execution

### Factor Diagnostics (P3)
- Gram-Schmidt orthogonalization removes factor redundancy
- Factor covariance/correlation matrix computation
- IR-based auto-weighting with configurable bounds and EMA smoothing
- PnL attribution surface for factor-level analysis

### Exit Pipeline & Ledger Authority (P4)
- Unified position ledger as canonical source for positions + TP/SL
- Ledger-registry reconciliation with mismatch breakdown
- Exit coverage metrics and underwater-without-TP/SL detection

### NAV & Risk Snapshot Coherence (P5)
- NAV anomaly guards with max jump detection
- DD state machine (NORMAL/DRAWDOWN/RECOVERY) with flap detection
- VaR/CVaR computation with configurable limits and breach flags

### Dashboard Regime Strip (P6)
- Unified state loader with consistent error handling
- Regime badges for Vol/DD/Router/Risk regimes
- Color-coded status indicators for operational visibility

### State Contract Hardening (P7)
- Single-writer atomic surfaces for all state files
- Timestamp coherence enforcement
- Schema validation via integration tests

### Preflight & Versioning (P8)
- `make pretag-v7.6` target for pre-release validation
- `scripts/runtime_sanity_check_v7_6.py` for runtime health verification
- engine_metadata.json with version tracking

### Schema Reconciliation (P9)
- Test expectations aligned with v7.6 canonical state surfaces
- Full test suite green (1314 passed, 47 skipped)

---

## Upgrade Instructions

### Pre-Upgrade Validation

```bash
# Pull the release
git fetch --tags origin
git checkout v7.6

# Run preflight validation
make pretag-v7.6

# Run runtime sanity check
python scripts/runtime_sanity_check_v7_6.py
```

### Deployment

```bash
# Restart services
sudo supervisorctl restart hedge:
```

### Post-Activation Checks

1. Verify `logs/state/*.json` timestamps are fresh
2. Confirm `engine_metadata.json` shows `version: v7.6`
3. Dashboard regime strip renders correctly
4. Router health metrics populated
5. Exit coverage > 90%
6. Factor diagnostics panel shows weights/IR
7. No repeated errors in executor logs

---

## Rollback

If issues are discovered post-activation:

```bash
git fetch --tags origin
git checkout v7.5
sudo supervisorctl restart hedge:
```

State surfaces are forward-compatible. On rollback to v7.5:
- v7.5 reads v7.6 state files successfully (ignores new fields)
- Some v7.6-only fields will be absent
- No data loss or corruption expected

---

## Documentation

| Document | Purpose |
|----------|---------|
| [Release Notes](docs/v7.6_Release_Notes.md) | Operator-facing summary |
| [Change Log](docs/v7.6_Change_Log.md) | Developer-facing details |
| [Runtime Activation](docs/v7.6_Runtime_Activation.md) | Ops deployment guide |
| [Investor Stability](docs/v7.6_Investor_Stability.md) | Risk controls overview |
| [Pre-Tag Audit](docs/v7.6_Pre_Tag_Audit.md) | Validation checklist |
| [State Contract](docs/v7.6_State_Contract.md) | State surface specs |
| [Architecture](docs/v7.6_Architecture.md) | Runtime flow and modules |

---

## Test Results

```
1314 passed, 47 skipped
```

All integration, unit, and runtime tests pass. Legacy tests are skipped as expected.

---

## Contributors

GPT-Hedge v7.6 development team.

---

## License

Proprietary. All rights reserved.
