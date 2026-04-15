# Conviction Authority — FROZEN

**Date:** 2026-04-12  
**Status:** FROZEN — conviction surface economically falsified  
**Effective immediately. Binding until replaced by a new validated surface.**

---

## Decision

Futures conviction authority is suspended. The conviction score is demoted from entry authority to telemetry-only. No entries, position sizing, candidate ranking, or band gating may use conviction scores to permit or size futures trades.

## Evidence

Post-audit validation (`data/post_audit_validation/validation_results.json`) demonstrates:

| Symbol | Spearman ρ | Q5−Q1 | Monotonicity | Verdict |
|--------|-----------|-------|-------------|---------|
| BTCUSDT | +0.1177 | −0.4402 | 0.4444 | FAIL (all 3) |
| ETHUSDT | +0.2003 | +4.6671 | 0.5556 | FAIL (monotonicity) |
| SOLUSDT | +0.0341 | −4.1510 | 0.4444 | FAIL (all 3) |

The conviction distribution is collapsed: IQR = 0.057, 80% of 1,358 episodes in [0.61, 0.68]. Score does not separate winners from losers. All quintiles show negative mean PnL. Win rate across all bands: 10.1%.

## What Changed

1. `config/strategy_config.json`: `conviction.enabled` → `false`, `conviction.mode` → `"off"`
2. `config/strategy_config.json`: `conviction_enabled` → `false` (meta-scheduler)
3. All conviction authority paths are now inert:
   - Executor conviction band gate (mode != "live" → skipped)
   - Conviction size multiplier (enabled=false → early return)
   - Candidate selector band gate (min_conviction_band="" → no filter)
   - Screener conviction enrichment (enabled=false → skipped)
4. Hybrid scoring pipeline still runs for telemetry/logging — scores are computed but do not gate.

## What Is Retained

- **Zero-score guard**: The 3-layer zero-score abstain (Hydra/Screener/Selector) remains active as a safety invariant, not conviction authority.
- **Router quality filter**: Separate from conviction, remains active.
- **Hybrid score computation**: Runs for observability, drift detection, and research. Logged to `selector_v2_shadow.jsonl` and `symbol_scores_v6.json`.
- **All conviction code paths**: Preserved and tested. Conviction can be re-enabled in config if a new surface passes validation.

## What Is Prohibited

1. **Re-enabling conviction authority** without a new surface that demonstrates:
   - Spearman ρ > 0.15 for all traded symbols
   - Q5−Q1 > 0 (top quintile outperforms bottom)
   - Monotonicity ratio ≥ 0.75
   - Fee-clearing expectancy in shadow mode
2. **Tuning conviction thresholds, bands, or weights** on the current surface. The problem is collapsed distribution, not miscalibration.
3. **Rebucketing conviction bands** or remapping the conviction gradient. These are rescue tactics explicitly prohibited by containment doctrine.
4. **Using ETH's faint positive signal (ρ=0.20)** to justify partial re-enablement. Monotonicity still fails and total PnL is negative.

## Replacement Path

The next futures entry surface should be a **new family**, not a repaired conviction gradient:

- Sparse event-style entry states
- Regime-state transitions
- Structural dislocations
- Explicit binary permits (not continuous confidence)
- Causal price-state setups with low turnover
- PM Sleeve / payoff-asymmetry structures

## Verification

```bash
# Confirm conviction is frozen
python3 -c "
import json
cfg = json.load(open('config/strategy_config.json'))
conv = cfg.get('conviction', {})
assert conv.get('enabled') == False, f'conviction.enabled={conv.get(\"enabled\")}'
assert conv.get('mode') == 'off', f'conviction.mode={conv.get(\"mode\")}'
print('CONVICTION AUTHORITY: FROZEN ✓')
"
```
