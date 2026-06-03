# Phase 1a Signal Audit — Remediation Memo

**Date:** 2026-06-03
**Branch:** `claude/evaluate-bot-performance-7k0VA`
**Audit script:** `research/signal_causality_audit.py`
**Full results:** `logs/state/signal_causality_audit.json`

---

## Resume status: blocked

`hybrid_score` is empirically falsified on 377 scored closed episodes drawn from the live episode ledger. BTC ρ=−0.213 (p=0.027) and ETH ρ=−0.172 (p=0.049) are statistically significantly anti-predictive. Pooled ρ=−0.082 fails the pre-registered ρ>0.15 gate on all three criteria.

| Criterion | Gate | Pooled | BTC | ETH | SOL |
|-----------|------|-------:|----:|----:|----:|
| Spearman ρ | >0.15 | −0.082 | −0.213 | −0.172 | −0.095 |
| p-value | <0.05 | 0.112 | **0.027** | **0.049** | 0.271 |
| Q5 − Q1 | >0 | −0.002 | — | — | — |
| N | — | 377 | 109 | 132 | 136 |

Higher `hybrid_score` on BTC and ETH predicts **worse** outcomes with statistical significance. All five quintiles show negative mean return. Hit rate across 1,037 total episodes: 10.4%. Total realized PnL: −$1,124.

---

## Sign flip is not the fix

Inverting every signal direction (treating the score as a contrarian indicator) yields pooled ρ=+0.082 — still below the 0.15 gate — and hit rate 73.5% with per-trade gross return of +0.146%. After the ~0.08% round-trip fee floor, this is roughly breakeven. The anti-prediction is concentrated in BTC and ETH; the wider universe is noise. Flipping the sign on a broken model produces a marginally less broken model. The signal needs **redesign**, not inversion.

Reproduce at any time:

```bash
python -m research.signal_causality_audit            # standard audit
python -m research.signal_causality_audit --sign-flip  # sign-flip counterfactual
```

---

## What happened to the halt diagnosis

The comprehensive evaluation report (2026-05-19) attributed the system halt to `nav_stale_age=97s`, `min_notional` vetoes from the fee gate, and 1,119 `min_notional` risk vetoes. Phase 0 forensics on the production system found all three to be incorrect:

- **NAV:** Fresh, `sources_ok: true`, ~10s old. No NAV bug.
- **`min_notional` vetoes:** Periodic health probes (`notional=0.0`, `intent_id=None`), not real intents.
- **Fee gate:** Has not been reached since 2026-05-06 because intents are vetoed upstream by doctrine.
- **Actual halt cause:** Doctrine `VETO_ENVIRONMENT_DEGRADED` on every intent for ~8 days. Sentinel-X split MEAN_REVERT 55% / CHOPPY 45% — no dominant regime. Doctrine Law #2 ("no stable regime = no trade") is working as designed.

The system is not broken. It is correctly refusing to trade in a degraded regime.

---

## Next valid work

**Hypothesis A backtest** (funding rate extremes) is ready on the production server:

```bash
python3 -m research.funding_rate_extremes \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --days 180 --horizon 4
```

Also run at `--horizon 2` and `--horizon 8` to check whether the edge is horizon-specific.

Falsification criteria are pre-registered in `research/funding_rate_extremes.py` as constants and will not change after the first data run. If the hypothesis passes (ρ>0.15, p<0.05, Q5−Q1>0, fee-adjusted ρ>0) on all three symbols, proceed to paper trade design. If it fails or is conditional, redesign before re-testing on new data.

**Phase 1b** (replay vs live direction divergence, 69.3% mismatch) is deferred. It is forensically useful but not a resume blocker given the signal audit result — both paths are trading on a falsified signal. Revisit after Hypothesis A produces a verdict.

---

## Audit reproducibility

All work is committed to `claude/evaluate-bot-performance-7k0VA`:

| Artifact | Path |
|----------|------|
| Audit script | `research/signal_causality_audit.py` |
| Unit tests (7, all pass) | `tests/research/test_signal_causality.py` |
| Full results JSON | `logs/state/signal_causality_audit.json` |
| Sign-flip results | `logs/state/signal_causality_audit_signflip.json` |
| Audit doc appendices | `docs/SIGNAL_OUTCOME_CAUSALITY_AUDIT_2026-03-18.md` |
| Funding rate backtest | `research/funding_rate_extremes.py` |
| Backtest tests (15, all pass) | `tests/research/test_funding_rate_extremes.py` |
