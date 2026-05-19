# Comprehensive Evaluation: GPT-Hedge — Six-Month Post-Mortem

**Date:** 2026-05-19
**Current state:** NAV $8,999.87 (−10.00%). System **HALTED** (`nav_stale_age=97s`). All subsystems idle 15+ minutes. Zero open positions.

---

## Executive Summary

The fund lost $1,000 not because the market was unfavorable. It lost because the system was never fit to trade. Three compounding failures drove every dollar of the loss:

1. **Broken wiring** — a key mismatch in the fee gate silently vetoed every single trade for at least 2.5 months
2. **Unproven foundations** — the core signal's causal relationship to returns has never been measured; the test framework exists but the tables are blank
3. **Complexity that concealed the damage** — 72,000 LOC of execution code, six formal audits, a falsified conviction engine, and now a Phase 5 catastrophic failure where the replay strategy trades the opposite direction from the live executor in 69% of orders

---

## Part 1: What the Numbers Actually Mean

The NAV curve is monotonically negative from day one. There is no recovery window, no period of offsetting gains. Every data point is lower than the one before it.

**Three loss sources:**

| Source | Detail |
|--------|--------|
| Direct trade losses | All 5 documented trades closed `THESIS_INVALIDATED`. Avg net loss: −$0.43/trade |
| Fee drag | 67,385 order attempts → 3,283 fills. Fee/PnL ratio in 3-day window: **4.02×** |
| Slippage | Maker fill ratio: **5.26%** (target: 50%+). P95 slippage: **43.1 bps** |

The reported 10.5% win rate is misleading: 660 of 692 episodes are from the pre-conviction era (Dec 2025). The conviction-era sample is 32 trades — statistically meaningless and also losing.

---

## Part 2: The Broken Plumbing

### 2.1 Fee gate universal veto — `executor_live.py` (fixed 2026-05-10, ~2.5 months late)

The fee gate read expected edge from two fields that **no upstream component ever writes to**:

```python
READ:  intent["metadata"]["expectancy"]        ← never set
READ:  intent["metadata"]["expected_edge_pct"] ← never set

ACTUAL data lives at:
  intent["expected_edge"]                      ← signal_generator.py L342
  intent["hybrid_components"]["expectancy"]    ← signal_screener.py L1306
```

**Result:** `expected_edge_pct = 0.0` on every intent. Every trade blocked with `$0.00 edge < $0.057 required`. The system appeared to be running — signals generated, doctrine evaluated — but the fee gate was reading from an empty dict key and vetoing everything.

The bug was introduced in an intent schema refactor. It wasn't caught for at least 2.5 months because the system's "flat by design" framing (see §6.4 below) made total inactivity look like disciplined doctrine behavior.

**The fee gate bug is the proximate cause of essentially the entire fund loss.**

### 2.2 True edge heuristic always returns zero — `true_edge.py`

Even after fixing the fee gate read path, the upstream edge value it reads is still broken:

```python
confidence_threshold = 0.5   # arbitrary, never calibrated
adv = clamp(confidence - 0.5, 0, cap)
# Live conviction scores: 0.41–0.44 (always below 0.5)
# → adv = 0.0 → edge = $0.00
```

An empirical replacement (`expectancy_bridge.py`) was built and validated in shadow mode but was **never promoted to live**. This means that even with the fee gate wiring fixed, the edge input it receives is still computed by a known-broken heuristic.

### 2.3 NAV desync — current halt cause (active right now)

The system is halted **at this moment** because `nav_confirmed.json` uses a conditional write (only updates on successful API call) while `nav.json` writes unconditionally. When Binance API degrades even briefly, the confirmed cache goes stale >90 seconds, risk fires `NAV_STALE`, and all trading halts — even though `nav.json` is current and the executor is healthy. This is a self-inflicted halt caused by the conditional-write pattern.

### 2.4 TWAP crash inflation (P0, still open)

No slice state persisted. Restart after partial TWAP → re-sends all slices → 1.5–2× position size. Flagged in the April 2026 First-Strike audit. Not fixed.

### 2.5 Dual-executor race (P0, still open)

No PID lock: `supervisorctl restart` during slow shutdown → two executors reading same position cache → contradictory orders. Flagged in the April 2026 First-Strike audit. Not fixed.

---

## Part 3: The Over-Architecture Problem

**The execution codebase: 72,613 LOC across 139 modules**

```
executor_live.py      6,747 LOC  ← single-file monolith
cerberus_router.py    1,400 LOC  ← DISABLED
sentinel_x.py         1,468 LOC  ← DISABLED
alpha_decay.py        1,422 LOC  ← DISABLED
alpha_miner.py        1,181 LOC  ← DISABLED
cross_pair_engine.py  1,147 LOC  ← DISABLED
```

The over-architecture audit (`docs/OVER_ARCHITECTURE_AUDIT_2026-03-18.md`) classified every major layer:

| Component | Classification | Proven edge? |
|-----------|---------------|-------------|
| Regime model (Sentinel-X) | TRANSFORM | No — no Brier/log-loss eval, no ablation |
| Conviction scoring | TRANSFORM | No — deterministic sizing transform of upstream scores |
| Doctrine kernel | CONTROL | No — constitutional guard, not alpha source |
| Order router | CONTROL | No — execution quality, not predictive edge |
| Cerberus/Alpha router | **DEAD** | No — disabled in config |

The audit's conclusion: **"There is no reviewed component in this list that currently clears the bar for proven SIGNAL."** If `hybrid_score` lacks edge, every layer downstream is just reshaping noise.

### 3.1 Conviction falsification (2026-04-12)

Post-audit validation on 1,358 episodes:

| Symbol | Spearman ρ | Q5−Q1 | Verdict |
|--------|-----------|-------|---------|
| BTCUSDT | +0.12 | **−0.44** | FAIL (all 3 criteria) |
| ETHUSDT | +0.20 | +4.67 | FAIL (monotonicity) |
| SOLUSDT | +0.03 | **−4.15** | FAIL (all 3 criteria) |

Distribution collapsed: IQR = 0.057, 80% of scores in [0.61, 0.68]. All quintiles show negative mean PnL. Win rate across all bands: **10.1%**. Conviction frozen on 2026-04-12. **No replacement surface exists.** The system is currently trading with no primary entry-quality filter.

### 3.2 Signal causality — the tables are blank

The `docs/SIGNAL_OUTCOME_CAUSALITY_AUDIT_2026-03-18.md` defines a rigorous Spearman ρ test for whether `hybrid_score` predicts trade outcomes. Every result cell in the document reads `—`. The methodology was written. The passive observation logger was built. The p-value computation was implemented. **The test was never run on real data.**

The foundational question — *does the signal predict returns?* — has never been answered in six months of live trading.

---

## Part 4: What the Audits Keep Finding

| Audit | Date | Primary finding |
|-------|------|-----------------|
| Full Audit | 2026-02-22 | Signal-execution wiring gaps |
| First-Strike | 2026-04-11 | 4 P0 stop-trading findings |
| Second-Strike | 2026-04-11 | `DRY_RUN` defaults false outside `ENV=prod` |
| Catastrophic State | 2026-04-11 | Structural gates failing at baseline |
| Fourth-Strike Forensic | 2026-04-11 | Restart-time exit blindness; score integrity |
| Fifth-Strike Rebuild | 2026-04-11 | Core code "intact but unverified" |
| Sixth-Strike Hardening | 2026-04-11 | 30/60/90-day remediation backlog |
| **Phase 5 Postmortem** | **2026-05-15** | **Replay trades opposite direction in 69.3% of orders** |

Four separate critical audits on the **same day** (2026-04-11). Almost none of the P0 findings resolved before the next audit.

### 4.1 Phase 5 catastrophic failure (most recent — 2026-05-15)

Shadow soak comparing certified replay vs. live executor over 3,677 orders:

| Metric | Value |
|--------|-------|
| Catastrophic direction mismatches | **2,548 (69.3%)** |
| Direction match rate | **30.7%** |
| Quantity bucket match rate | **0%** |
| Replay strategy net PnL (certification) | **−$131.45** on 536 trades |

Every catastrophic mismatch is a direction divergence: same symbol, same time, opposite side. Phase 5 terminated. Phase 6 denied. This is the current state of the most recent validation attempt.

---

## Part 5: Technical Debt Summary

### Infrastructure (P0, all open except fee gate fix)

- TWAP state not persisted → position inflation on crash
- No PID lock → dual-executor race
- Non-atomic peak state write → corrupt drawdown tracking on crash
- `FAIL_CLOSED_ON_NAV_STALE=0` env var exists → silently disables NAV safety gate
- NAV conditional-write desync → current active halt
- `DRY_RUN` defaults false outside `ENV=prod`
- Testnet overrides relax drawdown limit from 10% → 95%

### Strategy (critical, all open)

- `hybrid_score` causality never proven
- Conviction falsified, no replacement
- True edge heuristic always zero (bridge built, not promoted)
- Replay ↔ live direction divergence (69.3%)
- No backtest framework — all validation is live
- 5 of 6 Hydra heads generate zero intents (only TREND used)

### Code quality

- `executor_live.py`: 6,747 LOC single-file monolith
- ~7,000–8,000 LOC of disabled modules in the execution hot path
- <10% test coverage on 40+ risk gates
- 25+ env vars control behavior; several are dangerous
- Silent exception swallowing in config loaders (returns `{}`)
- Manifest tests fail on clean checkout (26 required artifacts missing)

---

## Part 6: Why It Went Wrong

### 6.1 Built signal-last

The infrastructure was constructed before answering whether the signal works. Conviction engine, regime model, doctrine kernel, 6 Hydra heads — all built before running a single Spearman ρ test.

### 6.2 Complexity concealed bugs

72,000 LOC of execution infrastructure made it easy for a one-line key mismatch to hide in the fee gate for 2.5 months. The system looked sophisticated. It was just big.

### 6.3 Audit without remediation

Six methodologically correct audits. Almost none of the P0 findings were fixed before the next audit. The most critical fix (fee gate) shipped only after the loss was complete.

### 6.4 "Flatness as discipline" masked brokenness

In January 2026, the extended observation report celebrated 39 days of flat as doctrine discipline: *"The system spent the majority of 39 days flat by design. This is not a failure."* That framing was correct if the system was flat by design. It was wrong when the system was flat because the fee gate was vetoing everything. The framing delayed diagnosis by weeks.

### 6.5 The replay divergence is not a tuning problem

69% of orders trading the opposite direction from the certified replay is not a threshold mismatch. It requires tracing the exact point where the signal generation logic diverges between the two code paths and resolving it before live trading means anything.

---

## Part 7: What Must Happen Before Trading Resumes

### Immediate — do these first, in this order

| # | Action | Est. effort |
|---|--------|-------------|
| 1 | Run the Spearman ρ test — populate the blank tables in the causality audit | 1 day |
| 2 | Trace and resolve the replay ↔ live direction divergence | 2–4 days |
| 3 | Fix NAV desync (`nav_confirmed` unconditional write + `sources_ok` flag) | 2 hours |
| 4 | Add PID lock at executor startup | 30 min |
| 5 | Atomic peak state write (`tempfile` + `os.replace`) | 15 min |
| 6 | Persist TWAP slice state to disk | 2 hours |

### Before scaling capital

| # | Action |
|---|--------|
| 7 | Build minimal backtest harness (fixed seed, no lookahead, actual signal code) |
| 8 | Remove all disabled modules from executor hot path (~7,000–8,000 LOC) |
| 9 | Promote `expectancy_bridge.py` to replace true-edge heuristic |
| 10 | Write tests for the 38+ untested risk gates |
| 11 | Remove `FAIL_CLOSED_ON_NAV_STALE` env override |

### Before adding any new strategy or capital

- Prove signal ρ > 0.15 with p < 0.05 on the new entry surface
- Define falsification criteria *before* deployment, not during post-mortem
- Paper trade for ≥ 50 episodes before committing capital

---

## Summary

The fund lost 10% in six months because:

1. A fee gate read from an empty dict key for ~2.5 months, blocking all trades
2. The upstream edge computation always returned zero regardless
3. The system was deployed without ever proving the signal predicts returns
4. The architecture accumulated 72,000 LOC before any of the foundational questions were answered
5. The conviction engine was economically falsified and disabled, with no replacement
6. The current replay validation shows 69% opposite-direction signals from the live executor

**The documentation is excellent. The audits are methodologically sound. The remediation didn't happen.**

Do not resume live trading until the signal causality test returns a verdict and the replay divergence is traced to its root. Both are questions that can be answered in days with the tooling that already exists in this repository.

---

## File Citations

### Executor and Core Logic
- `execution/executor_live.py` — fee gate read path (L~4500-4600), main loop (~6,747 LOC)
- `execution/doctrine_kernel.py` — doctrine gating logic
- `execution/true_edge.py` — broken edge heuristic (confidence - 0.5 threshold)
- `execution/expectancy_bridge.py` — empirical replacement (not promoted)

### Signal Generation
- `execution/signal_generator.py:342` — actual `expected_edge` write location
- `execution/signal_screener.py:1306` — actual `hybrid_components.expectancy` write location

### Audits and Analysis
- `docs/OVER_ARCHITECTURE_AUDIT_2026-03-18.md` — component classification (SIGNAL/TRANSFORM/CONTROL/DEAD)
- `docs/SIGNAL_OUTCOME_CAUSALITY_AUDIT_2026-03-18.md` — blank Spearman ρ test results
- `docs/v8_phase_5_failure_postmortem.md` — 69.3% direction mismatch findings (2026-05-15)

### State Files
- `logs/state/nav_state.json` — unconditional write
- `logs/state/nav_confirmed.json` — conditional write (current halt cause)
- `logs/state/positions_state.json` — live position snapshot
- `logs/state/diagnostics.json` — system health

### Config
- `config/risk_limits.json` — per-symbol caps, global limits
- `config/runtime.yaml` — trading window, signal gates
- `config/strategy_config.json` — universe, per_trade_nav_pct

---

**Report prepared:** 2026-05-19
**Evaluator:** Claude (Comprehensive Post-Mortem Analysis)
