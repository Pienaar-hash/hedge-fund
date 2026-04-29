# P6 Branch A Containment Decision — Futures Conviction Authority Suspended

**Decision ID:** `DEC_P6_BRANCH_A_CONTAINMENT_V1`
**Date:** 2026-04-11
**Status:** BINDING
**Applies to:** All futures entry surfaces (conviction, simple rules, price-state)
**Does not apply to:** PM Sleeve / binary structures, futures execution infrastructure (preserved)

---

## 1. Decision

**Futures conviction authority is suspended.**

Futures remains active only as research, replay, and audit infrastructure until a new entry surface demonstrates fee-clearing expectancy in shadow under the existing bridge and fee gate.

This decision is the outcome of the P6B.5 historical replay (2026-04-11), which conclusively demonstrated that no candidate surface produced a fee-clearing futures band.

---

## 2. Evidentiary Basis

### 2.1 Replay Run

| Parameter | Value |
|---|---|
| Run timestamp | 2026-04-11T12:57:03Z |
| Git commit | `a779cc7a` |
| Episodes | 815 total, 800 in-universe (BTC/ETH/SOL) |
| Records | 1,712 (1,358 signals, 339 NO_SIGNAL, 354 skipped) |
| Klines | 33,633 bars (3 symbols × 11,211 × 15m) |
| Warmup | 60 bars |
| Fee threshold | 0.12 (bridge convention: raw fraction) |
| Manifest | `logs/state/p6_replay_run.json` (SHA-256 hashes of all frozen inputs) |

### 2.2 All Six Candidates Failed Fast-Fail

| candidate_id | n | pass_rate | mean_expected | mean_realized | ρ | fast_fail_reason |
|---|---|---|---|---|---|---|
| C1_MR_NORMAL | 196 | 0.0% | -0.257% | -0.126% | +0.257 | no_fee_band, pass_rate_zero |
| C1_MR_INVERTED | 196 | 0.0% | -0.257% | -0.126% | +0.257 | no_fee_band, pass_rate_zero |
| C1_TREND_NORMAL | 23 | 0.0% | -0.458% | -0.451% | -0.094 | no_fee_band, ρ≤0, pass_rate_zero |
| C1_TREND_INVERTED | 23 | 0.0% | -0.458% | -0.451% | -0.094 | no_fee_band, ρ≤0, pass_rate_zero |
| C2_REGION_NORMAL | 460 | 0.0% | -0.837% | -0.214% | +0.192 | no_fee_band, pass_rate_zero |
| C2_REGION_INVERTED | 460 | 0.0% | -0.837% | -0.214% | +0.192 | no_fee_band, pass_rate_zero |

**Outcome:** `BRANCH_A_CONTAINMENT` — 0 survivors, 0 promoted.

### 2.3 Chain of Failed Rescue Attempts

| Attempt | Surface | Result |
|---|---|---|
| Original | Hybrid conviction surface | No fee-clearing edge |
| P4 | Empirical expectancy bridge | No rescue |
| P4 | Regime conditioning | No rescue |
| P6A | Simple rules (C1) | No rescue |
| P6A | Price-state surface (C2) | No rescue |
| P6B.5 | Full historical replay (all 6 candidates) | **Conclusive: no fee-clearing band** |

---

## 3. Key Findings

### 3.1 C1_MR: Weak Sorting, Insufficient Depth

C1_MR is the only surface with real ordering information (ρ = +0.257). Best realized band (0.75-0.80) reached +0.046%, but this is still below fee friction. The problem is not ranking — it is that the surface cannot produce enough edge to survive friction.

### 3.2 C1_TREND: Dead

23 signals, negative ρ, negative best-band edge. Insufficient sample, wrong direction. No ambiguity.

### 3.3 C2_REGION: Broad but Flat-Negative

Widest coverage (460 signals) but near-constant conviction clustering at 0.60-0.65. Both sufficient bands negative. Price-state alone, in this futures context, does not create a causal positive region the way PM Sleeve does in binary markets.

### 3.4 Inversion Hypothesis Retired

Normal and inverted pairs produced identical bridge outcomes. Polarity is not the bottleneck. The inversion hypothesis is retired as a practical path.

---

## 4. What Is Preserved

The following infrastructure is preserved exactly as-is:

| Component | Status | Reason |
|---|---|---|
| DLE / shadow governance | Preserved | Proven falsification infrastructure |
| Expectancy bridge | Preserved | Validated lookup and band mechanics |
| Opportunity surface | Preserved | Research and audit |
| P6 replay harness | Preserved | Deterministic episode replay |
| Fee gate | Preserved | Core discipline gate |
| Manifest discipline | Preserved | Reproducibility |
| Doctrine kernel | Preserved | Entry/exit authority (no futures entries permitted) |
| Risk limits | Preserved | Secondary veto |

---

## 5. What Is Prohibited

The following actions are prohibited without a new decision note superseding this one:

1. **Fee threshold loosening** — the threshold is established and tested
2. **Conviction remapping** — local tuning is curve-fitting, not discovery
3. **Region rebucketing** — same rationale
4. **New candidate families (P6.6+)** — more rule families cannot rescue a surface that is structurally non-economic
5. **Silent resurrection** — any futures entry authority must demonstrate fee-clearing expectancy in shadow under the existing bridge and fee gate before activation

---

## 6. Futures Operational Status

| Surface | Status |
|---|---|
| Futures legacy/hybrid conviction | **Retired from entry authority** |
| Futures execution stack | Preserved (research mode) |
| Futures diagnostics / replay | Preserved |
| P6 simple rules | **Falsified** |
| P6 price-state surface | **Falsified** |
| Branch classification | **A — Containment** |

---

## 7. Strategic Redirect

Active strategy effort should redirect to PM Sleeve / binary structures, which remain the only surface in the system family with demonstrated structural promise:

- Region-first entry logic
- Payoff asymmetry
- Positive paper behavior
- Causal logic not dependent on conviction ranking

This does not mean immediate promotion. It means the center of gravity for live strategy research moves there.

---

## 8. Resurrection Criteria

Futures entry authority may be restored only if:

1. A new entry surface (not a modification of C1/C2) demonstrates fee-clearing expectancy
2. Demonstrated in shadow mode under the existing bridge and fee gate
3. Replayed against the same episode ledger (or a strictly larger one)
4. Documented in a new decision note that supersedes this one
5. Passes the same fast-fail and promotion gates used in P6B.5

---

## 9. Artifacts

| Artifact | Path |
|---|---|
| Replay manifest | `logs/state/p6_replay_run.json` |
| Replay summary | `logs/state/p6_replay_summary.json` |
| Replay tables (CSV) | `logs/state/p6_replay_tables.csv` |
| Replay signals (JSONL) | `logs/execution/p6_replay_signals.jsonl` |
| Kline cache | `logs/state/p6_kline_cache.json` |
| P6 simple rules | `execution/p6_simple_rules.py` |
| P6 price-state | `execution/p6_price_state.py` |
| P6 shadow evaluator | `execution/p6_shadow_evaluator.py` |
| P6 replay pipeline | `execution/p6_replay.py` |
| Expectancy bridge | `execution/expectancy_bridge.py` |
| Bridge tables | `logs/state/expectancy_bridge.json`, `logs/state/expectancy_bridge_regime.json` |

---

**This decision is binding. No futures entry authority is permitted until superseded by a new decision note meeting the resurrection criteria above.**
