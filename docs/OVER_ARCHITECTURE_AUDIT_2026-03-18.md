# Over-Architecture Detection Audit — 2026-03-18

## Scope

This audit answers a narrow question: which layers in the futures trading stack are **actually producing edge**, versus merely reshaping, governing, or operationalizing existing edge.

The components explicitly reviewed are:

1. Regime model (`Sentinel-X`)
2. Conviction scoring (`conviction_engine`)
3. Veto logic (`doctrine_kernel` and veto paths)
4. Routing layer (`order_router`, plus disabled allocation/router overlays)

## Definition: signal-producing component

A component qualifies as **SIGNAL** only if it can be shown, in isolation, to improve at least one of the following on held-out or post-deployment data:

- **Prediction calibration**: lower Brier score or log loss for a forecast that the system actually acts on.
- **Edge monotonicity**: higher model score / band / bucket maps to higher realized expectancy in a monotonic or near-monotonic way.
- **Realized PnL after friction**: improvement survives fees, slippage, spread, and routing behavior.

If a module only converts an upstream score into bands, sizing multipliers, hard permits, or execution tactics, it is **not SIGNAL by default**. It must prove incremental contribution over a simpler baseline.

## Evidence standard used

I classified each component using the strictest standard available in the repository:

- **Direct evidence present** if the repo contains explicit measurement hooks or tests tied to the component's claimed function.
- **Indirect evidence only** if the component is richly implemented but lacks an attributable scorecard.
- **No evidence** if the component is disabled, shadow-only, or has no isolation path.

## Component classification table

| Component | What it does in this repo | Calibration impact (Brier / log loss) | Edge monotonicity impact | Realized PnL after friction impact | Classification | Why |
|---|---|---|---|---|---|---|
| **Regime model** | Sentinel-X converts market features into regime probabilities and hard labels, then doctrine consumes those labels for entry/exit permission. | **No direct proof in repo.** Sentinel-X emits probabilities, but there is no attached Brier/log-loss evaluation for those regime probabilities. | **Unproven directly.** The repo has monotonicity tooling for Hydra scores, not for regime buckets. | **Unproven directly.** Regime gates clearly affect trading behavior, but no isolated ablation shows net PnL improvement versus simpler gating. | **TRANSFORM** | It is a market-state abstraction layer, but the current repo evidence does not prove incremental predictive edge beyond re-encoding price/volatility structure. |
| **Conviction scoring** | Deterministic sizing transform over upstream factors: `hybrid_score`, `expectancy_alpha`, `router_quality`, `vol_regime`, DD/risk mode, and optional overlays. | **No.** No scoring-rule evaluation exists for conviction bands. | **Partially instrumented, but indirect.** Monotonicity tooling exists for Hydra scores, not for conviction bands themselves. | **Possible impact, but not isolated.** Conviction changes size and can veto when router quality is poor, yet the repo shows no clean ablation proving better net PnL than simpler size clamps. | **TRANSFORM** | The module explicitly describes itself as a deterministic sizing engine using existing factors, so it is downstream of signal rather than a new source of signal. |
| **Veto logic** | Doctrine kernel is the hard permit / deny layer governing entries and exits based on regime, execution quality, alpha survival, and safety conditions. | **Not applicable as alpha calibration.** Veto logic is not a probabilistic forecaster. | **No.** It does not create score ordering; it blocks actions. | **Potentially yes for tail-risk containment, but measured mostly as governance telemetry rather than causal PnL uplift.** | **CONTROL** | This is constitutional risk/governance logic. It can be essential, but it is not edge production. |
| **Routing layer** | Order router chooses maker-first / fallback / TWAP / microstructure-aware execution and tracks slippage, latency, fallback ratio, and maker-fill statistics. | **Not applicable to predictive calibration.** | **No.** Routing does not order predictions by expected return. | **Yes, at execution level.** The repo explicitly measures maker fill ratio, fallback ratio, and slippage distribution, which are friction terms. | **CONTROL** | It improves implementation quality and may improve net realized PnL, but through execution efficiency, not signal generation. |

## Per-component audit notes

### 1) Regime model — `Sentinel-X`

### What is real

`Sentinel-X` is a substantial regime engine. It extracts return, volatility, trend, breakout, mean-reversion, volume, correlation, and microstructure features; produces regime probabilities; then converts those into hard labels with smoothing and stickiness. That is a meaningful market-state encoder, not a trivial switch. However, its architecture alone is not evidence of edge. The file header itself describes the system as `Feature Extraction → ML-Style Scoring → Rule-Based Labels`, which is exactly the pattern that needs post-hoc falsification, not architectural admiration.

### What is missing

The repo does **not** show a regime-probability scorecard equivalent to Brier/log loss. There are prediction-scoring utilities elsewhere in the repo, but not attached to Sentinel-X regime outputs. There is also no explicit regime ablation showing:

- signal quality with regime gating versus no regime gating,
- simpler trend/range heuristics versus Sentinel-X,
- or net episode expectancy by regime-conditioned permit policy.

### Conclusion

Treat the regime model as **TRANSFORM**, not proven **SIGNAL**. It may be useful, but in this repo it is a **state abstraction with downstream policy effects**, not a demonstrated alpha source.

## 2) Conviction scoring — `conviction_engine`

### What is real

The conviction engine is explicitly documented as a **pure deterministic function** that computes conviction scores for sizing from existing inputs, including `hybrid_score`, `expectancy`, `router_quality`, `vol_regime`, drawdown state, and risk mode. By construction, that means it consumes upstream signal and converts it into:

- a conviction score,
- a conviction band,
- a size multiplier,
- and sometimes a veto.

This is useful plumbing if it improves implementation discipline.

### What is missing

There is no repo evidence that the conviction bands themselves improve:

- Brier score or log loss,
- bucket-level monotonicity relative to raw `hybrid_score`,
- or net PnL after friction versus a much simpler sizing policy.

Because conviction is downstream of existing factors, it is highly vulnerable to being **cosmetic complexity**: a second encoding of the same underlying edge.

### Conclusion

Classify conviction as **TRANSFORM** until an ablation proves that conviction-sized trades outperform a baseline such as:

- raw score only,
- raw score + fixed risk cap,
- or raw score + one simple router-quality clamp.

## 3) Veto logic — doctrine / permit-deny layer

### What is real

The doctrine kernel is intentionally a constitutional control plane. It enforces regime stability, confidence floors, direction compatibility, crisis overrides, execution crunch restrictions, head-budget constraints, and alpha survival floors. That is coherent **CONTROL** logic.

### What is missing

The current observability is mostly veto counting, veto attribution, and doctrine consistency. That is valuable governance telemetry, but it does not prove alpha contribution. A veto system should be judged on:

- avoided-loss efficiency,
- false-veto rate,
- missed-opportunity cost,
- and whether it improves post-friction expectancy conditional on denied trades.

The repo does not provide those causal measurements as a first-class doctrine scorecard.

### Conclusion

Veto logic is **CONTROL**, not SIGNAL. Keep only the parts that are necessary for catastrophic risk containment, stale-state refusal, and execution sanity. Any veto branch without measurable avoided-loss benefit is a pruning candidate.

## 4) Routing layer — execution router and overlays

### What is real

The order router is the cleanest case in the set: it is **not signal**, but it does have measurable execution-value instrumentation. The router implements maker-first posting, taker fallback, TWAP slicing, fee-aware effective pricing, and slippage recording. The test suite also asserts router effectiveness metrics like maker fill ratio, fallback ratio, and slippage quartiles.

That means this layer can be falsified against post-friction outcomes much more cleanly than the regime or conviction layers.

### What is missing

Even here, the evidence is execution-local, not end-to-end alpha-local. The repo shows that routing friction is measured, but not a simple causal report such as:

- net basis points saved vs immediate market execution,
- realized fill quality by route policy,
- or whether router sophistication beats a simpler maker/taker heuristic.

### Conclusion

Routing is **CONTROL**. It may be essential control, because friction is real PnL, but it is still implementation quality rather than predictive edge.

## Components that cannot be independently falsified cleanly

These are the biggest over-architecture risks.

1. **Regime model as currently wired**
   - It feeds doctrine, sizing overlays, and downstream filters simultaneously.
   - There is no attached proper-scoring evaluation for regime probabilities.
   - Its effect is therefore broad, but not isolated.

2. **Conviction scoring**
   - It is built from upstream scores plus execution/risk overlays.
   - Without controlled ablations, any improvement can be falsely attributed to raw signal quality rather than conviction itself.
   - This is the classic "re-encoding looks like intelligence" failure mode.

3. **Doctrine veto branches beyond core safety**
   - Many vetoes may be locally plausible while globally unaudited.
   - Without counterfactual replay of vetoed trades, the system cannot distinguish protection from over-blocking.

4. **Disabled overlay routers (`alpha_router`, `cerberus_router`)**
   - These are especially weak from a falsifiability perspective because they add conceptual complexity but are disabled in config.
   - A disabled component cannot currently contribute edge, so its live burden is cognitive and maintenance overhead rather than trading value.

## Candidates for removal or aggressive simplification

### High-priority removal / freeze candidates

1. **`cerberus_router`**
   - It is explicitly described as a research-only multi-strategy router.
   - It is disabled in `strategy_config.json`.
   - It adjusts downstream overlays rather than placing trades directly.
   - This is prime architectural dead weight unless there is an active experiment plan.

**Classification:** **DEAD** in the live system today.

2. **`alpha_router`**
   - Also disabled in `strategy_config.json`.
   - It is another allocation overlay stacked on top of risk and conviction.
   - With conviction already live, this is overlapping capital-allocation complexity without active contribution.

**Classification:** **DEAD** in the live system today.

### Simplification candidates inside live paths

3. **Conviction banding granularity**
   - Keep the sizing clamp concept if needed, but reduce it to the smallest testable form.
   - Example: one continuous multiplier or two-band sizing policy, instead of rich band taxonomy plus multiple overlays.

4. **Doctrine veto taxonomy expansion**
   - Preserve only vetoes that guard stale state, crisis, and hard execution/risk invalidity.
   - Any veto that has no measured avoided-loss value should be consolidated or removed.

5. **Regime complexity above a falsifiable baseline**
   - If Sentinel-X cannot beat a minimal baseline, collapse it to a simpler regime gate:
     - trend / non-trend,
     - crisis / non-crisis,
     - stale / fresh.

## Minimal viable system

If the objective is to retain only **SIGNAL + essential CONTROL**, the repo evidence supports the following minimal system:

### Keep

1. **Primary signal score**
   - Keep the upstream score that already has monotonicity and calibration instrumentation around Hydra / expectancy workflows.
   - This remains the actual candidate SIGNAL source.

2. **Minimal doctrine control**
   - Keep only essential hard controls:
     - stale data refusal,
     - crisis / kill-switch behavior,
     - hard risk caps,
     - reduce-only exit exemption,
     - exchange/execution invalid-state protection.

3. **Minimal execution router**
   - Keep the smallest routing logic that can still manage fees, spread, and slippage.
   - Prefer a stripped maker-first + fallback policy with simple TWAP only where clearly beneficial.

### Remove or disable from the viable core

1. Full `cerberus_router`
2. Full `alpha_router`
3. Rich conviction band taxonomy unless ablation proves incremental value
4. Any non-core doctrine veto branch lacking avoided-loss evidence
5. Any regime nuance that cannot beat a simple regime baseline

## Bottom line

On the evidence currently present in the repository:

- **Regime model:** **TRANSFORM**
- **Conviction scoring:** **TRANSFORM**
- **Veto logic:** **CONTROL**
- **Routing layer:** **CONTROL**

There is **no reviewed component in this list that currently clears the bar for proven SIGNAL**.

That does **not** mean the system has no edge. It means the edge appears to live **upstream of these layers**, while these layers mostly:

- translate,
- gate,
- size,
- or execute that edge.

So the over-architecture suspicion is substantially supported:

- the live stack contains several layers that are **important operationally**,
- but the repo evidence does **not** show them as independently edge-producing,
- and at least two overlay systems (`alpha_router`, `cerberus_router`) are currently best described as **dead weight in the live configuration**.
