# Probability-First Mapping Audit — 2026-03-18

## Scope

This audit rewrites the current futures decision stack into a probability-first system with the form:

`features -> calibrated probability -> thresholded decision`

The current stack contains three distinct layers:

1. `signal_screener` ranks intents by hybrid score, then adds router-quality filters and conviction outputs before emission. `signal_screener.py` explicitly loads hybrid score results, filters on router quality, and computes `conviction_score`, `conviction_band`, and `conviction_size_multiplier`.【F:execution/signal_screener.py†L1245-L1255】【F:execution/signal_screener.py†L1288-L1295】【F:execution/signal_screener.py†L1324-L1347】
2. `conviction_engine` converts `hybrid_score`, `expectancy_alpha`, `router_quality`, `trend_strength`, volatility regime, drawdown state, and risk mode into a scalar `conviction_score`, then maps that score into discrete bands and size multipliers.【F:execution/conviction_engine.py†L4-L13】【F:execution/conviction_engine.py†L100-L116】【F:execution/conviction_engine.py†L195-L236】
3. `doctrine_kernel` and `alpha_router` apply hard permission and allocation rules keyed off regime labels, regime confidence floors, blocked execution regimes, and regime-based multipliers.【F:execution/doctrine_kernel.py†L53-L79】【F:execution/doctrine_kernel.py†L87-L100】【F:execution/alpha_router.py†L477-L519】【F:execution/alpha_router.py†L527-L616】

Hydra adds multi-head intent generation, per-head budgets, conflict resolution, and merged routing, but not a shared calibrated probability surface.【F:execution/hydra_engine.py†L13-L18】【F:execution/hydra_engine.py†L149-L176】【F:execution/hydra_engine.py†L198-L220】

## 1. Minimal prediction target

The minimal target should be a binary event on a fixed horizon.

### Primary target

For symbol `s` at decision time `t` and holding horizon `H` bars:

- `Y_long(s,t,H) = 1{ r_{t->t+H}(s) - cost_t(s) > 0 }`
- `p_long(s,t,H) = P(Y_long = 1 | x_t)`

where:

- `r_{t->t+H}(s) = ln(P_{t+H} / P_t)`
- `cost_t(s)` is total expected execution + funding + fee drag in return units
- `x_t` is the full feature vector available at decision time

### Derived short target

If the system allows both directions, define a symmetric target:

- `Y_short(s,t,H) = 1{ -r_{t->t+H}(s) - cost_t(s) > 0 }`
- `p_short(s,t,H) = P(Y_short = 1 | x_t)`

### Decision statistic

The trade/no-trade rule should not use raw price direction labels. It should use expected edge:

- `edge_long = p_long * G_long - (1 - p_long) * L_long`
- `edge_short = p_short * G_short - (1 - p_short) * L_short`

with `G_*` and `L_*` estimated from realized conditional payoff distributions under the same horizon `H`.

### Minimum viable implementation

If only one scalar prediction target is desired, use:

- `p_up = P(r_{t->t+H} > 0 | x_t)`

and convert to tradable probability after cost adjustment:

- `p_long = P(r_{t->t+H} > cost_t | x_t)`
- `p_short = P(r_{t->t+H} < -cost_t | x_t)`

## 2. Probability-first architecture

## 2.1 Feature vector

Every current input must become a numeric feature in `x_t`.

Recommended minimum feature set:

- `f_hybrid` = existing hybrid score input before thresholding
- `f_expectancy` = existing expectancy alpha input
- `f_router_q` = router quality score
- `f_trend` = continuous trend-strength feature
- `f_vol` = volatility state encoded as numeric values, not labels
- `f_dd` = drawdown fraction
- `f_risk` = risk-state numeric indicators
- `f_head_k` = Hydra head score for head `k`
- `f_exec` = execution quality metrics such as slippage and spread state
- `f_regime_probs[j]` = class probabilities from the regime model, if retained

The DLE prediction spec already uses a belief-aggregate architecture with explicit probabilities and deterministic constraints, which is directly aligned with this feature-to-probability framing.【F:docs/dle/DLE_PREDICTION_LAYER_SPEC.md†L15-L47】【F:docs/dle/DLE_PREDICTION_LAYER_SPEC.md†L53-L80】【F:docs/dle/DLE_PREDICTION_LAYER_SPEC.md†L174-L199】

## 2.2 Probability model

Train a single calibrated binary model per direction and horizon:

- `p_long_raw = model_long(x_t)`
- `p_short_raw = model_short(x_t)`

Then calibrate:

- `p_long = Cal_long(p_long_raw)`
- `p_short = Cal_short(p_short_raw)`

The existing `binary_lab_s2_model` already demonstrates the preferred pattern: preserve baseline output, then apply explicit isotonic calibration once enough observations exist, and report calibration state rather than hiding it.【F:execution/binary_lab_s2_model.py†L2-L24】【F:execution/binary_lab_s2_model.py†L117-L169】【F:execution/binary_lab_s2_model.py†L217-L237】

## 2.3 Decision rule

Decision logic becomes:

- enter long iff `p_long >= tau_long`
- enter short iff `p_short >= tau_short`
- size is a deterministic function of `p` or `edge`
- vetoes are only allowed if they are equivalent to threshold checks on probability, expected loss, or execution-failure probability

A minimal numeric rule is:

- `enter_long = 1{ p_long >= tau_long and q_exec >= tau_exec and c_loss <= tau_loss }`
- `enter_short = 1{ p_short >= tau_short and q_exec >= tau_exec and c_loss <= tau_loss }`

where:

- `q_exec = 1 - P(execution_failure | x_t)`
- `c_loss = P(portfolio_constraint_breach | x_t)`

## 2.4 Sizing rule

Remove banded sizing and use a monotone function:

- `size_frac = clip(k * (p - tau_entry) / (1 - tau_entry), 0, size_max)`

or, with payoff asymmetry:

- `size_frac = clip(k * edge / L_ref, 0, size_max)`

This replaces discrete conviction bands with a continuous mapping from calibrated probability or expected edge.

## 3. Mapping current components into probability space

## 3.1 Regime classification

Current behavior:

- `doctrine_kernel` uses fixed labels such as `TREND_UP`, `TREND_DOWN`, `MEAN_REVERT`, `BREAKOUT`, `CHOPPY`, and `CRISIS` to permit or forbid directions and to apply confidence/stability floors.【F:execution/doctrine_kernel.py†L53-L79】【F:execution/doctrine_kernel.py†L141-L163】
- `alpha_router` converts the Sentinel-X primary regime label into fixed allocation multipliers such as `1.05`, `0.90`, `0.85`, and `0.60`.【F:execution/alpha_router.py†L477-L519】

Probability-first role:

- **Probability estimation input**, if and only if the regime model emits a full probability vector.
- **Noise**, if only hard labels are retained.
- **Decision thresholding**, only for model-quality controls such as stale-state checks; not for directional authorization.

Required replacement:

- Replace `primary_regime` as an action controller with `regime_prob_vector`:
  - `z_regime = [P(C_1|x_t), ..., P(C_m|x_t)]`
- Feed `z_regime` into the directional probability model.
- Remove hard rules such as `CHOPPY -> no trade` and instead learn whether `P(Y>0|x_t)` falls below the entry threshold when the regime posterior places mass on that state.

Minimal acceptable fallback if a regime classifier must stay:

- keep only `regime_probs`
- delete `primary_regime` from entry authorization
- retain `state_stale` as a data-quality veto

## 3.2 Conviction score

Current behavior:

- `conviction_engine` is already a scalar in `[0,1]`, but it is not defined as an event probability; it is a deterministic mixture that is later discretized into bands and size multipliers.【F:execution/conviction_engine.py†L4-L13】【F:execution/conviction_engine.py†L100-L116】【F:execution/conviction_engine.py†L195-L236】
- `signal_screener` computes the conviction output after ranking and uses the result to satisfy downstream minimum-entry-band logic.【F:execution/signal_screener.py†L1324-L1347】

Probability-first role:

- **Noise** in its current semantic form.
- **Calibrated transformation candidate** only if it is empirically mapped to a realized event frequency.

Required replacement:

- Delete `conviction_band` entirely.
- Replace `conviction_score` with either:
  1. `p_long` or `p_short`, if the scalar is meant to decide entry, or
  2. `q_exec = 1 - P(execution_failure)` if the scalar is meant to decide size attenuation due to routing quality.

If migration must be staged, calibrate the existing scalar:

- estimate `g(c) = P(Y_long=1 | conviction_score = c)`
- use isotonic calibration: `p_conv = g(conviction_score)`
- deprecate direct use of raw `conviction_score`

That turns conviction from a narrative summary into a measurable probability surface.

## 3.3 Hydra outputs

Current behavior:

- Hydra defines per-head intents with `score`, `nav_pct`, `priority`, and conflict-resolution logic across six heads, then merges them into a unified order stream.【F:execution/hydra_engine.py†L13-L18】【F:execution/hydra_engine.py†L55-L115】【F:execution/hydra_engine.py†L149-L176】【F:execution/hydra_engine.py†L198-L220】

Probability-first role:

- **Probability estimation input** when head outputs are treated as numeric predictors.
- **Decision thresholding** when budgets cap exposure ex post.
- **Noise** when `priority` and qualitative rationale are used as directional authority.

Required replacement:

For each head `k`:

- replace `head_score_k` with `p_k = P(Y=1 | x_t, head=k)` or with a calibrated likelihood contribution
- replace `priority_k` with either:
  - a portfolio budget cap outside the prediction model, or
  - nothing
- replace qualitative rationale strings with logged numeric feature contributions

Recommended aggregation:

- `logit(p_final) = b + sum_k w_k * logit(p_k)`

with weights `w_k` fit on realized outcomes.

Alternative conservative aggregation:

- stack head outputs as features in a meta-model and calibrate the final probability directly

## 3.4 Veto layers and routing logic

Current behavior:

- `doctrine_kernel` issues hard `ALLOW`/`VETO_*` decisions from regime and execution-state conditions.【F:execution/doctrine_kernel.py†L87-L100】
- `signal_screener` drops intents on router-quality thresholds and other filters before conviction is attached.【F:execution/signal_screener.py†L1288-L1295】
- `alpha_router` turns health, router quality, drawdown, and regime state into multiplicative allocation penalties rather than explicit risk probabilities.【F:execution/alpha_router.py†L527-L616】

Probability-first role:

- **Decision thresholding** for genuine risk controls.
- **Noise** when qualitative labels substitute for measured probabilities.

Required replacement:

Decompose vetoes into explicit numeric probabilities:

- `q_exec = 1 - P(order_reject or unfillable_slippage | x_t)`
- `q_live = 1 - P(state_stale or missing_inputs | x_t)`
- `q_risk = 1 - P(portfolio_breach | x_t)`

Final admissibility rule:

- `enter = 1{ p_trade >= tau_trade and q_exec >= tau_exec and q_live >= tau_live and q_risk >= tau_risk }`

This preserves fail-closed behavior while making every decision threshold numeric.

## 4. Simplified target system

## 4.1 Online inference graph

1. Build numeric feature vector `x_t`.
2. Produce raw directional probabilities:
   - `p_long_raw = model_long(x_t)`
   - `p_short_raw = model_short(x_t)`
3. Calibrate:
   - `p_long = Cal_long(p_long_raw)`
   - `p_short = Cal_short(p_short_raw)`
4. Produce auxiliary control probabilities:
   - `q_exec`
   - `q_live`
   - `q_risk`
5. Apply threshold rule:
   - long if `p_long >= tau_long` and all control probabilities exceed their thresholds
   - short if `p_short >= tau_short` and all control probabilities exceed their thresholds
6. Size from `p` or expected edge.
7. Log every component.

## 4.2 Minimal architecture

### Inputs

- market features `x_market`
- execution features `x_exec`
- portfolio features `x_port`
- optional regime posterior `x_regime_probs`
- optional Hydra head posterior features `x_hydra`

### Model layer

- `p_long = Cal_long(Model_long([x_market, x_exec, x_port, x_regime_probs, x_hydra]))`
- `p_short = Cal_short(Model_short([x_market, x_exec, x_port, x_regime_probs, x_hydra]))`
- `q_exec = Cal_exec(Model_exec(x_exec))`
- `q_risk = Cal_risk(Model_risk(x_port))`

### Decision layer

- `long = 1{ p_long >= tau_long and q_exec >= tau_exec and q_risk >= tau_risk }`
- `short = 1{ p_short >= tau_short and q_exec >= tau_exec and q_risk >= tau_risk }`
- `flat = 1 - max(long, short)`

### Sizing layer

- `size_long = clip(k_long * (p_long - tau_long), 0, size_long_max)`
- `size_short = clip(k_short * (p_short - tau_short), 0, size_short_max)`

### Logging layer

For every candidate:

- `p_long_raw`
- `p_long`
- `p_short_raw`
- `p_short`
- `q_exec`
- `q_risk`
- thresholds used
- resulting action
- realized outcome label after horizon `H`

## 5. Mapping table

| Current component | Current numeric object | New role | Action |
|---|---:|---|---|
| `primary_regime` / `secondary_regime` | categorical label | remove from direct authorization | replace with regime posterior features `regime_probs[j]` |
| `regime confidence floor` | scalar threshold | decision thresholding only if interpreted as data-quality probability | convert to `q_live >= tau_live` or remove |
| `cycles_stable` | integer | feature only | include as numeric feature if predictive; otherwise remove |
| doctrine `REGIME_DIRECTION_MAP` | label-to-direction map | noise | remove from entry authorization |
| doctrine `VETO_*` regime rules | binary gate | decision thresholding | rewrite as numeric threshold checks on `q_live`, `q_exec`, or `q_risk` |
| `hybrid_score` | scalar | probability estimation input | feed into model as feature or calibrate directly |
| `expectancy_alpha` | scalar | probability estimation input | feed into model as feature |
| `router_quality` | scalar | probability estimation input and control probability input | use in `p_trade` and `q_exec` models |
| `trend_strength` | scalar | probability estimation input | feed into directional model |
| `conviction_score` | scalar in `[0,1]` | replace | map to calibrated probability or delete |
| `conviction_band` | ordinal bucket | remove | delete and replace with continuous sizing from `p` or `edge` |
| `conviction_size_multiplier` | scalar multiplier | decision sizing | replace with monotone `size(p)` |
| Hydra head `score` | scalar in `[0,1]` | probability estimation input | calibrate each head score or use as meta-model feature |
| Hydra head `priority` | integer | remove from alpha authority | keep only as portfolio scheduling metadata if needed |
| Hydra head `nav_pct` | scalar | post-decision sizing cap | keep outside prediction model |
| Hydra merged-intent side | categorical | remove as authority primitive | derive from `argmax(p_long, p_short)` |
| alpha-router regime factor | scalar multiplier | decision sizing or remove | move to `q_risk` or `size(p)` if empirically justified |
| drawdown / risk-mode overrides | scalar caps | decision thresholding | convert to `q_risk` threshold or exogenous risk cap |
| stale-state checks | binary gate | decision thresholding | keep as `q_live >= tau_live` |

## 6. Recommended migration sequence

1. Freeze horizon `H` and realized label definitions.
2. Log current features plus realized outcomes for every emitted and vetoed candidate.
3. Fit baseline `p_long_raw` and `p_short_raw` models.
4. Calibrate with isotonic or Platt scaling.
5. Backfill `p_trade` next to current conviction outputs.
6. Replace `conviction_band` thresholds with `p_trade` thresholds.
7. Convert vetoes one-by-one into `q_exec`, `q_live`, and `q_risk` thresholds.
8. Retire regime labels from direct entry authorization after calibration parity is verified.

## 7. Bottom line

The irreducible production system is:

- `x_t -> p_long, p_short, q_exec, q_risk -> threshold rule -> size`

Everything else must justify itself as either:

- a feature contributing to calibrated probability estimation, or
- a numeric threshold on a control probability.

If it cannot do one of those two jobs, it should be removed.
