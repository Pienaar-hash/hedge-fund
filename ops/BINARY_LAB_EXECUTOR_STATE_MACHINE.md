# Binary Lab Executor State Machine

**Status:** Draft implementation spec  
**Module:** `execution/binary_lab_executor.py`  
**Purpose:** Deterministic governance enforcement for `binary_lab_state.json`

## Scope

This state machine enforces:

- activation gating,
- 15m horizon lock,
- freeze/hash integrity,
- position-rule violations,
- kill-line termination,
- day-window completion.

It is a pure reducer:

```text
state + event + limits -> next_state + actions
```

No exchange API calls. No discretionary logic.

## State Surface

Output aligns with `v7_manifest.json` `binary_lab_state` contract:

- `sleeve_id`
- `status`: `DISABLED | NOT_DEPLOYED | ACTIVE | TERMINATED | COMPLETED`
- `day`, `day_total`
- `capital.current_nav_usd`, `capital.pnl_usd`
- `kill_line.distance_usd`, `kill_line.breached`
- `metrics.total_trades`, `metrics.wins`, `metrics.losses`, `metrics.win_rate`, `metrics.by_conviction_band`
- `rule_violations`
- `freeze_intact`
- `config_hash`
- `last_checkpoint_utc_date`

## Events

- `ACTIVATE`
- `ROUND_CLOSED`
- `DAILY_CHECKPOINT`
- `TERMINATE`

## Transition Rules

### 1) Activation

`NOT_DEPLOYED -> ACTIVE` only if all pass:

- activation gate is `GO`,
- requested horizon equals locked horizon (`15m`),
- config hash provided and locked,
- mode checks:
  - `PAPER`: prediction datasets cannot be `REJECTED`,
  - `LIVE`: requires `P2_PRODUCTION` and both datasets `PRODUCTION_ELIGIBLE`.

`DISABLED` is runtime fail-closed state (limits missing/hash proof missing/hash mismatch).

### 2) Round closed

When `trade_taken=true`, update trades/PnL/band stats, then enforce:

- max concurrent (`position_rules.max_concurrent`),
- per-round size exact match (`capital.per_round_usd`),
- no same-direction stacking,
- no martingale,
- no size escalation after wins.

Any rule breach:

- increments `rule_violations`,
- sets `status=TERMINATED`,
- emits `TERMINATE_IMMEDIATELY`, `CLOSE_ALL_POSITIONS`.

### 3) Kill-line

Evaluated after trade and at checkpoint:

- `kill_nav_usd`,
- `sleeve_drawdown_usd`,
- `sleeve_drawdown_pct`.

Any breach terminates immediately.

### 4) Daily checkpoint

While `ACTIVE`:

- increments day counter only once per UTC day,
- verifies config hash unchanged,
- enforces freeze and contamination flags.

If `day >= day_total` and still active:

- set `status=COMPLETED`,
- emit `WINDOW_COMPLETE`.

Checkpoint dedup key is persisted in reducer state:

- `last_checkpoint_utc_date` (`YYYY-MM-DD`)
- duplicate checkpoints on the same UTC day are reducer no-ops for `day`.

### 5) Terminal behavior

From `TERMINATED` or `COMPLETED`, all future events are rejected with `terminal_state`.

## Determinism Guarantees

- No wall-clock calls inside decision logic.
- No hidden mutable globals.
- All outcomes are functions of explicit inputs.
- Same input sequence always yields identical state sequence.

## Test Coverage

Unit tests: `tests/unit/test_binary_lab_executor_state_machine.py`

Covered cases:

- paper activation allowed under observe-only datasets,
- live activation blocked without phase/dataset promotion,
- horizon mismatch rejection,
- PnL/metrics updates,
- kill-line immediate termination,
- max-concurrent violation termination,
- config-hash mismatch freeze break + termination,
- day-window completion.
