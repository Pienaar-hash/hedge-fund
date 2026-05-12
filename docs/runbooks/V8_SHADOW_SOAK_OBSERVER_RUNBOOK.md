# V8 Shadow Soak Observer Runbook

## Status

Manual observation only.

This runbook does not authorize:
- cron
- Supervisor
- runtime restart
- execution changes
- doctrine changes
- risk-limit changes
- conviction re-enable

Current sealed runner commit:

`4879577623e4e7cbffc052dac16e274bb0794e6b`

## Purpose

Run the V8 Phase 5 shadow soak observer once per day and inspect the read-only state outputs.

The observer is research-only. It writes append-only observation events and a state snapshot. It does not gate execution.

## Daily manual command

From repo root:

```bash
bin/run-shadow-soak-observer.sh
```

## Inspect state

```bash
cat logs/state/shadow_soak_state.json
tail -n 20 logs/research/shadow_soak_events.jsonl
```

## Expected outputs

```text
logs/research/shadow_soak_events.jsonl
logs/state/shadow_soak_state.json
```

## Acceptable interim states

During the 14-day observation period, the state may be:

```text
PENDING
CONDITIONAL
PAUSED
FAILED
```

A non-PASS state is not itself a trading-system failure. It means the observer has not produced enough qualifying evidence or has detected a review condition.

## Hard stops

Do not proceed to Phase 6 if any of the following are present:

* `catastrophic_mismatch_count > 0`
* direction mismatch
* symbol mismatch
* unstable output hash
* missing/corrupt source logs for two consecutive checks
* median absolute slippage error above 3 bps at gate review
* p95 absolute slippage error above 10 bps at gate review

## 14-day gate

Do not discuss live activation until 14 days of real observation data exist.

PASS requires the Phase 5 criteria in `docs/v8_phase5_shadow_soak_spec.md`.

## Rollback / disable

Do not delete append-only logs.

To disable the observer:

1. Stop running `bin/run-shadow-soak-observer.sh`
2. Archive `logs/research/shadow_soak_events.jsonl` to `logs/research/archive/`
3. Write `logs/state/shadow_soak_state.json` with:

   * `status=PAUSED`
   * `reason=operator_disabled`
4. Leave executor unchanged