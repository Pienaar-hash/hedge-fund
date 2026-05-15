# V8 Shadow Soak Observer Runbook

## Status

Phase 5 is terminated with a hard-fail verdict.

Phase 6 is denied. No live authority increase is permitted.

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

This document is now an archival runbook for the completed Phase 5 observer.

The observer remains research-only and non-gating, but observation is stopped for this configuration.

## Observation loop status

Do not run the Phase 5 daily observer loop.

```bash
# prohibited: bin/run-shadow-soak-observer.sh
```

Manual inspection of already-written append-only logs is permitted for research-only analysis.

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

Historical reference (during the prior 14-day observation design), the state could be:

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

## 14-day gate (superseded)

Do not discuss live activation until 14 days of real observation data exist.

PASS requires the Phase 5 criteria in `docs/v8_phase5_shadow_soak_spec.md`.

This gate is superseded by the Phase 5 hard-fail postmortem in `docs/audits/V8_PHASE5_FAILURE_POSTMORTEM.md`.

## Next valid work (research-only)

1. Identify strategy divergence point between replay and live signal generation.
2. Run side-by-side replay vs live trace on identical inputs for a single symbol and hour.
3. Capture the first divergence event (signal, regime, doctrine decision, side mapping).
4. Keep executor, doctrine, risk, and live authority unchanged until root cause is documented.

Initial trace helper:

```bash
python -m research.phase5_divergence_trace \
  --logs-dir logs \
  --replay-dir data/replay_certifications/<run_id>
```

## Rollback / disable

Do not delete append-only logs.

If archival disable actions are needed:

1. Stop running `bin/run-shadow-soak-observer.sh`
2. Archive `logs/research/shadow_soak_events.jsonl` to `logs/research/archive/`
3. Write `logs/state/shadow_soak_state.json` with:

   * `status=PAUSED`
   * `reason=operator_disabled`
4. Leave executor unchanged
