# CARD-HEDGE-PUBTICK-BOUNDARY-HEARTBEAT-INSTRUMENTATION-001 — Result

**Status:** Implementation, tests, and documentation complete. Manual testnet verification is **pending** — it requires restarting the live executor process to load the new code, which was deliberately not done in this card without explicit operator go-ahead (see "Manual verification" below).

**Predecessor:** `docs/audits/CARD-HEDGE-LIVE-STATE-PUBLISH-FREEZE-ROOT-CAUSE-001.md` — narrowed a ~33.7-minute publication gap to two surviving hypotheses (`scheduler_or_control_flow_defect` primary, a non-`Exception` `BaseException` escape secondary) and could not distinguish them without live instrumentation at the `_pub_tick()` boundary. This card adds that instrumentation only; it does not change scheduling, cadence, or episode-ledger loading.

---

## Changed files

| File | Change |
|------|--------|
| `execution/executor_live.py` | Adds `_pub_tick_heartbeat()` writer + module state; wraps `_pub_tick()` body in `try/except Exception/finally`; adds `loop_id` parameter to `_pub_tick()`; emits `CALL_REACHED` at the call site; instruments the episode-ledger rebuild sub-block with `LEDGER_REBUILD_STARTED`/`LEDGER_REBUILD_COMPLETED` |
| `tests/integration/test_pub_tick_heartbeat.py` | New — 8 tests covering event ordering, ledger-due/not-due paths, ordinary-exception path, `BaseException` abort path, telemetry-sink-failure fail-open behavior, bounded event count, schema/duration fields, and `last_completed_*` propagation across calls |
| `tests/integration/test_executor_state_files.py` | Existing `test_pub_tick_writes_state` now monkeypatches `_PUB_TICK_HEARTBEAT_LOG` to a `tmp_path` logger — this repo's `logs/` directory is the live executor's real log directory, not a test fixture, and must not receive test writes |
| `v7_manifest.json` | Registers `pub_tick_heartbeat` under the append-only JSONL execution logs section, with its `event_types` list |
| `EXECUTION_ARCHITECTURE.md` | Adds `pub_tick_heartbeat.jsonl` to the execution-logs table and a new "Publication Heartbeat Diagnostics" section documenting event meanings and the diagnostic procedure for a future gap |
| `docs/active/TESTING.md` | Adds a one-line agent-guidance note on isolating the new logger in tests |

No other files were touched. No changes were made to `episode_ledger.py`, the poll gate, `_should_regenerate`, cadence constants, state payload construction/ordering, risk gates, order routing, sizing, exits, exchange calls, or dashboard code.

---

## Event schema

```json
{
  "ts": "ISO8601",
  "monotonic_s": 0.0,
  "event": "CALL_REACHED|ENTERED|LEDGER_REBUILD_STARTED|LEDGER_REBUILD_COMPLETED|COMPLETED|FAILED|ABORTED",
  "loop_id": 0,
  "duration_ms": 0.0,
  "episode_ledger_due": false,
  "episode_ledger_duration_ms": null,
  "last_completed_ts": "ISO8601|null",
  "last_completed_duration_ms": 0.0,
  "outcome": "ok|failed|aborted|null",
  "exception_type": "string|null",
  "exception_message": "string|null",
  "engine_version": "string",
  "git_sha": "string"
}
```

`last_completed_ts`/`last_completed_duration_ms` reflect the *previous* successful cycle at the time each event is emitted — except on a `COMPLETED` event itself, where the module-level tracking variables are updated immediately before that event is written, so a `COMPLETED` record's `last_completed_ts` equals its own timestamp. The next cycle's `ENTERED` event is what shows the true "time since last successful publish."

Sink: `logs/execution/pub_tick_heartbeat.jsonl`, written through the same `JsonlLogger` (`execution/log_utils.py`) already used for `execution_health.jsonl` — atomic append, size-based rotation, gzip archive of rotated segments.

## Event ordering

| Scenario | Sequence |
|---|---|
| Normal cycle, ledger not due | `CALL_REACHED → ENTERED → COMPLETED` (3 events) |
| Normal cycle, ledger due | `CALL_REACHED → ENTERED → LEDGER_REBUILD_STARTED → LEDGER_REBUILD_COMPLETED → COMPLETED` (5 events) |
| Ordinary `Exception` anywhere in `_pub_tick()` | `[CALL_REACHED →] ENTERED → FAILED`, then the exception is re-raised unchanged so the existing call-site `except Exception as exc: LOG.exception("[loop] publish_tick_failed: %s", exc)` behavior is fully preserved |
| Non-`Exception` `BaseException` (e.g. `KeyboardInterrupt`) | `[CALL_REACHED →] ENTERED → ABORTED`, then the exception continues propagating unmodified — **not suppressed** |
| Caller/scheduler skip (the still-unresolved hypothesis from the predecessor card) | **Zero events**, not even `CALL_REACHED` — this is the externally distinguishable signature a future gap needs to confirm or rule out `scheduler_or_control_flow_defect` |

Maximum routine volume is 5 events per publication attempt (ledger-due case); typical cycles are 3. No per-episode or per-fill events are emitted.

## Fail-open behavior

`_pub_tick_heartbeat()` wraps its own write in `try/except Exception`, so a telemetry-write failure (e.g. disk full, permission error) can never raise into `_pub_tick()` or the executor loop. Repeated failures are rate-limited to one `LOG.debug` line per 60 seconds rather than flooding the executor log. Verified directly by `test_telemetry_sink_failure_does_not_break_pub_tick`, which forces every heartbeat write to raise `OSError` and asserts `_pub_tick()` still completes and still performs its real state writes.

The `ABORTED` path is observation-only: the `finally` block checks a local flag and emits `ABORTED` if neither `COMPLETED` nor `FAILED` was reached, but it never catches `BaseException` — `KeyboardInterrupt`/`SystemExit`/process-termination semantics propagate exactly as before this change.

## Proof commands and results

```
$ python3 -m pytest tests/integration/test_pub_tick_heartbeat.py -q
........                                                                 [100%]

$ python3 -m pytest tests/integration/test_executor_state_files.py -q
.                                                                        [100%]

$ make test-fast
... 668 passed, several pre-existing skips, 0 failed ...

$ make test-runtime
... 5 passed, 4 pre-existing skips, 0 failed ...

$ python3 -m pytest -q tests/unit tests/integration -k "pub_tick or heartbeat or episode_ledger"
........................................................................ [ 93%]
.....                                                                    [100%]

$ git diff --check
(no output — clean)

$ python3 scripts/doc_invariants_check.py
22 passed, 0 failed (22 rules total)

$ PYTHONPATH=. python3 scripts/smoke_test.py
engine_version=v7.9 engine_metadata_updated_ts=<fresh>
state_health missing=0 stale=0 schema=0 cross=0
telemetry execution_health_issues=0
SMOKE_OK
```

`python3 -m py_compile execution/executor_live.py` was also run after every structural edit during implementation.

Confirmed no test run wrote to the live `logs/execution/pub_tick_heartbeat.jsonl` path — every test that exercises `_pub_tick()` or `_pub_tick_heartbeat()` monkeypatches `_PUB_TICK_HEARTBEAT_LOG` to an in-memory recorder or a `tmp_path` logger first.

## Manual verification

**Not performed in this card.** The card's acceptance criteria ask for post-deployment testnet verification, which requires restarting the live executor process (`hedge-executor`, PID running continuously since 2026-06-05) to load this code. That is a high-blast-radius action against a live trading process and was intentionally left for explicit operator approval rather than taken unilaterally — consistent with treating executor restarts as an action requiring confirmation, the same posture held throughout the predecessor investigation card.

Once approved and the executor is restarted, verification should confirm:
- `logs/execution/pub_tick_heartbeat.jsonl` is being created and append-only.
- Normal cycles show the full `CALL_REACHED → ENTERED → COMPLETED` (or ledger-due 5-event) sequence at the existing ~poll-interval cadence.
- Episode-ledger rebuild duration is visible and consistent with the drift trend documented in the predecessor report.
- No new executor errors or warning floods appear in `hedge-executor.err.log`.
- `positions_state.json`, `nav.json`, `diagnostics.json` retain their existing schemas (heartbeat instrumentation does not touch these payloads).
- Trading behavior (order submission, sizing, routing) is visibly unchanged.

## Explicit confirmation: trading behavior and publication cadence unchanged

- The poll-seconds entry-generation gate (`_should_regenerate`) was not touched.
- The episode-ledger rebuild trigger condition is unchanged: `(now_ts - state.last_episode_ledger_rebuild_ts) >= _EPISODE_LEDGER_REBUILD_INTERVAL_S`, only now assigned to a named variable (`_pub_tick_episode_ledger_due`) before the `if`, with identical semantics.
- No state payload (`nav_payload`, `positions_state_payload`, `synced_payload`, etc.) construction, field, or write call was modified, reordered, or removed. The one added parameter to `_pub_tick()` (`loop_id: int = -1`) is additive and defaults compatibly with the one pre-existing direct test call that omits it.
- No risk, sizing, routing, exit, or exchange-call code was touched.
- `_pub_tick_heartbeat()` calls are pure additions around the existing control flow; the `except Exception as exc: ... raise` block re-raises unchanged, preserving the exact exception that used to propagate to the call site.

## Known remaining causal frontier

This card does not resolve the predecessor investigation's open question — it makes the next occurrence resolvable. The two surviving hypotheses remain:

1. `scheduler_or_control_flow_defect` (primary) — some mechanism silently prevents the call site from reaching `_pub_tick()` at all despite entry-generation running.
2. A non-`Exception` `BaseException` escaping the top (still partially unguarded before this card, now fully bounded) section of `_pub_tick()`.

The next abnormal gap (>20 min between successful publications, per the drift trend: 6→12→17.6→22–23→33.7 min observed in one day) should be diagnosable directly from `pub_tick_heartbeat.jsonl` using the classification table in `EXECUTION_ARCHITECTURE.md`'s new "Publication Heartbeat Diagnostics" section:

- Zero events during the gap → confirms `scheduler_or_control_flow_defect`.
- `ENTERED` present with no terminal event and no ledger events → confirms hypothesis 2 (pre-ledger section escape).
- `ENTERED` + `LEDGER_REBUILD_STARTED` with no `LEDGER_REBUILD_COMPLETED` or terminal event → points at the episode-ledger rebuild itself (e.g. `_load_execution_log()`'s unbounded full-history reread, flagged as the likely cadence-drift driver in the predecessor report) rather than a control-flow defect.

## Next eligible card

Contingent on what the next abnormal gap's heartbeat evidence shows:

- If it proves episode-ledger blockage: `CARD-HEDGE-EPISODE-LEDGER-INCREMENTAL-REBUILD-001` (making `_load_execution_log()` incremental instead of re-reading full rotated history every call).
- If it proves a caller/control-flow skip: a narrowly scoped scheduler/control-flow investigation card targeting the exact code path between the screener-submission loop and the `_pub_tick()` call site.
- If no abnormal gap has recurred by the time this is reviewed: the incremental-ledger performance work remains independently justified by `_load_execution_log()`'s confirmed unbounded-scaling design, but should stay a separate card from this instrumentation, per the predecessor report and this card's own scope boundary.

No implementation work for either follow-on card was started here.
