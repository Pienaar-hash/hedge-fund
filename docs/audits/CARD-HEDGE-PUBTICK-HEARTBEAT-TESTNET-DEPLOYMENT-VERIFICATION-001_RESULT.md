# CARD-HEDGE-PUBTICK-HEARTBEAT-TESTNET-DEPLOYMENT-VERIFICATION-001 — Result

**Final disposition: `verified_with_nonblocking_findings`**

## Deployment identity and boundary

- Deployed code commit: `970852a2fca4996f07f057c5d07e7ad3dbfb38ee`.
- Executor startup logged `commit=p6-mainline-landing-2026-04-29-51-g970852a2`.
- Supervisor program: `hedge:hedge-executor`; command: `sudo supervisorctl restart hedge:hedge-executor`.
- Restart: 2026-07-12T10:09:27Z.  PID changed from `68641` (36d 20h uptime) to `349509`; it remained `RUNNING` through 2026-07-12T10:23:02Z (13m 34s uptime), with no restart loop.
- Environment was confirmed twice by the executor and by a separate read-only client: `ENV=testnet`, `DRY_RUN=0`, `BINANCE_TESTNET=1`, base URL `https://testnet.binancefuture.com`.

## Mixed-commit review

All 47 changed files and their hunks were inspected against `970852a2^`.  The categories below classify every changed hunk; “incidental” means no executor runtime behavior.

| Classification | Files / hunks | Review conclusion |
| --- | --- | --- |
| Heartbeat instrumentation | `execution/executor_live.py` `_pub_tick_heartbeat`, `_pub_tick` terminal/ledger events, and `_loop_once` `CALL_REACHED` hunk | Append-only, fail-open JSONL telemetry.  It only records timestamps/durations/outcomes and re-raises the pre-existing publication exception path; no sizing, routing, cadence, payload, or risk decision mutation. |
| Heartbeat tests/documentation | `tests/integration/test_pub_tick_heartbeat.py`, `tests/integration/test_executor_state_files.py`, `docs/audits/CARD-HEDGE-PUBTICK-BOUNDARY-HEARTBEAT-INSTRUMENTATION-001_RESULT.md`, `docs/active/OPERATIONS.md`, `docs/active/Runbook.md`, `docs/active/TESTING.md`, `README.md`, `Makefile`, `.github/workflows/ci.yml`, `scripts/smoke_test.py`, `scripts/doc_invariants_check.py`, `pytest.ini` | Tests cover normal, ledger, terminal failure/abort, append-only, and sink-failure paths.  CI/Makefile/docs additions are non-runtime. |
| Unrelated startup/shadow/runtime work | `execution/executor_startup.py`, `execution/startup_reconciliation.py`, `execution/executor_shadow_services.py`, `execution/doctrine_bridge.py`, `execution/executor_live.py` import/delegation/startup/doctrine/shadow hunks, `execution/intel/pipeline_v6_compare.py`, `execution/mirror_builders.py`, `dashboard/state_client.py`, `execution/state_publish.py`, `execution/exchange_utils.py`, `scripts/go_live_now.sh`, `scripts/pipeline_compare_service.py`, `tests/integration/test_doctrine_entry_runtime.py`, `tests/unit/test_doctrine_bridge.py`, `tests/unit/test_startup_reconciliation.py`, `utils/jsonl_tail.py` | Startup delegation preserves the existing position check and existing cancel-all behavior.  New reconciliation reads the four-hour local ACK window, performs signed read-only order/trade lookups for unresolved ACKs, can append synthetic recovery events, then invokes the pre-existing cancel-all call and writes a summary.  Shadow extraction remains fail-open and writes only telemetry/state.  Doctrine bridge can fail closed for new entries when dependency state is stale; reduce-only bypass is tested.  This is a genuine entry-veto dependency change, but it caused no order and no failure during the observation. |
| Incidental/generated/other mixed work | `BACKTEST_LEDGER.md`, `EXECUTION_ARCHITECTURE.md`, `INVESTOR_REPORT_TEMPLATE.md`, `PROJECT_BRIEF.md`, `RISK_POLICY.md`, `STRATEGY_REGISTRY.md`, `docs/RESEARCH_PROGRAM_STATUS.md`, `docs/audits/CARD-HEDGE-LIVE-STATE-PUBLISH-FREEZE-ROOT-CAUSE-001.md`, `docs/audits/CARD-HEDGE-RED-AUDIT-ROOT-CAUSE-CLASSIFICATION-001.md`, `docs/audits/LIVE_STATE_VERIFICATION_AUDIT_2026-07-10.md`, `config/runtime.yaml`, research/Hydra tests, and `v7_manifest.json` | Documentation, manifest/test expectation, fee-comment/config metadata, or formatting-only changes; no deployed executor dependency beyond the separately reviewed runtime files above. |

### Unrelated runtime safety assessment

- **Startup reconciliation:** expected behavior is crash-window event repair before the pre-existing stale-order cleanup.  Dependencies are the local execution log plus Binance testnet order/trade read APIs.  Failures are caught and logged; startup continues.  Tests cover ACK scanning, all four classifications, recovery event shape, and call order.  At restart: `unresolved=0`, `cases={}`, `errors=0`; therefore no synthetic fill/close event was written.
- **Existing cancel-all startup behavior:** before restart, read-only queries found zero active positions and zero open orders.  The startup API returned successful cancel-all acknowledgements for BTCUSDT and ETHUSDT; the executor summary recorded `0` errors and no order existed to cancel.  This behavior was present in the parent commit, not introduced by this card.
- **Shadow services:** pipeline shadow was enabled by existing Supervisor configuration; the extracted functions have no order-send path and are exception-contained.  The shadow runner initialized normally.  DLE enforcement remained off (`SHADOW_DLE_ENABLED=0`, `DLE_ENFORCE_ENTRY_ONLY=0`).
- **Doctrine bridge:** fresh dependency states were available at restart.  Three generated entry candidates were vetoed by the existing doctrine layer (`VETO_DIRECTION_MISMATCH`) and no order ACK was emitted.  It can alter future entry vetoes when dependencies are degraded, so it is not behavior-neutral in design; no unsafe behavior was observed in this testnet window.
- **Precision cache:** startup’s existing exchange-precision refresh rewrote the tracked `config/exchange_precision_cache.json`; this was a generated side effect outside the trading universe and is restored to `HEAD` after this audit.  It is not included in the deployment result commit.

## Pre-restart operating state

Captured 2026-07-12T10:09:26Z:

- Git `HEAD`: `970852a2`; status was clean except intentionally untracked `.claude/`.
- Supervisor executor: PID `68641`, `RUNNING`, uptime 36d 20h.
- Testnet/dry-run: `ENV=testnet`, `DRY_RUN=0`; preflight read-only client confirmed testnet base URL.
- Exchange positions: none.  Dashboard `positions_state.json`: `positions: []`.
- Current NAV: `6746.7903486800005`; `nav_mode: live_wallet`.
- Risk mode: `HALTED`, reason `nav_stale_age=117s` (pre-existing publication freshness condition).
- `pub_tick_heartbeat.jsonl`: absent.
- State-file top-level schema fingerprints were captured and are identical post-restart:
  - positions: `858d7929e4a5a0f3362f4fbfbfc05c79db69e4dc336500171223c07d83f30fa0`
  - nav: `781ba2d5b337cef36255cf5ba5e8ab71ea9486402f574c9393a8fc6807424282`
  - diagnostics: `447033ef6c5ea12d965796c6f3fcc5b96ffa38d2df179f55e8dc55a3702fb226`
  - risk snapshot: `a2feb84d460ce7801f149e505e8547001effd1998b72ea41ba12e688ee34afc8`
  - router health: `15c752bd2a5241fadd57174b3e18b1e797b05d99bdf1a8bef0cf4d4ef05c6526`
- Existing error-log condition: high-volume `exit_reason_unmapped raw='unknown' source='episode_ledger'` warnings were present before restart.  They recur during each full ledger rebuild and were not introduced by heartbeat instrumentation.

## Restart and reconciliation evidence

The minimum process was restarted; dashboard and sync service were not restarted.  The new executor logged testnet routing, HEDGE position mode, zero open positions, zero unresolved ACKs, and normal startup completion.  Post-restart Binance queries again found zero active positions and zero BTCUSDT/ETHUSDT orders since restart.  `positions_state.json` remained `[]`.

No post-restart order ACK, fill, close, `publish_tick_failed`, `LOOP_CRASH`, heartbeat sink failure, `FAILED`, `ABORTED`, traceback, or restart loop was observed.  The only post-restart `orders_executed.jsonl` record was the reconciliation summary (`unresolved_count: 0`, `case_counts: {}`).

## Heartbeat evidence

`logs/execution/pub_tick_heartbeat.jsonl` was created at 2026-07-12T10:09:57Z and grew append-only from zero to 15 records.  Every record reports engine `v7.9` and Git SHA `p6-mainline-landing-2026-04-29-51-g970852a2`.

| Loop | Event order | `_pub_tick()` duration | Episode-ledger duration | Terminal fields |
| --- | --- | ---: | ---: | --- |
| 0 | `CALL_REACHED → ENTERED → LEDGER_REBUILD_STARTED → LEDGER_REBUILD_COMPLETED → COMPLETED` | 13062.06 ms | 10178.98 ms | `last_completed_ts=2026-07-12T10:10:10.102549+00:00`, `last_completed_duration_ms=13062.06` |
| 9 | `CALL_REACHED → ENTERED → LEDGER_REBUILD_STARTED → LEDGER_REBUILD_COMPLETED → COMPLETED` | 11215.36 ms | 8528.10 ms | `last_completed_ts=2026-07-12T10:16:15.746311+00:00`, `last_completed_duration_ms=11215.36` |
| 18 | `CALL_REACHED → ENTERED → LEDGER_REBUILD_STARTED → LEDGER_REBUILD_COMPLETED → COMPLETED` | 11707.96 ms | 8925.53 ms | `last_completed_ts=2026-07-12T10:22:20.836595+00:00`, `last_completed_duration_ms=11707.96` |

All three attempts were ledger-due, successful, and correctly ordered.  The required normal no-ledger-due sequence was not naturally observable: the unchanged ledger interval is 300 seconds while the observed publication intervals were about 355–363 seconds.  Thus each natural `_pub_tick()` was due for a ledger rebuild.  No cadence/configuration was changed and no artificial event was forced.  This is the nonblocking finding behind the disposition.

## Publication surfaces and runtime regression checks

- `positions_state.json`, `nav.json`, and `diagnostics.json` refreshed at 2026-07-12T10:22:11Z with unchanged schemas.
- Independent writers continued: `risk_snapshot.json` refreshed at 10:22:11Z and `router_health.json` at 10:22:01Z.
- Latest NAV was `6746.65034868`; `nav`, `nav_usd`, and AUM futures/total agree, and positions are empty on both exchange and dashboard state.
- Risk state was briefly `OK` after the first refresh, then returned to pre-existing conservative `HALTED` with `nav_stale_age` once its 150-second freshness threshold elapsed between the roughly six-minute publication cycles.  This is a pre-existing cadence/freshness mismatch, not a heartbeat failure.
- Candidate entry signals were vetoed; no unsolicited non-reduce-only order, sizing change, routing action, exit, leverage/margin action, or risk-bypass event occurred.
- The recurring `exit_reason_unmapped` warning burst during ledger rebuild remains a known baseline warning stream.  There was no new heartbeat warning storm.

## Rollback status

No rollback condition occurred.  The executor remains running on testnet at deployed commit `970852a2`.  No live patch was applied.  The only runtime-generated tracked cache update is restored after this audit.

## Follow-up

Observe future publication gaps with this heartbeat stream.  Open a separate bounded episode-ledger cadence/scaling card: the full rebuild presently costs 8.5–10.2 seconds and, because publication cadence exceeds the 300-second ledger interval, naturally occurs on every observed publication attempt.  Do not treat this verification as proof of the historical silent-gap root cause.
