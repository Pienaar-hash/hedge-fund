# CARD-HEDGE-LIVE-STATE-PUBLISH-FREEZE-ROOT-CAUSE-001

**Type:** Read-only forensic investigation
**Scope:** Root cause of the 2026-07-10 publication freeze in `positions_state.json`, `nav.json`, `diagnostics.json`
**Method:** Static code trace of `execution/executor_live.py`, `execution/episode_ledger.py`, `execution/exit_reason_normalizer.py`, `execution/position_ledger.py`; log forensics against `/var/log/hedge-executor.err.log`; live file-mtime and process inspection.
**Hard boundaries respected:** No restarts, no process termination, no config/state/leverage/position changes, no runtime instrumentation added to the live process, no fix implemented. All findings are read evidence only.

---

## Executive verdict

**Classification: `scheduler_or_control_flow_defect` (primary), `not_reproducible_insufficient_evidence` (secondary, for the exact skipped line), `latent_safety_defect` (secondary, for the growth trend).**

The publication freeze reported in `CARD-HEDGE-RED-AUDIT-ROOT-CAUSE-CLASSIFICATION-001` was **real but transient, not permanent** — it self-recovered without any restart. As of this report (2026-07-10 19:36 UTC), `positions_state.json` / `nav.json` / `diagnostics.json` are current as of **19:29:02–03 UTC**, roughly 7 minutes old, consistent with the system's normal publish cadence.

The actual anomaly is a **single abnormally long gap between successful `_pub_tick()` completions**: 18:55:29 → 19:29:11 UTC, **33.7 minutes**, versus a normal cadence of 6–12 minutes. This is the worst gap observed in the retained log, but it is **not unprecedented** — two earlier gaps of 22.3 and 23.4 minutes occurred the same day (14:16–15:02 UTC), on top of a visible upward drift in baseline cadence (6 min in early afternoon → 12 min by mid-afternoon → 17.6 min once → 22–23 min twice → 33.7 min tonight). This is a **degrading trend, not a one-off glitch**.

Root cause is **narrowed but not fully proven** (Acceptance criterion B). Two facts eliminate the two most obvious explanations:

- **Not a crash/exception.** `"[loop] publish_tick_failed"` (the exception handler wrapping every call to `_pub_tick()`) and `"[executor] LOOP_CRASH"` (the outermost handler wrapping the entire main loop) both occur **zero times** anywhere in the retained log (back to 2026-07-06). No exception of any kind escaped either handler during this incident or in the preceding four days.
- **Not a full main-loop hang.** During the 33.7-minute gap, the executor continued to log soak-cycle status every 30–90s, continued polling NAV, and — critically — ran **at least two additional full entry-generation cycles** (BTCUSDT `auto_reduce` signal → `INTENT` → `SEND_ORDER` → `ORDER_FILL`, at 18:56:41 and 19:02:07) that share the exact same code path (`_loop_once` → `generate_intents()` → screener submission loop) that structurally precedes, and is *not gated from*, the `_pub_tick(state)` call site.

That last point is the unresolved core of the investigation: entry-generation demonstrably ran to completion at least twice during the gap, immediately followed in the source by `_pub_tick(state)` with no conditional between them — yet neither `_pub_tick`'s file writes nor its unconditional completion log line (`"[v6-runtime] state write complete"`) appeared until 19:29:11. This is documented below as the surviving, unresolved causal frontier.

---

## Runtime timeline

All times UTC, 2026-07-10, from `/var/log/hedge-executor.err.log`.

| Time | Event |
|---|---|
| 18:49:29.891 | Previous `_pub_tick()` cycle completes normally (`Episode ledger saved: 1622 episodes`) |
| 18:49:30.199 | `[v6-runtime] state write complete ... nav=True positions_state=True ... diagnostics=True ...` |
| 18:55:11–18:55:18 | Entry-generation cycle runs: Hydra pipeline (3 merged intents), SOLUSDT screener signal, INTENT/fee-gate veto, `[screener] attempted=3 emitted=3 submitted=1` |
| 18:55:18.563 | `[nav-detail] nav_total=6790.64...` (nav computation, part of `_pub_tick`'s unguarded top section) |
| 18:55:18.844 | `[executor] positions.json updated n=2 len=2` |
| 18:55:19.419 | `[nav-state] AUM present: futures=6790.64...` (last log line before the warning burst) |
| 18:55:21.220–18:55:21.699 | **35,050** `exit_reason_unmapped raw='unknown' source='episode_ledger' ...` WARNING lines emitted in ~0.48s (global unmapped-counter jumps 298,802 → 333,850) |
| 18:55:26.762 | `B.4: Shadow index loaded — 6 ENTRY keys, 6 EXIT keys, 9 DECISION records` |
| 18:55:28.484 | `B.4: Authority binding complete — entry 34%, exit 5%, 1537 missing, 0 ambiguous` |
| 18:55:29.026 | `Episode ledger saved: 1622 episodes` |
| **18:55:29.399** | **`[v6-runtime] state write complete ... nav=True positions_state=True ... diagnostics=True ... episode_ledger=True`** — last confirmed successful `_pub_tick()` completion before the gap |
| 18:55:59–19:05:34 | Executor continues normally: soak-cycle logs every 30–90s, NAV cache snapshots, `[risk_loader]` checks, `[executor] account OK ... positions: 2` |
| 18:56:00–18:56:41 | **Full entry-generation cycle**: BTCUSDT `auto_reduce` signal → `[executor] INTENT` → `SEND_ORDER` → `ORDER_ACK` → `ORDER_FILL id=20720785852` → `[screener] attempted=1 emitted=1 submitted=1` → `[doctrine] MEAN_REVERT ... blocked=0/1` |
| 19:01:27–19:02:07 | **Second full entry-generation cycle**: BTCUSDT `auto_reduce` again → `INTENT` → `SEND_ORDER` → `ORDER_ACK` → `ORDER_FILL id=20721775184` → `[screener] attempted=1 emitted=1 submitted=1` |
| 18:55:29 → 19:29:11 | **No `"[v6-runtime] state write complete"` line. No `"publish_tick_failed"`. No `"LOOP_CRASH"`. No writes to `positions_state.json`/`nav.json`/`diagnostics.json` (confirmed by current file mtimes, see below).** |
| 19:29:04.245 | `exit_reason_unmapped ... count=333900` (only +50 since the burst — confirms the burst was one-time, not repeating) |
| 19:29:11.035 | `Episode ledger saved: 1622 episodes` |
| **19:29:11.423** | **`[v6-runtime] state write complete ...` — recovery, gap closes at 33.7 minutes** |
| 19:35–19:36 (report time) | `positions_state.json`/`nav.json`/`diagnostics.json` mtime = 19:29:02–03 (fresh, 7 min old); `risk_snapshot.json` mtime = 19:35:13; `router_health.json` mtime = 19:36:01 |

---

## Publication architecture map

`_pub_tick(state)` (`execution/executor_live.py:5565`) is a single function that, when entered, does the following **in order**:

1. **Unguarded top section (no try/except):**
   `_compute_nav_with_detail()` → `_collect_rows()` → `_persist_positions_cache(rows)` → `_persist_nav_log(nav_val, rows)` → `_persist_spot_state()` (lines 5569–5573).
2. **Individually try/except-guarded writes** (each catches its own exception, logs, and continues — one section's failure cannot block another): `write_nav_state`, `_write_positions_state`, `write_positions_snapshot_state`, `write_positions_ledger_state`, `write_risk_snapshot_state` (a *second*, cache-based writer distinct from the one below), VaR/CVaR state, alpha-decay state, execution-alpha state, `write_symbol_scores_state`, `write_runtime_diagnostics_state`, `write_engine_metadata_state`, `write_synced_state`, Phase-C readiness state, binary-lab state.
3. **Throttled sub-block** (only runs if `now_ts - state.last_episode_ledger_rebuild_ts >= _EPISODE_LEDGER_REBUILD_INTERVAL_S`): calls `execution.episode_ledger.rebuild_and_save()`, then edge-calibration / engine-lift / hydra-monotonicity snapshot persistence, all individually guarded (`executor_live.py:5804–5843`).
4. **Unconditional completion log** at line 5844: `"[v6-runtime] state write complete ..."`, listing every `*_written` boolean. This is the **only** place any of these flags are reported, and it is reached by falling through the entire function body — there is no early `return` anywhere in `_pub_tick()` except the final `return None` at line 5861.

**Caller:** `_loop_once(state, i)` (`executor_live.py:5864`) calls `_pub_tick(state)` at line 6521, wrapped in its own try/except that logs `"[loop] publish_tick_failed: %s"` on any exception (line 6520–6523). Immediately before this call, `_loop_once` runs exit-scanning (lines 5931–6078, its own distinct `[exit_scanner] DOCTRINE EXIT` / `SEATBELT EXIT` logging), then a poll-gate check (`if not _should_regenerate: ... return` — commit `5cf28c3b`/`4e81bcac`), then (if the gate opens) `generate_intents()` and a screener submission loop ending in the `"[screener] attempted=%d emitted=%d submitted=%d"` log line at 6513–6517 — **immediately** followed, with nothing in between, by the `_pub_tick(state)` call.

**Independent, non-`_pub_tick` writers** (called from the outer `while True:` loop, `executor_live.py:6524–6529`, *after* `_pub_tick` returns, each individually throttled and guarded):
- `_maybe_emit_router_health_snapshot()` (`executor_live.py:893`) — writes `router_health.json`, throttled by `ROUTER_HEALTH_REFRESH_INTERVAL_S`, own try/except per sub-step.
- `_maybe_emit_risk_snapshot()` (`executor_live.py:924`) — writes `risk_snapshot.json` (via a *different* code path than `_pub_tick`'s risk write), throttled by `_HEALTH_PUBLISH_INTERVAL_S`, own try/except.

---

## Caller/control-flow trace

```
main while True: loop (executor_live.py:6658)
 └─ _loop_once(_STATE, i)                         [try/except → "LOOP_CRASH" if uncaught]  ← 0 occurrences
     ├─ exit_scanner block (5931–6078)             [distinct log tags, runs every tick]
     ├─ poll-gate check (~6193–6222)                if not elapsed(poll_seconds): return    ← skips everything below
     ├─ generate_intents() + screener submit loop  [confirmed ran at 18:56 & 19:02]
     ├─ "[screener] attempted=... submitted=..."   ← confirmed logged at 18:56:41, 19:02:07
     └─ try: _pub_tick(state)                      [try/except → "publish_tick_failed"]     ← 0 occurrences
         except Exception: LOG.exception(...)
 └─ _maybe_emit_router_health_snapshot()           [independent, unconditional every tick]
 └─ _maybe_emit_risk_snapshot()                    [independent, unconditional every tick]
```

No conditional exists in the source between the `"[screener] attempted..."` log line and the `_pub_tick(state)` call. Given the screener-submission log line fired at 18:56:41 and 19:02:07, the call to `_pub_tick(state)` must have been reached immediately after both times, under normal control flow.

---

## Frozen-vs-live surface comparison

| File | Writer | Freshness at report time (19:36 UTC) | Explanation |
|---|---|---|---|
| `positions_state.json` | `_pub_tick` (unconditional write, guarded) | 19:29:02 (7 min old) | Part of `_pub_tick`; froze 18:55:29→19:29:02, then recovered |
| `nav.json` | `_pub_tick` (unconditional write, guarded) | 19:29:02 (7 min old) | Same as above |
| `diagnostics.json` | `_pub_tick` (unconditional write, guarded) | 19:29:03 (7 min old) | Same as above |
| `risk_snapshot.json` | `_maybe_emit_risk_snapshot()` — **independent function**, called unconditionally every loop tick outside `_pub_tick` | 19:35:13 (1 min old) | Never depended on `_pub_tick` completing; stayed fresh throughout |
| `router_health.json` | `_maybe_emit_router_health_snapshot()` — **independent function**, same pattern | 19:36:01 (fresh) | Same reason |

This fully answers "why did `risk_snapshot.json` and `router_health.json` remain fresh": **they were never coupled to `_pub_tick()` in the first place.** They are written by separate, independently-throttled functions called unconditionally after `_pub_tick` returns (or doesn't complete usefully) each tick. The three-file freeze is a genuine shared failure seam (all three are written from within the same `_pub_tick()` call, all before its one unconditional completion log line), but it does **not** extend to `risk_snapshot.json`/`router_health.json`, which were never part of that seam.

---

## Episode-ledger causality analysis

The 35,050-line `exit_reason_unmapped` burst is real but **cannot, by itself, explain a 34-minute gap**:

- The burst itself was fast: all 35,050 log lines were written in ~0.48 wall-clock seconds (18:55:21.220–18:55:21.699).
- It was a **one-time event**, not a recurring storm: the global `_UNMAPPED_COUNTER` (`execution/exit_reason_normalizer.py:162`) jumped from 298,802 (last seen 2026-07-06) to 333,850 in that single burst, then advanced by only ~50 over the subsequent 34 minutes — consistent with normal per-cycle behavior, not a repeat.
- The immediately following cycle (18:55:21→18:55:29, ~8 seconds) completed **successfully**, including the full episode-ledger rebuild and the terminal completion log. So the burst's *own* cycle was not the stalled one — the *next* cycle was.

The call site (`execution/episode_ledger.py:981`, inside `build_episode_ledger`) fires once per reconstructed "episode close" event during ledger reconciliation. With only ~1,622 total episodes ever recorded, 35,050 unmapped calls in one pass is roughly 21× that count — meaning the reconciliation loop synthesized far more intermediate "episode close" events in that one pass than the persisted episode count reflects. This is consistent with (but not proven to be caused by) the fact that `_load_execution_log()` (`execution/episode_ledger.py:615`) **reads the current `orders_executed.jsonl` plus every rotated `orders_executed.*.jsonl` file, every single call, with no incremental caching** (confirmed via its own docstring: "Reads all `orders_executed*.jsonl` files ... so entries from rotated files are not lost"). This is an unbounded-by-design, full-history reprocessing cost that scales with total historical fill count, not with "fills since last rebuild." It is the most direct candidate for **why the baseline cadence has been drifting upward across the day** (6 min → 12 min → 17.6 min → 22–23 min → 33.7 min) — but it is a *cost driver*, not a proven cause of the specific 34-minute stall, since the rebuild that actually spanned the stall window is not directly observable (no intermediate log lines exist to show it running slowly versus not running at all).

---

## Eliminated hypotheses

| Hypothesis | Status | Evidence |
|---|---|---|
| Uncaught exception inside `_pub_tick()` | **Eliminated** | `"publish_tick_failed"` appears 0 times in the entire retained log (back to 2026-07-06) |
| Uncaught exception anywhere in `_loop_once` (incl. after `_pub_tick`) | **Eliminated** | `"LOOP_CRASH"` appears 0 times anywhere in the retained log |
| Full main-loop hang / deadlock | **Eliminated** | Soak-cycle logs, NAV snapshots, and two full entry-generation→order-fill cycles ran normally during the gap |
| Permanent freeze / process death | **Eliminated** | Files resumed updating at 19:29:02 without any restart; process (PID 68641) has run continuously since 2026-06-05 with no gap in `ps` uptime |
| `risk_snapshot.json`/`router_health.json` freshness implies `_pub_tick` was fine | **Eliminated** | Those two files are written by structurally independent, unconditional functions never gated behind `_pub_tick()`'s success |
| `stale_flags`/NAV `balances_ok` misconfiguration causing the freeze | **Eliminated** (carried over from prior card) | Confirmed double-negative naming, not a real staleness signal |
| `poll_seconds` misconfigured/changed to a large value | **Eliminated** | `config/strategy_config.json` mtime is 2026-06-04, unchanged; no git diff; value is `300` as expected |
| Filesystem/resource exhaustion (disk, inodes, fds) | **Eliminated** | Disk 42% used (21G free), inodes 7% used, process has 12 open FDs vs. 1024 soft limit, RSS 227MB — no resource pressure |
| Repeating `exit_reason_unmapped` storm throughout the gap | **Eliminated** | Counter delta during the gap window was ~50, not another burst |
| `4e81bcac`/`5cf28c3b` poll-gate commits caused this incident directly | **Eliminated as direct cause** | Process has run continuously since 2026-06-05 13:24:25, predating any restart that could pick up code changes; the fix has been in the running process the entire time |

---

## Confirmed or remaining causal mechanism

**Confirmed:** The freeze is isolated to the three files written inside `_pub_tick()`'s guarded-write section, gated behind its one unconditional terminal log line. It is not an exception, not a full hang, and not caused by the two files that stayed fresh being on the same code path (they are not). The baseline publish cadence has been drifting upward across the day in a pattern consistent with an unbounded, full-history reprocessing cost inside the throttled episode-ledger rebuild sub-block, which reads all rotated `orders_executed*.jsonl` files on every invocation with no incremental caching.

**Not fully proven — narrowed to two surviving hypotheses (Acceptance B):**

1. **`scheduler_or_control_flow_defect` (primary candidate):** Some condition — not visible in the ~4 lines of source between the screener-submission log line and the `_pub_tick(state)` call, and not present as any exception in the logs — caused `_pub_tick(state)` to not be entered, or to return before doing any work, during the entry-generation cycles at 18:56:41 and 19:02:07 (and likely one or two more inferred cycles around 19:07–19:24 that were not individually examined). This is the hypothesis most consistent with all direct evidence, but the exact mechanism was not located by static reading alone.

2. **A `BaseException`-derived (not `Exception`-derived) error escaping `_pub_tick()`'s unguarded top section** (lines 5569–5573), which would bypass the `except Exception` wrapper at the call site without being logged as `publish_tick_failed`, and would also need to not propagate as `LOOP_CRASH` at the outer loop level — this requires either an even-higher-level handler not yet located, or a self-suppressing exception class (e.g., a misused generator/coroutine `GeneratorExit`) somewhere in the `_compute_nav_with_detail()` / `_collect_rows()` call graph. This is a lower-probability but not excluded candidate.

**Evidence needed to distinguish them:** entry/exit heartbeat logging placed at the very first and very last lines of `_pub_tick()` (before the unguarded top section, and immediately before the final `return None`), correlated against the next occurrence of a gap >20 minutes. This would directly show whether `_pub_tick()` is entered without exiting (implicates hypothesis 2) or not entered at all (implicates hypothesis 1). This instrumentation was **not** added in this investigation per the read-only hard boundary, and is the explicit next step recommended below.

---

## Severity and recurrence assessment

- **Recurrence: highly likely, and worsening.** Three gaps >20 minutes occurred within a single trading day (14:16–15:02, and 18:55–19:29), against a rising baseline (6→12→17.6→22–23→33.7 min). If the episode-ledger full-history reprocessing cost is a contributing driver, this will mechanically worsen as `orders_executed.jsonl` and its rotated siblings accumulate further, independent of whether the unresolved control-flow question is ever triggered.
- **Impact:** During each gap, `positions_state.json` and `nav.json` — the two surfaces most likely to be trusted by a dashboard or an investor-facing view — silently serve stale data with no visible staleness flag distinguishing this from a healthy state, for up to half an hour at the worst observed point so far. `risk_snapshot.json` (used for HALTED-mode gating) is unaffected, which limits the safety blast radius, but the previously-reported "positions shown as open when actually closed" investor-facing defect is a direct, reproducible consequence of this freeze pattern, not a one-time fluke.
- **Not tied to a single corrupted runtime state.** The pattern recurred multiple times across the same continuous 35-day process uptime, with full recovery each time — this looks like a performance/scheduling characteristic of the current code, not damage from one bad event, and should be expected to recur after a restart as well.

---

## Smallest safe implementation seam (for a future card — not implemented here)

Two independent, additive changes, each individually bounded:

1. Add entry/exit heartbeat logging (or a monotonic "last `_pub_tick` entered / last `_pub_tick` exited" timestamp pair written to a small always-fresh sidecar file) around `_pub_tick()`'s call site, to convert hypothesis 1 vs. hypothesis 2 above from "narrowed" to "proven" the next time a >20 minute gap occurs.
2. Make `_load_execution_log()` incremental (cache parsed fills keyed by file+offset, or bound it to a rolling lookback window by default) instead of re-reading and re-parsing every rotated `orders_executed*.jsonl` file on every throttled rebuild — this directly addresses the observed upward cadence drift regardless of which control-flow hypothesis is eventually confirmed.

Neither change is implemented in this card.

---

## Required regression proof (for the future implementation card)

- A test that asserts `_pub_tick()`'s completion log/heartbeat fires within N seconds of being called, using a fixture that forces the episode-ledger rebuild path to run against a synthetically large (rotated-file-scale) fills dataset, to catch both a control-flow skip and a performance regression.
- A test asserting `_load_execution_log()`'s cost does not scale linearly with total historical file size once made incremental (e.g., time a rebuild before/after a synthetic rotation event).
- An assertion that `"[loop] publish_tick_failed"` or an equivalent explicit marker is logged for **every** case where `_pub_tick()` does not reach its terminal log line — i.e., no silent-skip path should be reachable without a log trace, whatever the mechanism turns out to be.

---

## Recommended next implementation card

`CARD-HEDGE-PUBTICK-HEARTBEAT-AND-INCREMENTAL-LEDGER-001`: add the heartbeat instrumentation described above (read-only, additive, safe to deploy without behavior change) as a **first** step, observe the next >20-minute gap with it in place to definitively confirm hypothesis 1 vs. 2, and **only then** scope the incremental-episode-ledger-load fix as a second, separate card once the root cause is fully proven rather than narrowed.

---

## Explicit statement

No runtime changes were made during this investigation. Supervisor was not restarted, the executor was not terminated or paused, no leverage/margin/position/order/credential/configuration changes were made, and no state or log files were modified, rewritten, or deleted. All evidence above was gathered via read-only inspection of existing log files, state files, and source code, plus non-destructive process/filesystem introspection (`ps`, `stat`, `df`).
