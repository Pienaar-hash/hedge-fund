# Phase B — Shadow Authority Layer (Complete)

**Tag:** `v7.9-b4-shadow-authority`
**Baseline:** `v7.9-stable` (`6fe53c6c`)
**Branch:** `v7.6-dev`
**Suite:** 2968 passed, 47 skipped, 0 failures

> Every completed economic lifecycle can be replayed as an Episode with explicit
> authority provenance, without altering execution behavior.

---

## Commits

| Phase | Commit | Description |
|-------|--------|-------------|
| B.1 | `38dfbcfe` | Exit reason normalization — 10 canonical values, startup invariant |
| B.2 | `aa91435a` | DLE shadow decision enrichment — verdict, context_snapshot, provenance |
| B.3 | `ed86be5a` | DLE shadow permit enrichment — TTL, DENY suppression, scope_snapshot |
| B.4 | `14a6b554` | Episode schema binding — authority chain, deterministic UID, regime bind |

**Total diff from baseline:** 14 files, +2968/−174 lines.

---

## Surfaces Created / Expanded

### New files

| File | Purpose |
|------|---------|
| `execution/exit_reason_normalizer.py` | `CanonicalExitReason` enum (10 values), `normalize_exit_reason()`, startup invariant `verify_doctrine_coverage()` |
| `tests/unit/test_dle_shadow_b2.py` | 18 tests — DECISION v2 schema, backward compat, path invariant |
| `tests/unit/test_dle_shadow_b3.py` | 23 tests — PERMIT v2, DENY suppression, TTL, backward compat |
| `tests/unit/test_exit_reason_normalization.py` | 64 tests — all 10 canonical reasons, raw→canonical mapping |
| `tests/unit/test_episode_authority_b4.py` | 42 tests — UID, matcher, regime binding, index loading, integration |

### Modified files

| File | Changes |
|------|---------|
| `execution/dle_shadow.py` | V2 schema coexistence, DECISION enrichment (verdict, context_snapshot, provenance), PERMIT enrichment (action, TTL, scope_snapshot), DENY→PERMIT suppression, LINK payload.ts, path invariant |
| `execution/episode_ledger.py` | `EpisodeV2` + `AuthorityRef` + `AuthorityFlags` dataclasses, shadow log index loader, binary-search matcher, deterministic `episode_uid`, authority binding, regime binding from DECISION, V2 output format, authority stats |
| `execution/executor_live.py` | `_dle_enrichment()` helper (DECISION + PERMIT kwargs), 3 call sites wired (ENTRY_DENY, ENTRY_ALLOW, EXIT_ALLOW), startup invariants (exit reason map + DLE shadow path) |
| `execution/exit_scanner.py` | Exit reasons normalized via `_normalize_exit()` |
| `config/exit_reason_map.yaml` | Rewritten — 10 canonical entries with raw→canonical mappings |
| `v7_manifest.json` | DLE shadow: 4-event chain, per-type field docs, permit_behavior section, LINK.ts. Episode ledger: V2 fields, authority_stats, matching_algorithm |

---

## Invariants

All enforced at executor startup (fail-loud):

1. **Exit reason doctrine coverage** — `verify_doctrine_coverage()` asserts every canonical exit reason in `CanonicalExitReason` has a mapping in `exit_reason_map.yaml`. Missing mapping → `ValueError` on startup.

2. **DLE shadow log path** — `verify_shadow_log_path()` asserts `DEFAULT_LOG_PATH == MANIFEST_LOG_PATH`. Path mismatch → `ValueError` on startup.

Runtime invariants (shadow-safe, fail-open):

3. **PERMIT suppression on DENY** — When `verdict == "DENY"`, no PERMIT event is emitted. LINK.permit_id is null.

4. **V2 schema promotion** — Any enrichment field present → event promoted to `dle_shadow_v2`. Absent enrichment → stays `dle_shadow_v1`.

5. **Authority binding fail-open** — Missing shadow log, malformed events, unparseable timestamps all result in explicit `authority_flags` (never crashes, never guesses).

6. **Episode UID determinism** — `EP_<sha256_12>` from `(symbol, side, entry_ts, exit_ts, total_qty, avg_entry_price, avg_exit_price)`. Stable across rebuilds if fill aggregation is stable.

---

## DLE Shadow Event Chain

```
REQUEST (v1)  →  DECISION (v1|v2)  →  PERMIT (v1|v2, suppressed on DENY)  →  LINK (v1)
                     ↓                        ↓                                  ↓
              verdict, regime,          TTL, scope,                     ts, request_id,
              context_snapshot,         action snapshot,                decision_id,
              provenance                provenance                     permit_id (nullable)
```

Written to: `logs/execution/dle_shadow_events.jsonl` (append-only, manifest-registered).

---

## Episode Binding Algorithm

```
Shadow log  →  Index by (symbol, strategy, action)  →  Binary search nearest-in-time
                                                            ↓
Episode.entry_ts  ↔  nearest ENTRY LINK (±120s, then ±600s)
Episode.exit_ts   ↔  nearest EXIT LINK (±120s, then ±600s)
                                                            ↓
                                                     Disambiguation by strategy
                                                     Ambiguous → flag, don't guess
                                                     Missing → flag, don't guess
                                                            ↓
                                              authority.entry = {request_id, decision_id, permit_id}
                                              authority.exit  = {request_id, decision_id, permit_id}
                                              regime_at_entry = DECISION.context_snapshot.regime
                                              regime_at_exit  = DECISION.context_snapshot.regime
```

---

## Authority Stats (per rebuild)

| Metric | Purpose |
|--------|---------|
| `entry_coverage_pct` | % of episodes with entry authority bound |
| `exit_coverage_pct` | % of episodes with exit authority bound |
| `ambiguous_count` | Episodes with at least one ambiguous binding |
| `missing_count` | Episodes with at least one missing binding |
| `permit_null_on_executed_entry_count` | SHADOW_INCONSISTENCY: permit_id null on executed entry |
| `max_time_delta_s_entry` | Worst-case time gap for entry binding |
| `max_time_delta_s_exit` | Worst-case time gap for exit binding |

---

## What enforcement would require (B.5 checklist)

Phase B is **observation-only**. The following must be satisfied before any enforcement gate can be activated:

### Prerequisites

- [ ] **Shadow coverage baseline** — `entry_coverage_pct ≥ 95%` and `exit_coverage_pct ≥ 95%` sustained over 7+ days of live trading
- [ ] **Zero ambiguity** — `ambiguous_count == 0` for the same 7-day window (matching algorithm is tight enough)
- [ ] **Permit consistency** — `permit_null_on_executed_entry_count == 0` (every executed entry has a PERMIT)
- [ ] **Time delta budget** — `max_time_delta_s_entry < 30s` and `max_time_delta_s_exit < 60s` (binding is within TTL window)
- [ ] **Exit reason coverage** — 0 episodes with `exit_reason == "unknown"` in production

### B.5 enforcement rehearsal (metrics-only, zero behavior change)

- [ ] Count executed orders that **would have been rejected** under "no permit = no order"
- [ ] Track as `enforcement_rehearsal.would_reject_count` metric
- [ ] Track `enforcement_rehearsal.would_reject_pct` (% of executed orders without valid permit)
- [ ] Track `enforcement_rehearsal.would_reject_by_reason` (breakdown by denial code)
- [ ] Define threshold: enforcement can activate only when `would_reject_pct < 1%` sustained for 7 days
- [ ] Dashboard tile showing rehearsal metrics (read-only from state file)
- [ ] Postmortem required for every `would_reject` instance — is it a shadow gap or a real violation?

### B.5 → Phase C gate

- [ ] All B.5 prerequisites green for 7 consecutive days
- [ ] `would_reject_pct == 0%` for 48+ hours
- [ ] Manual sign-off documented in `docs/`
- [ ] Rollback plan documented (feature flag to disable enforcement)
- [ ] Phase C: DLE enforcement active — permit required for order submission
