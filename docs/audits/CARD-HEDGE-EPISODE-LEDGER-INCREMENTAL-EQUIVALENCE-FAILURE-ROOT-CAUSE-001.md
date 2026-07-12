# CARD-HEDGE-EPISODE-LEDGER-INCREMENTAL-EQUIVALENCE-FAILURE-ROOT-CAUSE-001

## Executive verdict

**Primary classification:** `canonicalization_or_hash_defect`.

The reported `12fa…` versus `0e02…` mismatch is not reproducible as an
incremental event-ingestion divergence.  Its first and only observable field
difference is `stats.max_drawdown_pct`, an aggregate derived from the
independently written `logs/state/nav_state.json`.  That auxiliary input was
not frozen by the historical comparison.  Episodes, UIDs, authority bindings,
reconciliation, exit reasons, PnL, fees, prices, and quantities all match.

With execution logs, DLE authority log, persisted ledger/checkpoint, and NAV
state frozen together, the known-good full builder, candidate clean first run,
and two candidate checkpoint-backed no-op rebuilds all hash to:

```
2d780f3cf6536dfb2774e7f1e81b7e3133f876f0da82b210863157279ce6b8cb
```

This removes the specific basis for the prior rollback, but does **not**
authorize redeployment.  Review found two incremental-only equivalence seams
that were not exercised by this corpus.  No runtime process, live file,
checkpoint, configuration, position, or exchange state was changed.  The
executor remains on known-good `4aa51102`; `f9250de1` was not redeployed.

**Candidate disposition:** `repairable_multiple_defects`.

## Boundaries and frozen corpus

| Item | Value |
| --- | --- |
| Candidate | `f9250de18b735d24c09349b1b01786c002299165` |
| Known-good | `4aa5110268db331baae022d5f818ea99b47b0256` |
| Immutable local corpus | `/tmp/card_episode_ledger_equivalence_20260712` |
| Execution source order | `.3`, `.2`, `.1`, then active; reverse lexical rotations followed by active |
| Loader contract | positive `order_fill`, dedup `(symbol, positionSide, side, ts_fill_first, orderId)` |

The corpus includes all files read by episode construction: execution logs,
`dle_shadow_events.jsonl`, `nav_state.json`, captured ledger, and captured
checkpoint.  Its immutable `SHA256SUMS`, `file_metadata.tsv`, and
`line_counts.tsv` are retained in the corpus directory.

| File | Bytes | Lines | SHA-256 |
| --- | ---: | ---: | --- |
| `orders_executed.3.jsonl` | 9,999,850 | 13,435 | `f68b6b61abb2f7780ee5e030e323f00b72e02844a496b6e08273e2a3d6aa6489` |
| `orders_executed.2.jsonl` | 9,999,664 | 12,168 | `96a48fe72ca984c063c04b1f5e8586528dfb9333059efe845459e72c724c8f72` |
| `orders_executed.1.jsonl` | 9,999,583 | 11,405 | `760f379bdd163051cc58da01a563639e6aad161290a962b961cca8af572a9e2b` |
| `orders_executed.jsonl` | 2,585,098 | 3,244 | `976569661cf5d4e0be2ad470e2b9925841ab6bda61039f9843a21af8694e4490` |
| `dle_shadow_events.jsonl` | 52,705,849 | 69,358 | `c6b81c25a9cb34838b34cbe81b1187ec2681ec4474665be7afde84433d4044cb` |
| `nav_state.json` | 70,913 | n/a | `d57727f07b070e49eb1f39f11f4d5f8a1af91247ec6be5e78f04657ae5a80a6d` |
| captured ledger | 3,572,072 | n/a | `5a3a63ebb9b0bde9e2ec1da98b425c76aa4e64591eea2b499844b718a567793d` |
| captured checkpoint | 1,478,667 | n/a | `75a93277877f66f049fe28dfb02cfcc46a8044fb4315bc5ed29de6d82c843e3d` |

All execution files were newline-terminated and valid JSON.  They held 16,127
positive-fill lines; normal dedup produced 15,855 events (272 duplicates). The
captured checkpoint recorded those four sources, 15,855 identities, and hash
`12fa…`.

## Reproduction, hashes, and canonical diff

Detached `4aa51102` and `f9250de1` worktrees each read a private copy of the
frozen corpus and wrote only to private temporary paths.  Canonicalization is
candidate `_canonical_ledger_hash`: sorted compact UTF-8 JSON with **only**
`last_rebuild_ts` excluded.  No semantic field was removed or tolerated.

| Reconstruction | Episodes | Canonical hash | Drawdown % |
| --- | ---: | --- | ---: |
| Captured historical snapshot | 1,627 | `12fa34ab4823d3f9c7128f047d771be31caf31fd81c06aa0a02a63d730623d3b` | 14.74 |
| Known-good clean full | 1,627 | `2d780f3cf6536dfb2774e7f1e81b7e3133f876f0da82b210863157279ce6b8cb` | 21.84 |
| Candidate initial, no checkpoint | 1,627 | `2d780f3cf6536dfb2774e7f1e81b7e3133f876f0da82b210863157279ce6b8cb` | 21.84 |
| Candidate checkpoint no-op | 1,627 | `2d780f3cf6536dfb2774e7f1e81b7e3133f876f0da82b210863157279ce6b8cb` | 21.84 |
| Candidate checkpoint no-op rerun | 1,627 | `2d780f3cf6536dfb2774e7f1e81b7e3133f876f0da82b210863157279ce6b8cb` | 21.84 |

The checkpoint contains live inodes.  For the isolated copy, only its source
identities were translated to copy inodes while retaining offsets and bounded
content anchors; this exercises the no-op path instead of intentionally
triggering the conservative inode-replacement fallback.  No source or ledger
content was changed.

The machine-readable comparison is
[the divergence report](CARD-HEDGE-EPISODE-LEDGER-INCREMENTAL-EQUIVALENCE-FAILURE-ROOT-CAUSE-001_DIVERGENCE_REPORT.json).
The historical snapshot's first difference is exactly:

```json
{
  "scope": "stats",
  "episode_uid": null,
  "episode_index_full": null,
  "episode_index_incremental": null,
  "field_path": "stats.max_drawdown_pct",
  "full_value": 21.84,
  "incremental_snapshot_value": 14.74
}
```

Top-level schema/keys, episode count, ordered UID sequence, UID set, every V1
episode, every V2 authority payload, authority coverage/flags, reconciliation,
exit distribution, PnL, fees, quantities, and prices are equal.  The same-frozen
full versus reproduced incremental comparison has no first difference.

## Source-event and algorithm trace

There is no differing entry, partial fill, close/reduce event, startup record,
rotation boundary, malformed record, duplicate, or authority chain.  Both
paths consume the same 15,855 deduplicated fills, group and stable-sort them by
the same keys, and bind the same DLE index.  Authority results are identical:
563 entry bindings (34.6%), 88 exit bindings (5.4%), 1,541 missing flags,
zero ambiguous, maximum entry delta 18.22 seconds, and maximum exit delta
583.49 seconds.

The cumulative PnL sequence is also identical.  Its maximum drawdown numerator
is 1,473.88; the trough is `EP_1450` at
`2026-07-11T15:44:45.184171+00:00`.  The only changed input is the denominator:

```
captured: round(1473.88 / 10000 * 100, 2)       = 14.74
frozen:   round(1473.88 / 6747.24534868 * 100, 2) = 21.84
```

10,000 is the builder's fallback/effective historical baseline.  The prior NAV
file was not preserved, so its exact old contents cannot be recovered; the
arithmetic and controlled same-corpus reruns prove NAV timing caused the
reported semantic-hash failure, not episode reconstruction.

## Checkpoint transaction analysis

Candidate writes an atomically replaced ledger first and then an atomically
replaced checkpoint.  The pair is recoverable but not a single transaction:

| Crash point | State | Recovery |
| --- | --- | --- |
| before ledger replacement | old pair | normal prior state |
| after ledger, before checkpoint | new ledger + old checkpoint | hash mismatch forces full fallback |
| after checkpoint replacement | matching pair | normal incremental resume |

Normal flow cannot checkpoint ahead of ledger.  No captured evidence shows a
partial ledger, checkpoint-ahead state, checkpoint/hash mismatch, or source
snapshot mismatch.  A source replacement race is conservatively handled by a
later validation/fallback.  Transaction ordering did not contribute here.

## Candidate review and hypothesis classification

| Failure class | Result |
| --- | --- |
| cursor/offset, rotation, duplicate/omitted event, partial line, late event | eliminated for the frozen corpus |
| open-state persistence/restoration, completed-episode mutation, UID/ID instability | eliminated: every episode and order match |
| authority incrementality/ties/regimes/flags | eliminated: every V2 payload and coverage field match |
| exit normalization, reconciliation-as-trade, serialization | eliminated for observed mismatch |
| output ordering only | eliminated: no order difference exists |
| checkpoint transaction | eliminated as cause; recovery is conservative |
| aggregate recalculation | exact mismatch is NAV-dependent drawdown percentage |
| canonicalization/hash | proven primary cause: hash includes NAV-dependent public aggregate without freezing NAV provenance |

Two paths are not definitionally equivalent and remain required repair scope:

1. Candidate applies a newly read multi-event batch in file order. It rejects
   an event older than the preceding checkpoint state, but does not per-group
   sort a batch such as `t3`, `t2` where both follow previous `t1`; the full
   builder stable-sorts that group.
2. Candidate `_metadata_pnl_apply` rounds after each appended event, while full
   `_compute_metadata_pnl` sums raw values then rounds once. Decimal-heavy
   appends can diverge.

Neither occurred here, so neither is the root cause of `12fa…`; both prevent
declaring the candidate definitionally equivalent.

## Smallest safe correction and regression proof

Do not weaken the public semantic hash or remove `max_drawdown_pct`.  The proof
correction is to freeze and hash `nav_state.json` with execution/DLE sources,
record `total_equity`, and run full and incremental outputs only from private
copies of that complete corpus.

A narrow repair card must also sort accepted new events per group using the
full-builder key (or fall back), and retain unrounded metadata-PnL accumulators
until publication.  Its required fixture must include rotations, active log,
DLE log, and NAV; compare per-episode hashes, UID order/set, authority fields,
all aggregate/reconciliation fields, and final hash across first full, no-op,
append, and restart.  It must fail `f9250de1` for out-of-order and decimal-fee
appends, then pass only after exact equivalence is restored.

## Final disposition

`repairable_multiple_defects`.  The original mismatch is fully explained by an
unfrozen NAV-dependent aggregate.  Keep `f9250de1` undeployed and testnet on
`4aa51102` until the stated narrow repair/proof card closes.
