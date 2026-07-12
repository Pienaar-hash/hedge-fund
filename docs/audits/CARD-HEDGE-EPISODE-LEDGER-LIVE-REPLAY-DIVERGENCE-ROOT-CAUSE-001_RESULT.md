# CARD-HEDGE-EPISODE-LEDGER-LIVE-REPLAY-DIVERGENCE-ROOT-CAUSE-001

## Disposition

`not_reproducible_insufficient_evidence`

`dec9acc7` must **not** be redeployed.  The candidate correctly encountered a
pre-existing inconsistent ledger/checkpoint pair, but the exact candidate-time
inputs needed to reproduce the subsequent `replay_divergence` were not
retained.  This result does not claim that the unavailable historical
`12fa…`/`0e02…` pair has been solved.

## Executive finding

The immutable pre-candidate state was already inconsistent before candidate
PID `360065` started:

| retained pre-deployment artifact | semantic hash | relevant fact |
| --- | --- | --- |
| `episode_ledger.json` | `12fa34ab4823d3f9c7128f047d771be31caf31fd81c06aa0a02a63d730623d3b` | `last_rebuild_ts=2026-07-12T15:32:37.740802+00:00` |
| `episode_ledger_checkpoint.json` | `2d780f3cf6536dfb2774e7f1e81b7e3133f876f0da82b210863157279ce6b8cb` | v2 checkpoint, `checkpoint_ts=2026-07-12T15:30:15.976718+00:00` |

The candidate's first cycle then retained exactly that public `12fa…` ledger
semantic payload (only `last_rebuild_ts` changed) and withheld a replacement
checkpoint because its full-replay comparison reported `replay_divergence`.
The post-failure checkpoint file is byte-identical to the pre-candidate
checkpoint (`0db221fb…`), so no candidate checkpoint was published.

This proves a pre-existing ledger/checkpoint provenance inconsistency and
explains why the candidate entered conservative full fallback.  It does not,
by itself, reproduce why the candidate's two reads during that fallback
diverged: the candidate-cycle execution and DLE source copies, the candidate
process cwd, and the NAV bytes at each read were not captured.

## Immutable evidence inventory

The evidence root is
`/tmp/card-episode-ledger-incremental-testnet-deployment-20260712T1535Z`.
Its original manifests and checksums remain authoritative:

| bundle component | retained artifacts | integrity record |
| --- | --- | --- |
| `pre/` | active `orders_executed.jsonl`; rotations `.1`, `.2`, `.3`; DLE shadow log; NAV; positions; ledger; checkpoint; rebuild and heartbeat telemetry; process/Supervisor/repository/disk/memory capture | `pre/manifest.json`, `pre/SHA256SUMS` |
| `post_deployment_failure/` | candidate-cycle ledger, unchanged checkpoint, candidate rebuild telemetry and heartbeat, NAV, positions, Supervisor status | `post_deployment_failure/failure_manifest.json`, `post_deployment_failure/SHA256SUMS` |
| cache evidence | `exchange_precision_cache.diff`, current and HEAD SHA-256 records | `exchange_precision_cache.evidence.sha256` |

Pre-candidate semantic source order is explicitly recorded as
`orders_executed.3.jsonl`, `.2`, `.1`, then `orders_executed.jsonl`.  The four
captured source hashes are, respectively,
`f68b6b61…`, `96a48fe7…`, `760f379b…`, and `17cf8378…`; the captured DLE hash
is `82cbb84d…`.  The pre NAV copy is `17636dda…`; post NAV is `a8d1235c…`.
The detailed, machine-readable artifact list (sizes, lines, source identities,
anchors, hashes, and absence markers) is in
`CARD-HEDGE-EPISODE-LEDGER-LIVE-REPLAY-DIVERGENCE-ROOT-CAUSE-001_EVIDENCE_MANIFEST.json`.

### Required artifacts not retained

1. A candidate-cycle copy (or hash/line-count/identity manifest) of all four
   execution files.  Candidate telemetry says it read `32,585,312` bytes;
   the frozen pre corpus totals `32,585,053` bytes: a `259` byte difference.
2. A candidate-cycle DLE/authority-log copy and fingerprint.
3. The exact NAV bytes read by each candidate full-builder and replay pass.
   The post capture was taken later and is not a substitute.
4. `/proc/360065/cwd`, its complete environment, and a Supervisor `directory`
   setting.  The captured command is only `./venv/bin/python -m
   execution.executor_live`; captured environment has only `ENV=testnet` and
   `DRY_RUN=0`.
5. Per-read source fingerprints/anchors.  Telemetry records aggregate bytes,
   not the two source snapshots used by `_full_rebuild_with_checkpoint`.
6. The candidate-period executor/Supervisor stderr log excerpt.  The bundle
   retains structured heartbeat and rebuild telemetry but no corresponding
   `/var/log/supervisor/hedge-executor.log` copy.

## Runtime input-resolution trace

At `dec9acc7`, `execution.episode_ledger` sets the ledger, checkpoint, NAV,
DLE, and execution paths to relative `logs/...` paths at import time.  It has
no environment override for these paths.  Its source resolver is the exact
legacy loader order: rotations matching `orders_executed.*.jsonl`, reverse
lexically sorted, then active `orders_executed.jsonl`.  Full construction reads
execution and DLE, then reads `NAV_STATE_PATH`; replay reads execution again.

Thus the relative-path selection and ordering are proven from code and the
pre manifest.  The absolute runtime paths are **not** proven: the candidate
cwd was not retained.  This matters because the code falls back to NAV 10,000
when the relative NAV path is missing, unreadable, or lacks `total_equity`.

The checkpoint did not undergo v1-to-v2 migration on the candidate cycle.  It
was already `episode_ledger_incremental_v2`; its ledger hash disagreed with
the loaded ledger before candidate execution, so `_checkpoint_is_valid()`
necessarily selected `ledger_checkpoint_hash_mismatch` before the conservative
full-replay attempt.  The terminal telemetry reports the fuller diagnostic
`replay_divergence` returned by that attempt.

## Reproduction and exact semantic difference

All replays used a private copy at
`/tmp/card_episode_ledger_live_replay_20260712`, never a live `logs/` path.
Canonicalization is the committed sorted, compact UTF-8 JSON contract with
only `last_rebuild_ts` excluded.

| construction | execution/DLE corpus | NAV basis | hash | replay result |
| --- | --- | --- | --- | --- |
| retained pre ledger | retained pre artifact | effective 10,000 | `12fa34ab…` | n/a |
| full builder, repeated three times | retained pre corpus | captured `total_equity=6749.61134868` | `2d780f3c…` | deterministic |
| dec full+checkpoint replay | retained pre corpus | captured NAV | `2d780f3c…` | matching checkpoint would be produced |
| full builder with NAV deliberately absent | retained pre corpus | code fallback 10,000 | `12fa34ab…` | matching checkpoint would be produced |
| candidate post-failure ledger | retained post artifact | unknown at its actual read | `12fa34ab…` | checkpoint withheld in live telemetry |

The retained pre ledger and the same-corpus full output have exactly one public
semantic difference.  Schema/keys, 1,627 V1 episodes, ordered V2 UIDs and UID
set, all V1/V2 payloads, authority flags/coverage, reconciliation, exit
distribution, prices, quantities, fees, gross/net/cumulative PnL, and every
other aggregate are equal:

```json
{
  "field_path": "$.stats.max_drawdown_pct",
  "retained_12fa_value": 14.74,
  "same_corpus_full_2d_value": 21.84
}
```

The numerator is the identical `max_drawdown_abs=1473.88`.  The exact `12fa…`
semantic hash is reproducible from the frozen execution/DLE corpus when the
builder uses its documented 10,000 NAV fallback.  That is strong evidence of
NAV provenance/input resolution, but cannot prove whether candidate PID
`360065` saw a missing path, a transient unreadable file, or a different NAV
file containing 10,000.

## Checkpoint-provenance timeline

| UTC | artifact/evidence | conclusion |
| --- | --- | --- |
| `15:30:15.976718` | retained v2 checkpoint | references `2d780f…`; source order, offsets and anchors match the frozen pre sources |
| `15:30:16.879816` | rebuild telemetry labelled `g4b3ff2a0` | reports a full fallback and `2d780f…` |
| `15:32:37.740802` | retained pre ledger | public semantic hash is `12fa…`, creating the stale pair before candidate start |
| `15:38:07` | candidate restart request | PID `360065` starts `dec9acc7` |
| `15:38:53.701212` | candidate telemetry | full fallback, `replay_divergence`, `12fa…`, 15,855 events, 32,585,312 bytes |
| `15:38:50.071548` | post-failure ledger | still `12fa…` semantically |
| `15:39:47` | failure capture | checkpoint raw hash remains `0db221fb…`, still references `2d780f…` |

There is an additional code-proven provenance defect: committed `4b3ff2a0`
does not contain incremental ledger code or the v2 schema, yet retained
checkpoint and telemetry label that commit.  The only repository commit that
introduces `episode_ledger_incremental_v2` is `dec9acc7`.  Therefore the
runtime `git_sha` was a repository-HEAD label, not a content attestation of
the loaded module.  It cannot establish the checkpoint's actual builder code.

The checkpoint NAV fingerprint is `cf2be589…`, whereas the retained pre NAV
copy is `17636dda…`; both are distinct from post NAV `a8d1235c…`.  Source
anchors/sizes do match the frozen pre corpus, but the checkpoint's recorded
raw NAV provenance does not.  The v2 checkpoint was not created by a known
clean `dec9acc7` deployment and must be treated as untrusted provenance.

## Principal-hypothesis classification

| hypothesis | classification | evidence |
| --- | --- | --- |
| NAV input/timing difference | plausible-unproven | exact `12fa…` reproduction at fallback 10,000; captured NAV produces `2d…`; candidate-time NAV read absent |
| stale/mismatched checkpoint provenance | proven | pre ledger `12fa…` and pre checkpoint `2d…` already disagree |
| candidate migration read old ledger/new checkpoint | eliminated | pre checkpoint was v2; no v1 migration occurred |
| source mutation during rebuild | not-testable | candidate source snapshots missing |
| differing source discovery order | eliminated for frozen corpus; not-testable live | same resolver/order replays exactly; candidate absolute paths missing |
| source identity/inode handling | not-testable | candidate identities not retained |
| DLE mutation/selection | not-testable | candidate DLE copy/fingerprint missing |
| cwd/path-resolution mismatch | plausible-unproven | all paths are relative; candidate cwd not retained |
| telemetry versus checkpoint canonicalization mismatch | eliminated | both use the committed canonical hash and retained raw payloads verify the stated hashes |
| full-builder nondeterminism | eliminated for frozen corpus | three full replays all produced `2d780f…` |
| checkpoint created from a different corpus | proven for raw NAV provenance | checkpoint NAV SHA differs from retained pre NAV; execution anchors/sizes match pre corpus |
| concurrent writer race | plausible-unproven | candidate byte total is +259, but no per-read snapshots exist |
| precision-cache write affects execution precision only | proven expected startup refresh; unauthorized for card | executor startup calls `refresh_precision_cache()` |
| precision-cache write affects ledger inputs indirectly | no direct path; indirect not-testable | ledger module neither reads nor imports that cache; candidate source artifacts are missing |

## Exchange-precision-cache classification

The preserved diff updates Binance testnet precision filters and adds
`BSBUSDT`.  `execution.executor_live` unconditionally calls
`refresh_precision_cache()` at startup; that function fetches Binance Futures
exchange info and overwrites the relative cache path.  The write is therefore
an expected startup refresh in code, but an unauthorized runtime write for the
deployment card.  `execution.episode_ledger` has no import or read path to
this cache.  It is classified as **unauthorized but ledger-benign on the
retained evidence**, not a proven cause of the replay divergence.

## Commands and non-mutation boundary

Read-only inventory and replay commands included `find`, `sha256sum`, `jq`,
`git show`, `git diff`, and private-copy Python calls to
`build_episode_ledger()` / `_full_rebuild_with_checkpoint()`.  The replay copy
was under `/tmp`; no command targeted the repository's live `logs/` tree.
No Supervisor action, order action, exchange action, live runtime-file edit,
or candidate redeployment occurred under this card.

`git diff --check` passes for these documentation-only, uncommitted files.

## Recommended next card

`CARD-HEDGE-EPISODE-LEDGER-RUNTIME-PROVENANCE-AND-ATOMIC-SNAPSHOT-FORENSICS-001`:
instrument only in an isolated/non-deployed harness (or capture before any
future authorized deployment) the immutable module content hash, cwd, resolved
absolute paths, and a single atomic source/NAV/DLE snapshot consumed by both
full and replay passes.  It must reproduce the first difference from that
single snapshot before any repair or redeployment is considered.
