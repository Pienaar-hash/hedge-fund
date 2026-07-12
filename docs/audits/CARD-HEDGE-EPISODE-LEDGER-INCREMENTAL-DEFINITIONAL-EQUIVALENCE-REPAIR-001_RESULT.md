# CARD-HEDGE-EPISODE-LEDGER-INCREMENTAL-DEFINITIONAL-EQUIVALENCE-REPAIR-001

## Result

Disposition: `offline_proven_not_deployed`.

Starting branch/commit: `card/episode-ledger-equivalence-forensics-001` at
`4b3ff2a086c2a565d41bf0ad9149aff95882f3b4`. The deployed executor remains
`4aa5110268db331baae022d5f818ea99b47b0256`; the unrepaired candidate examined
in an isolated detached worktree was
`f9250de18b735d24c09349b1b01786c002299165`. No deploy, Supervisor action,
exchange action, or repository `logs/` write occurred.

## Repairs

1. Incremental appends now preserve the full builder's first-seen group order,
   then stable-sort every newly discovered group batch by the canonical event
   sort key before replaying it. This covers non-canonical source order,
   interleaving, and equal timestamps without normalizing away any event.
2. Metadata PnL checkpoint state now retains raw gross-PnL and fee totals and
   rounds only when producing the public aggregate. The checkpoint schema is
   versioned as `episode_ledger_incremental_v2`, forcing a conservative full
   fallback from v1's rounded accumulator state.
3. NAV has an explicit module path and is fingerprinted in checkpoint
   provenance. It remains part of the semantic ledger output and canonical
   hash; `stats.max_drawdown_pct` is not excluded.

Files changed: `execution/episode_ledger.py`,
`tests/unit/test_episode_ledger_incremental.py`,
`tests/integration/test_episode_ledger_incremental_equivalence.py`,
`v7_manifest.json`, `EXECUTION_ARCHITECTURE.md`, `docs/active/OPERATIONS.md`,
`docs/active/TESTING.md`, this result record, and its JSON matrix.

## Frozen corpus and assertion

`tests/integration/test_episode_ledger_incremental_equivalence.py` creates a
new temporary corpus per case. Its manifest records active and rotated source
files in exact loader order; byte sizes, line counts, SHA-256 values and inode
identities; DLE authority log; NAV state; starting ledger/checkpoint; checkpoint
offsets/anchors; canonicalization; and builder identifiers. It rejects missing
inputs and detects silent mutation. It never uses the repository `logs/` tree.

The shared assertion compares all public top-level keys, episode count, V1
payloads, V2 payloads including authority/flags, UID sequence and set,
reconciliation, exit distribution, all quantities/prices/PnL fields and all
statistics (including `max_drawdown_pct`), then requires equal canonical hash.
Only `last_rebuild_ts` is excluded.

The detailed machine-readable matrix is
[`..._MATRIX.json`](CARD-HEDGE-EPISODE-LEDGER-INCREMENTAL-DEFINITIONAL-EQUIVALENCE-REPAIR-001_MATRIX.json).

## Adversarial proof

The isolated candidate command was:

```bash
PYTHONPATH=/tmp/hedge-f9250de1 pytest -q /tmp/test_f9250de1_adversarial.py
```

It passed two negative assertions: unmodified `f9250de1` produces a different
canonical hash from full rebuild for an unsorted append, and metadata gross PnL
is `0.00` incrementally versus `0.01` in full rebuild for the decimal fixture.

The repaired targeted command was:

```bash
pytest -q tests/integration/test_episode_ledger_incremental_equivalence.py \
  tests/unit/test_episode_ledger_incremental.py tests/unit/test_episode_ledger.py
```

Result: `46 passed`.

Additional proof commands completed with exit status 0:

```bash
git diff --check
make test-runtime     # 5 passed, 4 skipped
make test-fast
make test
python -m json.tool v7_manifest.json
```

Representative equal full/incremental hashes:

| Case | Hash |
| --- | --- |
| clean full/no-op | `b5139f15dd0e13b77652ba82df4d457d6003d58fa52e9242251685f3b0e46d14` |
| unsorted append | `aabcb020ca39bdf77e3f06ccfc29bb0a37609ed4bff1594d20a54f7fd507f34f` |
| decimal metadata PnL | `863e00bd9640b9e9bf37486d2c7f99e3426ec3d2180ddc09e1f5a0c7bc46204e` |

The decimal case publishes `metadata_pnl.gross_pnl == 0.01` after two `0.004`
events. NAV-only changes visibly alter drawdown and the canonical hash.

## Conservative recovery and limitations

Targeted fixtures verify valid restart, stale ledger/checkpoint recovery,
checkpoint mismatch, unproven inode replacement fallback, partial trailing-line
completion, duplicate append, rotation, and repeated no-op/recovery idempotence.
Existing checkpoint-anchor, truncation, malformed checkpoint, and fail-open
telemetry unit coverage remains in `tests/unit/test_episode_ledger_incremental.py`.

The historical `12fa…` versus `0e02…` divergence remains unproven because its
contemporaneous `0e02…` payload and NAV input were not retained. This work does
not change that conclusion and does not authorize deployment.
