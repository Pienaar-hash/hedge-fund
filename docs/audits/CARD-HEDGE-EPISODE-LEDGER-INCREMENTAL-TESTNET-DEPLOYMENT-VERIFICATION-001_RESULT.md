# CARD-HEDGE-EPISODE-LEDGER-INCREMENTAL-TESTNET-DEPLOYMENT-VERIFICATION-001

## Final disposition

`rolled_back_runtime_failure`

The testnet executor loaded `dec9acc7` from `2026-07-12T15:38:07Z` until
rollback began at `2026-07-12T15:39:48Z`. It was returned to `4aa51102`
immediately after the first candidate ledger-due cycle.

## Deployment and rollback

| Item | Value |
| --- | --- |
| Candidate commit | `dec9acc78b18352ea01bae42edfd059592517a69` |
| Candidate PID | `360065` |
| Candidate runtime heartbeat SHA | `p6-mainline-landing-2026-04-29-55-gdec9acc7` |
| Candidate terminal heartbeat | `2026-07-12T15:38:54.334456Z`, `COMPLETED` |
| Rollback commit | `4aa5110268db331baae022d5f818ea99b47b0256` |
| Rollback PID | `360171` |
| Rollback runtime heartbeat SHA | `p6-mainline-landing-2026-04-29-52-g4aa51102` |
| Rollback terminal heartbeat | `2026-07-12T15:40:30.751667Z`, `COMPLETED` |
| Routing | `ENV=testnet`, `DRY_RUN=0` |
| Positions before / after | `[]` / `[]` |

## Rollback trigger

The first candidate rebuild completed its publication heartbeat but recorded:

```json
{
  "mode": "full_fallback",
  "duration_ms": 11590.017,
  "fallback_reason": "replay_divergence",
  "ledger_hash": "12fa34ab4823d3f9c7128f047d771be31caf31fd81c06aa0a02a63d730623d3b",
  "checkpoint_version": "episode_ledger_incremental_v2"
}
```

The retained live checkpoint was v2 with ledger hash
`2d780f3cf6536dfb2774e7f1e81b7e3133f876f0da82b210863157279ce6b8cb`.
The replay-divergence path withheld a matching checkpoint, so a safe
checkpoint-backed continuation could not be established. This met the card's
immediate rollback criterion.

During the brief candidate run, the executor also modified the tracked runtime
cache `config/exchange_precision_cache.json`, outside the authorized
ledger/checkpoint/telemetry surfaces. No manual order, credential, risk,
strategy, leverage, or exchange-mode change was made. NAV changed naturally
from `6749.61134868` before deployment to `6748.90734868` after rollback.

## Limits

The hard rollback occurred after one candidate cycle. Three checkpoint-backed
candidate cycles, natural append-path observation, and frozen same-corpus
full-versus-live semantic comparison were not run and are not inferred.

The immutable external evidence bundle, including pre-deployment source/NAV/
authority/ledger/checkpoint manifests and failure captures, is retained at
`/tmp/card-episode-ledger-incremental-testnet-deployment-20260712T1535Z`.
