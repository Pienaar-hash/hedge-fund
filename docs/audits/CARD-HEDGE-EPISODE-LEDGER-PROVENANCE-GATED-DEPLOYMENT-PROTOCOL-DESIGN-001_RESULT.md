# CARD-HEDGE-EPISODE-LEDGER-PROVENANCE-GATED-DEPLOYMENT-PROTOCOL-DESIGN-001 Result

## Boundary

Starting boundary:

- Forensics baseline commit: `d450db17c1bd089426b9e7c35b84172876fa471d`
- Live executor commit: `4aa5110268db331baae022d5f818ea99b47b0256`
- Live executor PID at prior closeout: `360171`
- Live executor restart during this design card: `none`
- Environment: `Binance Futures testnet`
- `dec9acc7` redeployment remained prohibited throughout

Ending boundary:

- No live deployment occurred
- No live executor stop, restart, or checkout occurred
- No live ledger, checkpoint, NAV, execution log, DLE log, positions state, or runtime configuration file was modified
- Final rehearsal evidence directory: `/tmp/card-episode-ledger-deployment-protocol-rehearsal-20260712T172730Z`
- Rehearsal checksum manifest SHA-256: `5d32aaca9eab051d53cb4871e11404de025c603b784569ffce60ae45c3b291ac`

## Commands

```bash
PYTHONPATH=. /root/hedge-fund/venv/bin/python -m pytest tests/unit/test_episode_ledger_deployment_protocol.py tests/integration/test_episode_ledger_deployment_protocol.py -q
PYTHONPATH=. /root/hedge-fund/venv/bin/python scripts/rehearse_episode_ledger_deployment_protocol.py --output-dir /tmp/card-episode-ledger-deployment-protocol-rehearsal-20260712T172730Z
(cd /tmp/card-episode-ledger-deployment-protocol-rehearsal-20260712T172730Z && sha256sum -c SHA256SUMS)
git diff --check
PYTHON=/root/hedge-fund/venv/bin/python make test-fast
PYTHON=/root/hedge-fund/venv/bin/python make test-runtime
PYTHON=/root/hedge-fund/venv/bin/python make test
```

## Protocol Artifacts

Created artifacts:

- `docs/deployment/EPISODE_LEDGER_PROVENANCE_GATED_DEPLOYMENT_PROTOCOL.md`
- `config/episode_ledger_deployment_protocol.schema.json`
- `scripts/rehearse_episode_ledger_deployment_protocol.py`
- `tests/unit/test_episode_ledger_deployment_protocol.py`
- `tests/integration/test_episode_ledger_deployment_protocol.py`

The protocol implementation defines:

- a machine-validated state model with explicit authority and evidence requirements for every transition
- bundle validation that binds candidate commit, clean-worktree proof, module bytes, callable fingerprints, runtime config hashes, snapshot manifest, equivalence results, rollback target, and authorization record
- mandatory preflight provenance capture
- mandatory `SIGSTOP`/copy/`SIGCONT` atomic snapshot capture
- pre-deployment and post-start equivalence gates using immutable snapshot bytes only
- freshness rejection for `dec9acc7`, byte-identical module hashes, and dirty candidates
- unauthorized mutation detection and rollback verification

## Rehearsal Evidence

Rehearsal root:

- `/tmp/card-episode-ledger-deployment-protocol-rehearsal-20260712T172730Z`

Top-level rehearsal proof:

- `rehearsal_report.json` passed: `true`
- `SHA256SUMS` verified clean

Happy-path private deployment rehearsal:

- final state: `ACCEPTED`
- private candidate commit: `c7143a62a59486df2fa68f7e4afe9be7d61caa47`
- private candidate module hash: `908e2e3f83682f8eec08cc163a7827cbeccd079a2f30c4adb7bb4abe076aa8ca`
- rollback commit: `1f10d07291f981c6a2e5c580c1d8ec4dd0062891`
- rollback module hash: `0fa8ab04fbf046d812d17ffe3a8e540f949208f4ef5de86a728f877013f89911`
- happy-path bundle hash: `519fd930978a3a8d70b2b10266b6b151a5b0a9da697a0b819c2df8ce72ea3003`
- happy-path bundle checksum manifest SHA-256: `00743d02022e998afe4a005aeed39ad465ad0bc2f1d35cfd339c8986a82efa33`

Pre-deployment snapshot proof:

- method: `process_sigstop_copy_resume`
- snapshot started: `2026-07-12T17:27:38.098260+00:00`
- process stopped: `2026-07-12T17:27:38.100837+00:00`
- process resumed: `2026-07-12T17:27:38.105323+00:00`
- snapshot completed: `2026-07-12T17:27:38.105543+00:00`

Pre-deployment equivalence proof:

- passed: `true`
- first semantic difference: `null`
- `A_vs_B`: `null`
- `A_vs_C`: `null`
- `A_vs_D`: `null`
- `E_vs_F`: `null`
- checkpoint compatibility: `true`

Post-start validation proof:

- loaded module hash matched bundle hash: `908e2e3f83682f8eec08cc163a7827cbeccd079a2f30c4adb7bb4abe076aa8ca`
- post-start snapshot method: `process_sigstop_copy_resume`
- post-start full/replay equivalence passed: `true`
- unexpected mutation count on accepted path: `0`

## Failure-Rehearsal Matrix

Expected fail-closed and rollback scenarios all behaved as designed:

- `reject_prohibited_candidate` -> `FAILED_CLOSED_PROHIBITED_CANDIDATE`
- `reject_dirty_worktree` -> `FAILED_CLOSED_DIRTY_WORKTREE`
- `reject_module_hash_mismatch` -> `FAILED_CLOSED_MODULE_HASH_MISMATCH`
- `reject_preexisting_inconsistency` -> `FAILED_CLOSED_PREEXISTING_INCONSISTENCY`
- `reject_snapshot_failure` -> `FAILED_CLOSED_SNAPSHOT_FAILED`
- `reject_equivalence_failure` -> `FAILED_CLOSED_EQUIVALENCE_FAILED`
- `rollback_unauthorized_mutation` -> `ROLLED_BACK`

The unauthorized-mutation rollback preserved evidence and restarted the declared rollback target. The recorded rollback reason was `unexpected filesystem mutation`.

## Validation

Passed:

- `git diff --check`
- focused protocol unit and integration tests
- `make test-fast`
- `make test-runtime`
- `make test`

## Disposition

`deployment_protocol_proven`

Basis:

- the state machine is explicit and machine-validated
- every authorization transition requires explicit evidence
- the bundle binds commit, module bytes, runtime inputs, snapshot manifest, equivalence results, and rollback target
- prohibited, dirty, mismatched, inconsistent, snapshot-failure, and equivalence-failure scenarios all failed closed
- unauthorized mutation triggered rollback with rollback-byte verification
- the complete rehearsal succeeded entirely within private roots and private dummy processes

## Required Explicit Statements

- No deployment occurred during this card.
- `dec9acc7` remains prohibited from redeployment.
- No live process or live state was changed by this design card.
