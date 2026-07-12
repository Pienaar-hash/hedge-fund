# Episode Ledger Provenance-Gated Deployment Protocol

## Purpose

This protocol defines the mandatory control plane for any future episode-ledger incremental candidate. It is fail-closed by design. No candidate may be deployed unless runtime provenance capture, atomic snapshotting, offline full/replay equivalence, checkpoint consistency, bundle attestation, post-start validation, and rollback preparation have all completed with explicit evidence.

This protocol does not authorize deployment of `dec9acc7`. That candidate remains prohibited.

## State Machine

Machine-readable state and transition rules are defined by:

- `config/episode_ledger_deployment_protocol.schema.json`
- `execution/episode_ledger_deployment_protocol.py`

Required states:

- `UNPREPARED`
- `PREFLIGHT_CAPTURED`
- `SNAPSHOT_FROZEN`
- `OFFLINE_EQUIVALENCE_PASSED`
- `RUNTIME_PROVENANCE_VERIFIED`
- `DEPLOYMENT_AUTHORIZED`
- `DEPLOYING`
- `POST_START_VALIDATING`
- `ACCEPTED`
- `ROLLED_BACK`
- `FAILED_CLOSED`

Transition authority:

- `protocol_runner` advances provenance, snapshot, equivalence, and validation gates.
- `deployment_authorizer` advances `DEPLOYMENT_AUTHORIZED` and `ACCEPTED`.
- `supervisor_controller` advances `DEPLOYING` and `ROLLED_BACK`.

Terminal states:

- `ACCEPTED`
- `ROLLED_BACK`
- `FAILED_CLOSED`

Immediate rollback conditions:

- post-start loaded module hash differs from the deployment bundle hash
- unauthorized filesystem mutation outside the allowlist
- post-start full/replay comparison mismatch
- supervisor health failure after candidate start

Attempt-invalidating conditions:

- preexisting ledger/checkpoint inconsistency without a validated reconciliation stage
- inability to attest loaded module bytes
- dirty candidate worktree
- candidate module hash equal to the prohibited candidate hash
- snapshot freeze/copy/resume failure
- equivalence mismatch on any required semantic surface

## Mandatory Preflight Gate

Before any stop, restart, checkout, or deployment mutation, capture:

- live PID
- process start time
- command line
- cwd
- interpreter path and `/proc/<pid>/exe`
- repository provenance as contextual metadata only
- loaded module attestation where determinable
- ledger semantic hash
- checkpoint ledger hash
- NAV hash
- execution-log and DLE identities with terminal anchors
- open-position state
- supervisor state

Ledger/checkpoint mismatch must fail closed with `FAILED_CLOSED_PREEXISTING_INCONSISTENCY` unless a separate validated reconciliation stage exists.

## Atomic Snapshot Requirement

The protocol requires the same bounded freeze proven in `d450db17`:

1. stop the source-owning process
2. verify stopped state
3. copy all bound inputs
4. record timestamps, signals, and copied identities
5. resume the process
6. fail closed if resume cannot be verified

Sequential live copying without a frozen boundary is not acceptable.

## Deployment Bundle

Each deployment attempt must produce one immutable bundle that binds:

- candidate commit SHA
- clean-worktree proof
- candidate module-content hash
- callable fingerprints
- interpreter identity
- dependency/environment fingerprint
- runtime configuration hashes
- resolved source and destination paths
- canonicalization identifier
- NAV semantic-input declaration
- test results
- atomic snapshot manifest
- full/replay equivalence results
- checkpoint compatibility result
- rollback target and rollback procedure
- authorization record
- bundle checksum manifest

Repository HEAD is contextual metadata only. Authorization is tied to the attested module bytes in the bundle.

## Equivalence Gate

Using only the immutable snapshot, the candidate must pass:

- `A`: full rebuild
- `B`: candidate initial build
- `C`: candidate replay from captured checkpoint
- `D`: candidate replay rerun
- `E`: controlled append candidate replay
- `F`: controlled append full baseline

Required equality surfaces:

- canonical ledger hash
- episode count
- ordered UID hash
- per-episode payload hash
- authority hash
- aggregate statistics hash
- reconciliation hash
- exit-distribution hash
- NAV semantic-input hash

Any semantic difference blocks authorization.

## Candidate Freshness Rules

The protocol must reject:

- `dec9acc7`
- any candidate whose module hash matches the prohibited candidate module hash
- dirty candidates
- candidates whose loaded bytes cannot be tied to the bundle

Future real candidates must also prove derivation after this protocol baseline.

## Authorized Mutation Set

Authorized during deployment:

- candidate code checkout or copy
- declared bytecode/cache generation
- `logs/state/episode_ledger.json`
- `logs/state/episode_ledger_checkpoint.json`
- `logs/execution/episode_ledger_rebuild.jsonl`
- deployment evidence outputs
- supervisor transition files

Unauthorized unless separately approved:

- exchange precision cache
- strategy or risk configuration
- NAV history
- execution logs
- DLE logs
- positions state
- unrelated state surfaces

Unexpected mutation triggers rollback or failed-close classification.

## Post-Start Validation Gate

Acceptance requires:

- post-start loaded module attestation
- loaded hash equals bundle hash
- expected cwd and resolved paths
- ledger/checkpoint consistency
- first rebuild outcome captured
- post-start atomic snapshot
- post-start canonical full/replay comparison
- no unauthorized filesystem mutation
- healthy supervisor state
- zero unexpected exchange activity
- position-state confirmation
- explicit acceptance record

A started process is not an accepted deployment.

## Rollback

Rollback must:

1. preserve pre-deployment, failed-candidate, and post-start evidence
2. restart the declared rollback target
3. verify rollback loaded bytes against the rollback bundle target
4. record the rollback reason and resulting attestation

Reporting repository HEAD alone is insufficient.
