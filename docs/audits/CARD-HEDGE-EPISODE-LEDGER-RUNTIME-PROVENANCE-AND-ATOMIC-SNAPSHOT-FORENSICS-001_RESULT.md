# CARD-HEDGE-EPISODE-LEDGER-RUNTIME-PROVENANCE-AND-ATOMIC-SNAPSHOT-FORENSICS-001 Result

## Boundary

Starting runtime facts observed on host:

- Live executor process: PID `360171`
- Live executor start: `2026-07-12 15:39:48 UTC`
- Live executor command: `./venv/bin/python -m execution.executor_live`
- Live executor repository HEAD: `4aa5110268db331baae022d5f818ea99b47b0256`
- Live executor repository status: `?? .claude/`
- Repair/forensics implementation worktree: `/root/hedge-fund-episode-ledger-forensics`
- Repair/forensics worktree HEAD: `126916324db80931420e3e660f4369b715121c04`

Ending runtime facts:

- Live executor remained PID `360171`
- Live executor was not restarted
- Live executor repository HEAD remained `4aa5110268db331baae022d5f818ea99b47b0256`
- Final successful evidence directory: `/tmp/card-episode-ledger-runtime-provenance-20260712T1636Z`
- Evidence checksum manifest SHA-256: `dd361a94a3e1ad7405e5eb795d07414ad41b97a7fad1d093c1b73ba17ba828a2`

## Commands

```bash
git worktree add /root/hedge-fund-episode-ledger-forensics 12691632
PYTHONPATH=. /root/hedge-fund/venv/bin/python -m pytest tests/unit/test_episode_ledger_runtime_forensics.py tests/integration/test_episode_ledger_runtime_forensics.py -q
PYTHONPATH=. /root/hedge-fund/venv/bin/python -m pytest tests/unit/test_episode_ledger_incremental.py tests/integration/test_episode_ledger_incremental_equivalence.py tests/unit/test_episode_ledger_runtime_forensics.py tests/integration/test_episode_ledger_runtime_forensics.py -q
PYTHONPATH=. /root/hedge-fund/venv/bin/python scripts/run_episode_ledger_runtime_forensics.py --live-root /root/hedge-fund --live-pid 360171 --evidence-dir /tmp/card-episode-ledger-runtime-provenance-20260712T1636Z --repo-root /root/hedge-fund-episode-ledger-forensics
(cd /tmp/card-episode-ledger-runtime-provenance-20260712T1636Z && sha256sum -c SHA256SUMS)
git diff --check
PYTHON=/root/hedge-fund/venv/bin/python make test-fast
PYTHON=/root/hedge-fund/venv/bin/python make test-runtime
PYTHON=/root/hedge-fund/venv/bin/python make test
```

## Runtime Mutation Record

- Live executor restart: `no`
- Live executor stop method for successful snapshot: `SIGSTOP` then `SIGCONT`
- Successful atomic snapshot boundary:
  - snapshot started: `2026-07-12T16:26:08.949145+00:00`
  - process entered stopped state: `2026-07-12T16:26:08.949452+00:00`
  - process resumed: `2026-07-12T16:26:09.351031+00:00`
  - snapshot completed: `2026-07-12T16:26:09.351188+00:00`
- Prior aborted harness attempts also performed bounded `SIGSTOP`/`SIGCONT` pairs while fixing the isolated forensic wrapper; no restart occurred and no live state file was overwritten.
- Forensic output paths were confined to `/tmp/card-episode-ledger-runtime-provenance-*`; no live log/state path was used as an output target.

## Module Attestation

Repository provenance, loaded-module provenance, and runtime process provenance were separated explicitly.

- Repository provenance for the forensic runner:
  - HEAD `126916324db80931420e3e660f4369b715121c04`
  - worktree status was contextual only
- Loaded module under test:
  - module file: `/root/hedge-fund-episode-ledger-forensics/execution/episode_ledger.py`
  - module SHA-256: `0fa8ab04fbf046d812d17ffe3a8e540f949208f4ef5de86a728f877013f89911`
  - callable source fingerprints captured for `build_episode_ledger`, `rebuild_and_save`, `_load_execution_log`, `_read_new_execution_events`, `_canonical_ledger_hash`
- Runtime process provenance for the forensic runner:
  - PID `364195`
  - interpreter `/root/hedge-fund/venv/bin/python`
  - cwd `/root/hedge-fund-episode-ledger-forensics`
- Live executor process provenance was captured separately in `process_provenance.json`; its repository HEAD was not treated as proof of the module bytes used by the forensic comparison passes.

## Path Resolution

All relevant inputs were captured as resolved absolute paths with device/inode identity.

- Execution log read order:
  1. `/root/hedge-fund/logs/execution/orders_executed.3.jsonl`
  2. `/root/hedge-fund/logs/execution/orders_executed.2.jsonl`
  3. `/root/hedge-fund/logs/execution/orders_executed.1.jsonl`
  4. `/root/hedge-fund/logs/execution/orders_executed.jsonl`
- Active execution log: `/root/hedge-fund/logs/execution/orders_executed.jsonl`
- DLE authority log: `/root/hedge-fund/logs/execution/dle_shadow_events.jsonl`
- NAV input: `/root/hedge-fund/logs/state/nav_state.json`
- Preexisting ledger: `/root/hedge-fund/logs/state/episode_ledger.json`
- Preexisting checkpoint: `/root/hedge-fund/logs/state/episode_ledger_checkpoint.json`
- No successful-path source resolved outside `/root/hedge-fund`

## Atomic Snapshot Result

Snapshot method: `process_sigstop_copy_resume`

This card did not claim atomicity from sequential copy alone. The live executor was stopped before source copying, remained stopped through the copy window, and was resumed only after all bound inputs were copied into the evidence directory. The resulting immutable snapshot includes:

- all four execution-log files selected by the loader
- DLE shadow authority log
- NAV state input
- preexisting ledger
- preexisting checkpoint
- module attestation
- process provenance
- resolved path manifest

## Captured Preexisting Inconsistency

The successful snapshot preserved the same inconsistency reported by prior forensics:

- snapshotted ledger semantic hash: `12fa34ab4823d3f9c7128f047d771be31caf31fd81c06aa0a02a63d730623d3b`
- snapshotted checkpoint `ledger_hash`: `2d780f3cf6536dfb2774e7f1e81b7e3133f876f0da82b210863157279ce6b8cb`

This proves the atomic snapshot did not “heal” the source state before replay; the mismatch existed inside the frozen source set itself.

## Read Fingerprint Summary

Every comparison pass read only private copies under the evidence directory.

- `A_full`: 6 tracked input reads, all within the isolated runtime root
- `B_initial`: 16 tracked input reads, all within the isolated runtime root
- `C_noop`: 18 tracked input reads, all within the isolated runtime root
- `D_noop_rerun`: 17 tracked input reads, all within the isolated runtime root
- `E_append_candidate`: 14 tracked input reads, all within the isolated runtime root
- `E_append_full`: 6 tracked input reads, all within the isolated runtime root

Key replay facts:

- `C_noop` consumed the snapshotted preexisting ledger/checkpoint pair and deterministically fell back with `fallback_reason=ledger_checkpoint_hash_mismatch`
- `D_noop_rerun` was a true checkpoint-backed no-op incremental pass with `bytes_read=0`
- `E_append_candidate` was a true incremental append replay with `bytes_read=715`, `new_events=2`, `episodes_created=1`

## Reproduction Matrix

| Run | Result |
| --- | --- |
| A. full rebuild from immutable snapshot | ledger hash `2d780f3c...`, episodes `1627` |
| B. candidate initial build with no checkpoint | matched A exactly; checkpoint hash `33224871...` |
| C. candidate checkpoint-backed no-op replay | matched A exactly; deterministic full fallback on snapshotted mismatch |
| D. candidate checkpoint-backed no-op rerun | matched A exactly; true incremental no-op, `bytes_read=0` |
| E. candidate replay with controlled append batch | matched appended full baseline exactly; ledger hash `17de0abb...`, episodes `1628` |

Exact semantic diff across required comparisons:

- `A_vs_B`: `null`
- `A_vs_C`: `null`
- `A_vs_D`: `null`
- `E_full_vs_candidate`: `null`

## Final Disposition

`runtime_divergence_eliminated_under_atomic_provenance`

Basis:

- one immutable runtime provenance set was captured and checksum-verified
- the frozen source set preserved the preexisting `12fa34ab...` vs `2d780f3c...` mismatch
- full rebuild and candidate initial/no-op/append paths produced exact semantic equivalence when forced to consume identical immutable bytes
- the checkpoint-backed no-op path did not reproduce the historical divergence; it deterministically rejected the stale snapshotted ledger/checkpoint pair and converged to the full-build result
- the rerun after convergence was a true no-op incremental replay with `bytes_read=0`

## Limitations

- The live executor process at `4aa51102` was the source owner for snapshot capture, not the process running the `12691632` forensic comparison passes.
- Two earlier forensic harness attempts were aborted while hardening the isolated wrapper; only `/tmp/card-episode-ledger-runtime-provenance-20260712T1636Z` is the successful proof artifact.
- The large `dle_shadow_events.jsonl` authority log materially dominates read volume even though it is not the source of the proven semantic mismatch.

## Required Explicit Statement

`dec9acc7` remains prohibited from redeployment.

