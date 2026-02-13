# C.1 Ops Protocol — Entry-Only Enforcement

**Version:** 1.0  
**Phase:** C.1 (entry-only binding)  
**Effective:** When `DLE_ENFORCE_ENTRY_ONLY=1`  
**Owner:** Operator  

---

## 1. Constitutional Guarantee

> Entering risk requires authority. Exiting risk must remain always available.

C.1 blocks **ENTRY orders** that lack a valid DLE permit.  
C.1 **never** blocks exits, closes, stop-losses, or take-profits.

---

## 2. Activation Ladder

| Stage | Flag | Duration | Criteria |
|-------|------|----------|----------|
| **Dry run** | `DLE_ENFORCE_ENTRY_ONLY=0` + `SHADOW_DLE_ENABLED=1` | Pre-flight | B.5 rehearsal running, phase_c_readiness.json healthy |
| **Canary** | `DLE_ENFORCE_ENTRY_ONLY=1` + `DRY_RUN=1` | 1–3 days | Enforcement active on testnet, zero unexpected denials |
| **Live** | `DLE_ENFORCE_ENTRY_ONLY=1` + `DRY_RUN=0` | Ongoing | Production enforcement |
| **C.1 Stable** | — | After 14 consecutive clean days under enforcement | See §6 |

---

## 3. Daily Checkpoint Addendum

Add these checks to the existing [Phase C Daily Checkpoint](PHASE_C_DAILY_CHECKPOINT.md):

| # | Check | Command | Pass Criteria |
|---|-------|---------|---------------|
| 6 | Enforcement active | `jq '.enforcement.enforce_enabled' logs/state/phase_c_readiness.json` | `true` |
| 7 | Entry denial rate | `jq '.enforcement.entry_blocks_pct' logs/state/phase_c_readiness.json` | Expected range for universe |
| 8 | Split-brain count | `jq '.enforcement.split_brain_count' logs/state/phase_c_readiness.json` | `0` (nonzero = divergence) |
| 9 | Denial log growing | `wc -l logs/execution/dle_entry_denials.jsonl` | Consistent with entry_denied |
| 10 | No exit denials | `grep -c '"EXIT"' logs/execution/dle_entry_denials.jsonl` | `0` (constitutional invariant) |

### Quick-Check One-Liner (Enforcement)

```bash
jq '{
  enforce_on: .enforcement.enforce_enabled,
  evaluated: .enforcement.entry_evaluated,
  permitted: .enforcement.entry_permitted,
  denied: .enforcement.entry_denied,
  blocks_pct: .enforcement.entry_blocks_pct,
  split_brain: .enforcement.split_brain_count,
  last_denial: .enforcement.last_denial_symbol,
  last_split: .enforcement.last_split_brain_symbol
}' logs/state/phase_c_readiness.json
```

---

## 4. Ops Ledger Format

Append one line per day to `ops/phase_c_window.log`, extended for C.1:

```
2026-03-01 | Day 1/14 | enforce=ON | denied=0 | permitted=42 | split_brain=0 | CLEAN
2026-03-02 | Day 2/14 | enforce=ON | denied=0 | permitted=38 | split_brain=0 | CLEAN
```

Breach:
```
2026-03-03 | Day 0/14 | enforce=ON | denied=5 | permitted=33 | split_brain=1 | BREACH: split_brain detected — window reset
```

---

## 5. Red-Line Alerts (Immediate Action Required)

| Condition | Severity | Action |
|-----------|----------|--------|
| `split_brain_count > 0` | **CRITICAL** | Investigate immediately. Compare enforcement + rehearsal logs. Consider rollback. |
| Exit order appears in denial log | **CRITICAL** | Constitutional violation. Rollback to `DLE_ENFORCE_ENTRY_ONLY=0` immediately. |
| `entry_blocks_pct > 10%` | **HIGH** | Denial rate abnormal. Check permit pipeline / shadow log freshness. |
| `enforce_enabled=false` when expected `true` | **HIGH** | Flag not set or init failed. Check startup logs. |
| `entry_evaluated=0` after trading window | **MEDIUM** | No entries attempted — possibly upstream signal drought. Verify hydra/sentinel. |

---

## 6. Rollback Criteria

### Instant Rollback (no discussion)

Set `DLE_ENFORCE_ENTRY_ONLY=0` and restart:

1. **Any exit blocked** — Constitutional violation (should be impossible, but if it happens, rollback)
2. **split_brain_count nonzero and rising** — Enforcement and rehearsal have drifted
3. **Legitimate entries denied** — Permit pipeline broken, valid trades being rejected

```bash
# Emergency rollback
export DLE_ENFORCE_ENTRY_ONLY=0
sudo supervisorctl restart hedge:
```

### Considered Rollback (operator judgment)

Review logs, then decide:

1. **entry_blocks_pct significantly higher than rehearsal would_block_pct** — Possible permit timing issue
2. **Repeated single-symbol denials** — Possible stale permit for one asset
3. **NAV stale + enforcement active** — Compounding vetoes

### Post-Rollback

1. Log rollback in `ops/phase_c_window.log`:
   ```
   2026-03-03 | ROLLBACK | DLE_ENFORCE_ENTRY_ONLY → 0 | reason: split_brain detected
   ```
2. Window counter resets to Day 0/14
3. Root-cause before re-enabling
4. Re-enable only after fix committed and tested

---

## 7. "C.1 Stable" Declaration

C.1 is declared stable when **all** of the following hold:

| Criterion | Threshold |
|-----------|-----------|
| Enforcement active | `DLE_ENFORCE_ENTRY_ONLY=1` continuously |
| Consecutive clean days | ≥ 14 days |
| Split-brain count | 0 for the entire window |
| No rollbacks | Zero rollbacks during the window |
| Exit denials | 0 (constitutional invariant) |
| Denial rate | Consistent with rehearsal would_block_pct (±0.01%) |

When met: `phase_c_readiness.json` → `gate_satisfied: true`.

After declaration:
- C.1 becomes **baseline** (not experimental)
- Rollback still available but treated as incident
- Phase C.2 planning may begin (not before)

---

## 8. Split-Brain Monitoring

The split-brain counter tracks: *enforcement blocked an entry that rehearsal would have allowed*.

| Metric | Field | Expected |
|--------|-------|----------|
| Total divergences | `split_brain_count` | 0 |
| Last divergence time | `last_split_brain_ts` | `""` (never) |
| Last divergence symbol | `last_split_brain_symbol` | `""` (never) |

**Why it matters:** Rehearsal (B.5) and enforcement (C.1) share the same permit index. If they disagree, something is wrong — timing drift, index corruption, or logic divergence.

**Investigation steps:**
1. Check `dle_entry_denials.jsonl` for the denied symbol + timestamp
2. Check `dle_rehearsal.jsonl` for the same symbol around the same time
3. Compare permit_id, direction, expiry
4. Root-cause the divergence before re-enabling enforcement

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `logs/state/phase_c_readiness.json` | Live readiness + enforcement metrics |
| `logs/execution/dle_entry_denials.jsonl` | Append-only denial log |
| `logs/execution/dle_rehearsal.jsonl` | B.5 rehearsal events |
| `logs/execution/dle_shadow_events.jsonl` | Shadow layer events |
| `ops/phase_c_window.log` | Ops ledger (manual daily) |
| `ops/PHASE_C_DAILY_CHECKPOINT.md` | Daily SOP |
| `ops/C1_OPS_PROTOCOL.md` | This document |
