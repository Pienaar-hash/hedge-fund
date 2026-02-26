# Activation Window v8.0 — Full-Stack System Certification Protocol

**Version:** v8.0  
**Module:** `execution/activation_window.py`  
**Verify:** `scripts/activation_verify.py`  
**Config:** `config/runtime.yaml` → `activation_window` section  
**State:** `logs/state/activation_window_state.json`  
**Supersedes:** `CALIBRATION_WINDOW.md` (episode-based, futures-only)

---

## Design Philosophy

Shift from:

> "Calibrate expectancy over 30 episodes"

To:

> "Prove full-stack integrity over 14 calendar days before capital scale."

The activation window is **not a strategy test** — it is a **system
certification protocol**.  It covers the entire hedge stack:

- Futures engine
- DLE gating layer
- Binary Lab SHADOW sleeve
- Risk vetoes
- State machine transitions
- Telemetry integrity
- Manifest immutability

---

## Window Definition

### Duration

14 consecutive calendar days (clock-based, not episode-based).

- 15m rounds → 96 rounds/day → **1,344 rounds** over 14 days
- Sufficient directional sample across multiple regime transitions
- Detects slow structural drift that episodes cannot

### Scope

| Component | Coverage |
|-----------|----------|
| Executor core | KILL_SWITCH, sizing cap, episode counting |
| Risk limits | Drawdown monitoring, veto counting |
| DLE veto layer | Shadow mismatch detection |
| Binary Lab | SHADOW trades, freeze integrity |
| Episode ledger | Reconciliation |
| NAV accounting | Exchange match, stale detection |
| Manifest surfaces | Hash immutability check |
| Config | Hash drift detection |

### Constraints

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Duration | 14 days | sufficient regime transitions |
| Per-trade NAV | 0.5% | ~$48 USDT/trade at ~$9,666 NAV |
| Max concurrent | 2 | Existing risk_limits.json |
| Max leverage | 4x | Existing risk_limits.json |
| DD kill | 5% NAV | Auto-halt on cumulative DD ≥ 5% |
| Manifest lock | SHA-256 hash at boot, checked every loop |
| Config lock | SHA-256 hash at boot, checked every loop |

---

## Pre-Activation Checklist

### CI & Determinism

- [ ] `PYTHONPATH=. pytest -q` — all green, 0 failures
- [ ] `v7_manifest.json` unchanged since last production tag
- [ ] No pending migrations
- [ ] Dataset admission states unchanged

### State Integrity

- [ ] NAV matches exchange: `cat logs/state/nav_state.json | jq '.nav_usd'`
- [ ] No orphan positions: `cat logs/state/positions_state.json`
- [ ] Episode ledger reconciles: `cat logs/state/episode_ledger.json | jq '.episodes | length'`
- [ ] No stale telemetry files (all `updated_ts` < 90s)
- [ ] Risk snapshot fresh: `cat logs/state/risk_snapshot.json | jq '.ts'`

### Binary Lab

- [ ] Mode = SHADOW: `echo $BINARY_LAB_MODE`
- [ ] Freeze hash recorded in `binary_lab_state.json`
- [ ] 15m horizon lock confirmed
- [ ] No bundle code paths active

### DLE Layer

- [ ] DLE shadow active: `cat logs/state/dle_shadow_state.json`
- [ ] No enforcement flip mid-window
- [ ] No veto anomalies: `tail -50 logs/execution/dle_shadow_events.jsonl | jq '.event'`

### Execution Layer

- [ ] `DRY_RUN=0` only if futures live
- [ ] Binary Lab never touches capital
- [ ] Supervisor env variables logged
- [ ] Telegram alerts working

---

## Activation Steps

### 1. Configure runtime.yaml

```yaml
activation_window:
  enabled: true
  duration_days: 14
  start_ts: "2026-XX-XXTXX:XX:XXZ"  # ← Set to current UTC time
  drawdown_kill_pct: 0.05
  per_trade_nav_pct: 0.005
```

Generate `start_ts`:
```bash
python3 -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))"
```

### 2. Enable Dual-Key

Add `ACTIVATION_WINDOW_ACK="1"` to supervisor env:

```bash
sudo nano /etc/supervisor/conf.d/hedge.conf
# Add to [program:hedge-executor] environment:
# ACTIVATION_WINDOW_ACK="1"
```

**Both keys required:** `enabled: true` in YAML *and* `ACTIVATION_WINDOW_ACK=1` in env.
Config alone cannot activate — prevents accidental activation via config drift.

### 3. Restart Executor

```bash
sudo supervisorctl restart hedge:hedge-executor
```

### 4. Verify Activation

```bash
# Boot line confirms ACTIVE + hash snapshots
tail -50 /var/log/hedge-executor.out.log | grep -i 'activation_window.*BOOT'

# Expect: [activation_window] BOOT: ACTIVE — 14d window, start=..., manifest=..., ack=1

# Confirm state file is being written
cat logs/state/activation_window_state.json | jq '{active, elapsed_days, remaining_days, halted}'

# Confirm env vars
cat /proc/$(pgrep -f executor_live)/environ | tr '\0' '\n' | grep -E 'ACTIVATION_WINDOW_ACK'
```

---

## 14-Day Monitoring Discipline

### Daily Checks

| Surface | Check | Command |
|---------|-------|---------|
| NAV | Within expected variance | `cat logs/state/nav_state.json \| jq '.nav_usd'` |
| Drawdown | < 5% | `cat logs/state/activation_window_state.json \| jq '.drawdown_pct'` |
| Risk vetoes | No unexplained spikes | `cat logs/state/activation_window_state.json \| jq '.risk_veto_count'` |
| Doctrine events | No unexpected transitions | `tail -10 logs/doctrine_events.jsonl \| jq '{event, action}'` |
| Binary Lab SHADOW | Trades recorded | `wc -l logs/execution/binary_lab_trades.jsonl` |
| Manifest drift | None | `cat logs/state/activation_window_state.json \| jq '.manifest_intact'` |
| DLE mismatches | 0 or minimal | `cat logs/state/activation_window_state.json \| jq '.dle_mismatches'` |
| State machine | No violations | `cat logs/state/activation_window_state.json \| jq '.binary_lab_freeze_ok'` |

### Quick Status

```bash
cat logs/state/activation_window_state.json | jq '{
  active, elapsed_days, remaining_days, halted, halt_reason,
  drawdown_pct, episodes_completed, risk_veto_count,
  manifest_intact, config_intact, binary_lab_freeze_ok
}'
```

---

## Kill Conditions (Stack-Wide)

Immediate KILL_SWITCH activation if any of:

| Condition | Type |
|-----------|------|
| NAV drawdown ≥ 5% | Risk |
| Manifest file hash changed | Structural |
| Config hash drifted from boot | Structural |
| Binary Lab freeze violated | Structural |
| 14 days elapsed | Completion (not failure) |

Unlike the old calibration window, this isn't just a sizing kill —
it's **structural integrity enforcement**.

---

## Emergency Halt

### Option A: Manual KILL_SWITCH

```bash
# Add KILL_SWITCH="1" to supervisor env and restart
sudo supervisorctl restart hedge:hedge-executor
```

### Option B: Remove ACK (dual-key break)

```bash
# Remove ACTIVATION_WINDOW_ACK from supervisor env and restart
sudo supervisorctl restart hedge:hedge-executor
```

### Option C: Disable Window

```yaml
activation_window:
  enabled: false
```

---

## Day 14 — Formal Verification & Activation Run

### Run Verification

```bash
PYTHONPATH=. python scripts/activation_verify.py
```

Machine-readable:
```bash
PYTHONPATH=. python scripts/activation_verify.py --json > /tmp/activation_verdict.json
```

### 7-Gate Verification

| Gate | Criteria |
|------|----------|
| **nav_stable** | NAV > 0 |
| **drawdown_within_limits** | DD < 5% kill threshold |
| **risk_veto_consistent** | < 500 vetoes in 14 days |
| **binary_lab_shadow_valid** | Freeze intact, trades recorded |
| **manifest_intact** | v7_manifest.json unchanged since boot |
| **no_freeze_violations** | Config + Binary Lab hash stable |
| **dle_shadow_clean** | < 50 DLE mismatches |

### Decision Matrix

| Score | Verdict | Action |
|-------|---------|--------|
| **7/7 GO** | GO | Promote to Production |
| **6/7 GO** | EXTEND | Extend window 7 days, review failing gate |
| **≤5 GO** | NO-GO | Investigate, do not scale |

**No partial scaling. No subjective override.**

---

## Post-Window: Transition to Production

After 7/7 GO verdict:

1. Increase `per_trade_nav_pct` incrementally: 0.005 → 0.01 → 0.015
2. Remove activation window or start a new 14-day window at higher sizing
3. Keep DD kill active (adjust threshold as needed)
4. Keep manifest integrity checking active permanently

After 6/7 EXTEND:

1. Fix the failing gate
2. Start a new 7-day extension window
3. Re-run `activation_verify.py` at end

After ≤5 NO-GO:

1. Disable the activation window
2. Investigate structural failures
3. Do not deploy capital until issues resolved

---

## Deactivation

```yaml
activation_window:
  enabled: false
```

```bash
# Remove ACTIVATION_WINDOW_ACK from supervisor env
sudo supervisorctl restart hedge:hedge-executor
```

---

## File Inventory

| File | Purpose |
|------|---------|
| `execution/activation_window.py` | Full-stack check, hash tracking, KILL_SWITCH, state file |
| `scripts/activation_verify.py` | Day-14 7-gate verification script |
| `config/runtime.yaml` | `activation_window` section |
| `logs/state/activation_window_state.json` | Per-loop state for dashboard |
| `v7_manifest.json` | `activation_window_state` surface registered |
| `tests/unit/test_activation_window.py` | Unit tests |
| `docs/ACTIVATION_WINDOW_v8.md` | This document |

### Relationship to Calibration Window

The old `calibration_window` (v7.9-CW) remains in `runtime.yaml` and
`execution/calibration_window.py`.  Both can run concurrently — the
executor checks both.  The tighter sizing cap wins.

Eventually the calibration window will be retired.  The activation
window subsumes and extends it.

---

## Invariants

1. **Time-bounded, not episode-bounded** — 14 days tests the machine, not just trades
2. **No mid-window parameter changes** — config hash is locked at boot
3. **KILL_SWITCH is permanent until manually cleared** — requires config change + restart
4. **Structural kills override all** — manifest drift halts even if trading is profitable
5. **Post-window verification is mandatory** — never skip to production
6. **Dual-key activation** — prevents config drift from enabling the window
7. **State file emitted** — dashboard can observe window status in real-time
