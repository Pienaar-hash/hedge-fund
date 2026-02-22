# Calibration Window — Activation & Operations Runbook

**Version:** v7.9-CW  
**Module:** `execution/calibration_window.py`  
**Config:** `config/runtime.yaml` → `calibration_window` section  

---

## Overview

A 30-episode bounded calibration window for futures expectancy (E) calibration.
Reduced sizing (0.5% NAV/trade), hard episode cap, automatic KILL_SWITCH on
cap or drawdown breach. No parameter changes mid-window.

### Constraints

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Episode cap | 30 | MIN_EXPECTANCY_TRADES threshold for E maturity |
| Per-trade NAV | 0.5% | ~$48 USDT/trade at ~$9,666 NAV |
| Max concurrent | 2 | Existing risk_limits.json |
| Max leverage | 4x | Existing risk_limits.json |
| DD kill | 5% NAV | Auto-halt on cumulative DD ≥ 5% |
| Weight lock | No parameter changes mid-window |

### Risk Budget

- Worst case (30 consecutive max-loss trades): 30 × $48 × 4x leverage = $5,760
- But DD kill at 5% ($483 at current NAV) halts well before that
- Realistic max loss: $483 (DD kill) or less

---

## Pre-Activation Checklist

- [ ] CI suite: `PYTHONPATH=. pytest -q` — all green, 0 failures
- [ ] 9/9 Binary Lab gate: `PYTHONPATH=. python scripts/binary_lab_gate_cron.py` → GO
- [ ] NAV verified: `cat logs/state/nav_state.json | jq '.nav_usd'`
- [ ] Risk limits confirmed: `cat config/risk_limits.json | jq '.global'`
- [ ] No open positions: `cat logs/state/positions_state.json`
- [ ] Episode ledger current: `cat logs/state/episode_ledger.json | jq '.episodes | length'`
- [ ] Telegram alerts working: confirm via recent supervisor logs

---

## Activation Steps

### 1. Configure runtime.yaml

```yaml
calibration_window:
  enabled: true
  episode_cap: 30
  start_ts: "2026-XX-XXTXX:XX:XXZ"  # ← Set to current UTC time
  drawdown_kill_pct: 0.05
  per_trade_nav_pct: 0.005
```

Generate `start_ts`:
```bash
python3 -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))"
```

### 2. Enable Live Trading (dual-key)

Add `DRY_RUN="0"` and `CALIBRATION_WINDOW_ACK="1"` to supervisor env:

```bash
# Edit ops/hedge.conf — add to [program:hedge-executor] environment
# environment=BINANCE_TESTNET="1",DRY_RUN="0",CALIBRATION_WINDOW_ACK="1"
sudo nano /etc/supervisor/conf.d/hedge.conf
```

**Both keys required:** `enabled: true` in YAML *and* `CALIBRATION_WINDOW_ACK=1` in env.
Config alone cannot activate — prevents accidental activation via config drift.

### 3. Restart Executor

```bash
sudo supervisorctl restart hedge:hedge-executor
```

### 4. Verify Activation

```bash
# Boot line confirms ACTIVE + sizing cap in USD
tail -50 /var/log/hedge-executor.out.log | grep -i 'calibration_window.*BOOT'

# Expect: [calibration_window] BOOT: ACTIVE — cap=30, start=..., sizing=0.0050 ($48.33), ack=1

# Confirm DRY_RUN=0 and ACK=1
cat /proc/$(pgrep -f executor_live)/environ | tr '\0' '\n' | grep -E 'DRY_RUN|CALIBRATION_WINDOW_ACK'
```

---

## Mid-Window Monitoring

### Episode Progress

```bash
# Count episodes since start_ts
python3 -c "
import json
ledger = json.load(open('logs/state/episode_ledger.json'))
start = 'START_TS_HERE'  # replace
eps = [e for e in ledger.get('episodes', []) if e.get('exit_ts', '') >= start]
print(f'Episodes: {len(eps)}/30')
for e in eps[-5:]:
    print(f'  {e[\"symbol\"]} exit={e[\"exit_ts\"]} pnl={e.get(\"realized_pnl_usdt\", 0):.4f}')
"
```

### Drawdown Check

```bash
cat logs/state/nav_state.json | jq '{nav_usd, nav_age_s}'
cat logs/state/risk_snapshot.json | jq '.drawdown'
```

### Doctrine Events

```bash
tail -10 logs/doctrine_events.jsonl | jq '{event, action, ts}'
```

### Risk Vetoes

```bash
tail -20 logs/execution/risk_vetoes.jsonl | jq '{symbol, veto_reason, ts}'
```

---

## Emergency Halt

### Option A: Manual KILL_SWITCH

```bash
# Set env var — takes effect next executor loop iteration
sudo supervisorctl signal USR1 hedge:hedge-executor  # or:
# Add KILL_SWITCH="1" to supervisor env and restart
```

### Option B: Remove ACK (dual-key break)

```bash
# Remove CALIBRATION_WINDOW_ACK from supervisor env and restart
# Window becomes inactive immediately — config stays enabled but inert
sudo supervisorctl restart hedge:hedge-executor
```

### Option C: Disable Calibration Window

```bash
# Edit config/runtime.yaml: set enabled: false
# Restart executor
sudo supervisorctl restart hedge:hedge-executor
```

### Option C: Re-enable DRY_RUN

```bash
# Remove DRY_RUN="0" from supervisor env (defaults to "1")
sudo supervisorctl restart hedge:hedge-executor
```

---

## Post-Window Evaluation

After the window completes (30 episodes or DD kill), KILL_SWITCH fires automatically.

### Run Evaluation

```bash
PYTHONPATH=. python scripts/calibration_eval.py
```

Machine-readable:
```bash
PYTHONPATH=. python scripts/calibration_eval.py --json > /tmp/calibration_eval.json
```

### Evaluation Gates

| Gate | Threshold | Description |
|------|-----------|-------------|
| min_episodes | ≥ 20 | Enough data for meaningful E |
| symbol_maturity | ≥ 1 symbol at 30+ episodes | At least one symbol has mature E |
| positive_expectancy | E > 0 | Aggregate expectancy profitable |
| loss_streak | ≤ 10 consecutive | No extreme losing streak |
| differentiation | CV > 0.3 or range > 0.5 | E differentiates across symbols |

### Re-run Hybrid Audit

```bash
PYTHONPATH=. python scripts/hybrid_variance_audit.py
PYTHONPATH=. python scripts/binary_lab_gate_cron.py
```

### Decision Matrix

| Verdict | Action |
|---------|--------|
| **5/5 GO** | Safe to increase sizing toward production levels |
| **4/5 GO** | Extend window by 15-30 episodes, review failing gate |
| **≤3/5 GO** | Reset parameters, investigate strategy/regime alignment |

---

## Deactivation

After evaluation, disable the calibration window:

```yaml
calibration_window:
  enabled: false
```

If transitioning to production:
- Increase `per_trade_nav_pct` incrementally (0.005 → 0.01 → 0.015)
- Remove episode cap or increase to 100+
- Keep DD kill active (adjust threshold as needed)

---

## File Inventory

| File | Purpose |
|------|---------|
| `execution/calibration_window.py` | Episode-capped trading gate + DD kill |
| `scripts/calibration_eval.py` | Post-window evaluation & GO/NO-GO |
| `config/runtime.yaml` | `calibration_window` section |
| `tests/unit/test_calibration_window.py` | 18 tests (config, episodes, DD, sizing) |

---

## Invariants

1. **Calibration window is NOT a strategy change** — it calibrates E, not parameters
2. **No mid-window parameter changes** — frozen throughout
3. **KILL_SWITCH is permanent until manually cleared** — requires config change + restart
4. **Sizing cap is enforced in executor entry path** — screener/intent cannot exceed it
5. **Drawdown kill overrides episode cap** — safety takes priority
6. **Post-window evaluation is mandatory** — never skip to production
