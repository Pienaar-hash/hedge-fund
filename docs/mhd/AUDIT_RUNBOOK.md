# MHD Audit Runbook

> Commands and expected outputs for running MHD audits.

---

## Prerequisites

```bash
cd /root/hedge-fund
export PYTHONPATH=/root/hedge-fund
```

---

## Audit 0: Evidence Admission Gate

### Check DLE Events

```bash
# Count DLE events
wc -l logs/dle/dle_events_v1.jsonl

# Event type distribution
cat logs/dle/dle_events_v1.jsonl | jq -r '.event_type' | sort | uniq -c | sort -rn

# Expected output:
#    576 ENTRY_DENY
#    173 EXIT_ALLOW
#    126 ENTRY_ALLOW
```

### Check Episode Ledger Freshness

```bash
# Last rebuild timestamp
cat logs/state/episode_ledger.json | jq '.last_rebuild_ts'

# Episode count
cat logs/state/episode_ledger.json | jq '.episode_count'

# Expected: timestamp within 30 minutes, episode_count > 0
```

### Check Orders Coverage

```bash
# Orders attempted
wc -l logs/execution/orders_attempted.jsonl

# Orders executed  
wc -l logs/execution/orders_executed.jsonl

# Risk vetoes
wc -l logs/execution/risk_vetoes.jsonl
```

### Chain Linkage Check

```bash
# DLE chains with fills
cat logs/dle/dle_events_v1.jsonl | jq -r 'select(.event_type == "ENTRY_ALLOW") | .payload.chain_id' | wc -l

# Compare to actual fills in window
cat logs/execution/orders_executed.jsonl | tail -1000 | wc -l
```

---

## Audit 1A: Replay Determinism

### Sample 10 Episodes

```bash
# Get 10 random episode IDs
cat logs/state/episode_ledger.json | jq -r '.episodes[-10:][].episode_id'

# For each episode, check DLE context
EPISODE_ID="EP_0518"  # example
cat logs/dle/dle_events_v1.jsonl | jq "select(.payload.context.episode_id == \"$EPISODE_ID\")"
```

### Check Required Fields

```bash
# Check for sizing factors in DLE
cat logs/dle/dle_events_v1.jsonl | head -100 | jq 'select(.payload.context.sizing != null)' | wc -l

# Check for regime in context
cat logs/dle/dle_events_v1.jsonl | head -100 | jq 'select(.payload.context.regime != null)' | wc -l
```

---

## Audit 1B: Scoring Decomposability

### Check Symbol Scores

```bash
# Factor breakdown available
cat logs/state/symbol_scores_v6.json | jq 'keys'

# Per-symbol factors
cat logs/state/symbol_scores_v6.json | jq '.BTCUSDT'
```

### Check Expectancy

```bash
# Expectancy factors
cat logs/state/expectancy_v6.json | jq 'keys'
```

---

## Audit 2A: Intervention Frequency

### Supervisor Restarts

```bash
# Check uptime
sudo supervisorctl status hedge:

# Expected: uptime > 24h without restarts
```

### NAV Stale Events

```bash
# Current NAV age
cat logs/state/risk_snapshot.json | jq '.dd_state.drawdown.nav_cache_age'

# Count stale events in last 24h
grep "nav_stale" /var/log/hedge-executor.out.log | tail -100 | wc -l
```

### Error Rate

```bash
# Error count
grep -c "ERROR\|Exception" /var/log/hedge-executor.err.log

# Total log lines
wc -l /var/log/hedge-executor.out.log

# Calculate rate: errors / total < 1%
```

---

## Audit 2B: Freeze Integrity

### Config Changes

```bash
# Git changes in config/
git diff --stat HEAD~10 -- config/

# Expected: 0 changes during cycle
```

### Manifest Integrity

```bash
# Check VERSION
cat VERSION

# Check manifest
cat v7_manifest.json | jq '.version'
```

---

## Audit 3A: Degradation Capability

### Testnet Mode

```bash
# Verify testnet available
grep "BINANCE_TESTNET" /etc/supervisor/conf.d/hedge.conf

# Current mode
cat logs/state/risk_snapshot.json | jq '.risk_config_meta'
```

### Offline Episode Ledger

```bash
# Test rebuild (does not require live exchange)
PYTHONPATH=. ./venv/bin/python -m execution.episode_ledger

# Expected: rebuilds from local logs
```

---

## Audit 4A: Temporal Integrity

### Check for Inversions

```bash
# Extract timestamps and check ordering
cat logs/dle/dle_events_v1.jsonl | jq -c '{chain: .payload.chain_id, ts: .ts, type: .event_type}' | head -100

# Manual inspection: ts should increase within chain
```

### Automated Check (Python)

```python
import json
from pathlib import Path

events = []
for line in Path("logs/dle/dle_events_v1.jsonl").read_text().strip().split("\n"):
    events.append(json.loads(line))

# Group by chain
chains = {}
for e in events:
    cid = e.get("payload", {}).get("chain_id")
    if cid:
        chains.setdefault(cid, []).append(e)

# Check ordering
inversions = []
for cid, chain_events in chains.items():
    sorted_events = sorted(chain_events, key=lambda x: x["ts"])
    if sorted_events != chain_events:
        inversions.append(cid)

print(f"Inversions: {len(inversions)}")
```

---

## Audit 4B: Chain Completeness

### Count Orphans

```bash
# ENTRY_ALLOW without subsequent fill or veto
cat logs/dle/dle_events_v1.jsonl | jq -r 'select(.event_type == "ENTRY_ALLOW") | .payload.chain_id' | sort -u > /tmp/allows.txt

cat logs/execution/orders_executed.jsonl | jq -r '.chain_id // empty' | sort -u > /tmp/fills.txt

# Orphans = allows not in fills
comm -23 /tmp/allows.txt /tmp/fills.txt | wc -l
```

---

## Audit 4C: Deny Reason Closure

### Reason Histogram

```bash
# DLE deny reasons
cat logs/dle/dle_events_v1.jsonl | jq -r 'select(.event_type == "ENTRY_DENY") | .payload.deny_reason' | sort | uniq -c | sort -rn

# Risk veto reasons
cat logs/execution/risk_vetoes.jsonl | jq -r '.veto_reason' | sort | uniq -c | sort -rn
```

### Canonical Reasons

```
VETO_DIRECTION_MISMATCH
VETO_REGIME_UNSTABLE
VETO_REGIME_CONFIDENCE
SENTINEL_STALE
NAV_STALE
RISK_MODE_HALTED
min_notional
per_symbol_cap
portfolio_dd
correlation_cap
max_concurrent_positions
daily_loss_limit
```

### Check for Unmapped

```bash
# Find reasons not in canonical list
cat logs/dle/dle_events_v1.jsonl | jq -r 'select(.event_type == "ENTRY_DENY") | .payload.deny_reason' | sort -u | grep -v -E "VETO_|SENTINEL_|NAV_|RISK_|min_notional|per_symbol|portfolio_dd|correlation|max_concurrent|daily_loss"
```

---

## Generate Canonical Replay Pack

### Replay 1: ALLOW → Filled

```bash
# Find an ENTRY_ALLOW that resulted in fill
cat logs/dle/dle_events_v1.jsonl | jq 'select(.event_type == "ENTRY_ALLOW")' | head -1 > /tmp/replay_allow_filled.json
cat /tmp/replay_allow_filled.json | jq '.'
```

### Replay 2: ALLOW → Veto

```bash
# Find ENTRY_ALLOW followed by risk veto
# (requires chain_id correlation)
cat logs/dle/dle_events_v1.jsonl | jq 'select(.event_type == "ENTRY_ALLOW" and .payload.outcome == "VETOED")' | head -1
```

### Replay 3: DENY

```bash
cat logs/dle/dle_events_v1.jsonl | jq 'select(.event_type == "ENTRY_DENY")' | head -1 > /tmp/replay_deny.json
cat /tmp/replay_deny.json | jq '.'
```

---

## Quick Health Check (Daily)

```bash
#!/bin/bash
# Save as: scripts/mhd_daily_check.sh

echo "=== MHD Daily Health Check ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

echo "--- Evidence Gate ---"
echo "DLE events: $(wc -l < logs/dle/dle_events_v1.jsonl)"
echo "Episode ledger: $(cat logs/state/episode_ledger.json | jq -r '.last_rebuild_ts')"
echo ""

echo "--- Temporal Integrity ---"
echo "Checking for inversions... (manual review needed)"
echo ""

echo "--- Chain Completeness ---"
echo "ENTRY_DENY: $(cat logs/dle/dle_events_v1.jsonl | jq -r 'select(.event_type == "ENTRY_DENY")' | wc -l)"
echo "ENTRY_ALLOW: $(cat logs/dle/dle_events_v1.jsonl | jq -r 'select(.event_type == "ENTRY_ALLOW")' | wc -l)"
echo "EXIT_ALLOW: $(cat logs/dle/dle_events_v1.jsonl | jq -r 'select(.event_type == "EXIT_ALLOW")' | wc -l)"
echo ""

echo "--- Deny Reason Closure ---"
cat logs/dle/dle_events_v1.jsonl | jq -r 'select(.event_type == "ENTRY_DENY") | .payload.deny_reason' | sort | uniq -c | sort -rn | head -10
echo ""

echo "--- Ops Burden ---"
echo "Supervisor status:"
sudo supervisorctl status hedge: | grep -E "RUNNING|uptime"
echo ""

echo "=== Check Complete ==="
```

---

## Output Artifacts

After running audits, save:

```
docs/mhd/reports/
├── MHD-2026-01-29-001.md      # Daily report
├── MHD-2026-01-WEEKLY-01.md   # Weekly report
└── MHD-CYCLE_004-CLOSE.md     # Cycle close report
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial runbook |
