# Binary Lab — Daily Checkpoint SOP

**Duration:** ~15 seconds  
**Frequency:** Once daily (UTC morning, after core checkpoint)  
**Owner:** Operator  
**Experiment Window:** Day 0 → Day 30  
**Sleeve ID:** binary_lab_s1

---

## Prerequisites

This checkpoint only activates after `binary_lab_state.json` shows `status: "ACTIVE"`.  
Before deployment, this file exists in formalized state only.

---

## Checklist

| # | Check | Command / Source | Pass Criteria |
|---|-------|------------------|---------------|
| 1 | Sleeve NAV | `jq '.capital.current_nav_usd' logs/state/binary_lab_state.json` | > $1,700 (kill line) |
| 2 | Kill distance | `jq '.kill_line.distance_usd' logs/state/binary_lab_state.json` | > 0 |
| 3 | Open positions | `jq '.positions.open_count' logs/state/binary_lab_state.json` | ≤ 3 |
| 4 | Trade log growing | `wc -l logs/execution/binary_lab_trades.jsonl` | Increasing (unless 0-trade day) |
| 5 | Config hash | `sha256sum config/binary_lab_limits.json` | Unchanged from Day 0 |
| 6 | Rule violations | `jq '.rule_violations' logs/state/binary_lab_state.json` | 0 |
| 7 | Freeze intact | `jq '.freeze_intact' logs/state/binary_lab_state.json` | true |
| 8 | Core NAV unblended | Compare `nav_state.json` and `binary_lab_state.json` | Separate, no cross-contamination |

---

## Quick-Check One-Liner

```bash
jq '{
  day: .day,
  of: .day_total,
  status: .status,
  nav: .capital.current_nav_usd,
  pnl: .capital.pnl_usd,
  kill_dist: .kill_line.distance_usd,
  kill_breached: .kill_line.breached,
  trades: .metrics.total_trades,
  wins: .metrics.wins,
  losses: .metrics.losses,
  wr: .metrics.win_rate,
  violations: .rule_violations,
  freeze: .freeze_intact
}' logs/state/binary_lab_state.json
```

---

## Ops Ledger Entry Format

Append to `ops/binary_lab_window.log`:

```
2026-XX-XX | Day 1/30 | trades=3 | wins=2 | losses=1 | pnl=+$14.00 | sleeve=$2014 | kill_dist=$314 | violations=0 | CLEAN
```

Kill line hit:
```
2026-XX-XX | Day X/30 | KILL LINE HIT | sleeve=$1698 | EXPERIMENT TERMINATED IMMEDIATELY
```

Rule violation:
```
2026-XX-XX | Day X/30 | RULE VIOLATION | type=size_escalation | FREEZE BROKEN — EXPERIMENT INVALIDATED
```

---

## Red-Line Alerts (Immediate Action)

| Condition | Severity | Action |
|-----------|----------|--------|
| `kill_line.breached == true` | **CRITICAL** | Close all positions. End experiment. No reconsideration. |
| `rule_violations > 0` | **CRITICAL** | Experiment invalidated. Document and close. |
| `freeze_intact == false` | **CRITICAL** | Parameter changed mid-freeze. Experiment invalidated. |
| `open_count > 3` | **HIGH** | Concurrency cap breached. Close excess. |
| Core NAV contaminated | **HIGH** | Binary PnL leaked into core reporting. Investigate. |
| 0 trades for 5+ consecutive days | **MEDIUM** | Regime blocking all entries. Expected in CHOPPY — note but no action. |

---

## Mid-Point Review (Day 15)

At Day 15, produce an interim snapshot (observation only, no changes):

```bash
# Band separation check
cat logs/execution/binary_lab_trades.jsonl | \
  jq -r '[.conviction_band, .pnl_usd] | @tsv' | \
  awk '{sum[$1]+=$2; count[$1]++} END {for (b in sum) printf "%s: ev=$%.2f n=%d\n", b, sum[b]/count[b], count[b]}'

# Regime distribution of trades
cat logs/execution/binary_lab_trades.jsonl | \
  jq -r '.regime' | sort | uniq -c | sort -rn

# Cumulative PnL
cat logs/execution/binary_lab_trades.jsonl | \
  jq -r '.pnl_usd' | awk '{s+=$1; print NR, s}'
```

This is observation. Not a decision point. No changes allowed.

---

*Satellite Lab: binary_lab_s1*  
*Parent: Phase C Contraction Window (v7.9-S1)*
