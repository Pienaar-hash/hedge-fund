# Phase C — Daily Checkpoint SOP

**Duration:** ~30 seconds  
**Frequency:** Once daily (UTC morning recommended)  
**Owner:** Operator or automated cron  

---

## Checklist

| # | Check | Command / Source | Pass Criteria |
|---|-------|------------------|---------------|
| 1 | Manifest integrity | `cat logs/state/diagnostics.json \| jq '.manifest_ok'` | `true` |
| 2 | Exit reason map | `cat logs/state/diagnostics.json \| jq '.exit_reason_map_ok'` | `true` |
| 3 | DLE shadow path | `ls -la logs/execution/dle_shadow_events.jsonl` | File exists, growing |
| 4 | Rehearsal metrics | `cat logs/state/phase_c_readiness.json \| jq '.'` | See gate criteria below |
| 5 | Ops ledger entry | Append to `ops/phase_c_window.log` | One line per day |

---

## Gate Criteria (from manifest)

All must be **simultaneously true** for 14 consecutive days:

| Metric | Threshold | Source field |
|--------|-----------|-------------|
| `would_block_pct` | < 0.1% | `.current_metrics.would_block_pct` |
| `expired_permit_count` | == 0 | `.current_metrics.expired_permit_count` |
| `missing_permit_count` | == 0 | `.current_metrics.missing_permit_count` |
| `rehearsal_enabled` | `true` | `.rehearsal_enabled` |
| `total_orders` | > 0 | `.current_metrics.total_orders` |

---

## Quick-Check One-Liner

```bash
jq '{
  day: .window_days_met,
  of: .window_days_required,
  criteria_met: .criteria_met,
  gate: .gate_satisfied,
  block_pct: .current_metrics.would_block_pct,
  expired: .current_metrics.expired_permit_count,
  missing: .current_metrics.missing_permit_count
}' logs/state/phase_c_readiness.json
```

Expected output when healthy:

```json
{
  "day": 7,
  "of": 14,
  "criteria_met": true,
  "gate": false,
  "block_pct": 0,
  "expired": 0,
  "missing": 0
}
```

---

## Ops Ledger Entry Format

Append one line per day to `ops/phase_c_window.log`:

```
2026-02-13 | Day 1/14 | criteria_met=true | would_block_pct=0.00 | expired=0 | missing=0 | CLEAN
2026-02-14 | Day 2/14 | criteria_met=true | would_block_pct=0.00 | expired=0 | missing=0 | CLEAN
```

If a breach occurs:

```
2026-02-15 | Day 0/14 | criteria_met=false | would_block_pct=0.15 | expired=0 | missing=0 | BREACH: would_block_pct=0.15% >= 0.1% — window reset
```

---

## When Gate Satisfied

When `gate_satisfied == true` (14 consecutive clean days):

1. `phase_c_readiness.json` will show `"gate_satisfied": true`
2. Phase C becomes a **governance decision**, not a technical one
3. Recommended Phase C.1: **Entry-only enforcement** (block only ENTRY orders, never EXIT)
4. No code changes until explicit Phase C authorization
