# GPT Hedge Sprint 5.8 — Dashboard & Portfolio Equity Analytics

**Sprint Window:** Nov 6 → Nov 13 2025  
**Focus:** Portfolio Equity Analytics + Realtime Alerts  
**Environment:** prod | canonical treasury + Firestore mirrors validated Nov 6 @ 16:07 SAST  

---

## 🎯 Objectives

1. **Unify Equity Computation**
   - Dashboard, Doctor, and Firestore compute identical NAV + Reserves = Total Equity.
   - Remove cache drift between CLI ↔ UI.

2. **Modernize Dashboard UX**
   - Convert “FRESH”, “ZAR”, and “Telemetry” badges to hover tooltips.
   - Simplify metric rows for clean presentation.

3. **Reserves Analytics**
   - Add cost basis + PNL % per treasury asset.
   - Persist `avg_entry_price` via `data/treasury_txn.jsonl`.

4. **Execution & Telemetry Polish**
   - Deduplicate router table.
   - Add uptime + avg_confidence.
   - Merge Positions tab → Execution.

5. **ML & Doctor Reactivation**
   - Reinstate 24 h ML retrain cron.
   - Doctor tab subprocess robust + streamed output.

---

## 📋 Task Matrix

| Category | Goal | Files | Status |
|-----------|------|-------|--------|
| **Equity Consistency** | Align NAV/Reserves/Total Equity | `scripts/doctor.py`, `dashboard/app.py`, `dashboard/dashboard_utils.py` | ⬜ |
| **ZAR Tooltips** | Convert conversions → hover | `dashboard/app.py` | ⬜ |
| **Positions Mirror** | Fix 0-position bug | `execution/firestore_utils.py`, `dashboard/live_helpers.py` | ⬜ |
| **Treasury PNL** | Add avg_entry_price + Δ% | `execution/utils.py`, `dashboard/app.py` | ⬜ |
| **Router Dedup** | Drop duplicate client_order_id | `dashboard/router_health.py` | ⬜ |
| **Telemetry UX** | Tooltip statuses (FRESH, STALE, ZAR) | `dashboard/app.py` | ⬜ |
| **Merge Positions** | Consolidate tab | `dashboard/app.py` | ⬜ |
| **Remove Leaderboard** | Delete tab + sync logic | `dashboard/app.py`, `execution/leaderboard_sync.py` | ⬜ |
| **ML Retrain Cron** | Ensure daily run | `execution/ml/train.py`, `supervisor.conf`, `cron/` | ⬜ |
| **Doctor Subprocess** | Trap exit 1 + stream logs | `dashboard/app.py` | ⬜ |

---

## ⚙️ Engineering Notes

### 1️⃣ Canonical Equity Computation
```python
total_equity_usd = nav_trading_usd + reserves_usd_val
zar_rate = cached_usd_to_zar()
