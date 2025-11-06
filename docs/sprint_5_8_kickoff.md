# 🧭 GPT Hedge — Sprint 5.8 Kickoff  
**Theme:** Dashboard & Portfolio Equity Analytics  
**Window:** Nov 6 → Nov 13 2025  
**Environment:** `prod` (Validated Nov 6 @ 16:07 SAST)  

---

## 🎯 Sprint Objectives

| Tier | Focus | Outcome |
|------|--------|----------|
| **1 (Core)** | **Equity Consistency & Telemetry Stability** | Dashboard, Doctor and Firestore produce identical NAV + Reserves = Total Equity. No cache drift, no crashes on empty Firestore. |
| **2 (UX)** | **Modernize Dashboard Interface** | Replace status labels (FRESH/STALE/ZAR) with hover tooltips and merge Positions → Execution tab. |
| **3 (Analytics)** | **Treasury PNL Metrics** | Compute cost basis and Δ% PNL per treasury asset from `data/treasury_txn.jsonl`. |
| **4 (Ops)** | **Telemetry Schema Alignment + ML Cron** | Align Firestore paths (`telemetry/health`, `state/positions`); reinstate daily ML retrain cron. |

---

## 🧱 Core Engineering Tasks

### 1️⃣ `dashboard/app.py`
- Seed `nav_value = None` before use to prevent UnboundLocalError.  
- Move USD→ZAR conversion after `zar_rate` loads.  
- Add hover tooltips for FRESH/STALE/ZAR badges.  
- Merge Positions tab into Execution; remove Leaderboard.  
- Maintain 3-column metric layout.  
✅ Lint + Type checks: `ruff`, `mypy`, `pytest tests/test_dashboard_equity.py -v`

---

### 2️⃣ `execution/firestore_utils.py`
- Retarget `publish_heartbeat()` → `telemetry/health`.  
- Mirror `publish_positions()` to `state/positions {items:[...]}`.  
- Add `publish_health_if_needed()` for transition compatibility.  
✅ Validation: `pytest tests/test_firestore_publish.py -v`

---

### 3️⃣ `dashboard/dashboard_utils.py`
- Extend heartbeat reader to prioritise `telemetry/health`, fallback `telemetry/heartbeats`.  
- Normalize schema → `{timestamp, uptime, avg_confidence}`.  
✅ Validation: `pytest tests/test_dashboard_telemetry.py -v`

---

### 4️⃣ `scripts/doctor.py`
- Instantiate Binance client in `_collect_positions()`.  
- Load reserves from `logs/treasury.json` / Firestore snapshot.  
- Detect cached USD→ZAR rates and label STALE when age > 6 h.  
- Stream Doctor subprocess logs to UI; trap exit 1 gracefully.  
✅ Validation: `pytest tests/test_doctor_positions.py -v`

---

### 5️⃣ `execution/utils.py`
- Add `get_treasury_snapshot()` to load treasury dict.  
- Extend `get_usd_to_zar()` to return meta (source + age).  
- Implement `compute_treasury_pnl()` for asset-level PNL %.  
✅ Validation: `pytest tests/test_utils_treasury.py -v`

---

### 6️⃣ `execution/ml/train.py` + `supervisor.conf` + `cron/`
- Schedule daily retrain (03:15 SAST):  
  ```bash
  15 03 * * * python -m execution.ml.train --daily
