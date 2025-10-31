## 🧩 `docs/infra_v5.5_patch.md`

### **v5.5 — Router Health + Risk Autotune + Dashboard Phase II + ML Reactivation**

**Commit:** `v5.5 Router Health + Risk Autotune + Dashboard Phase II + ML Reactivation`
**Author:** Codex (`codex@example.com`)
**Date:** 2024-03-14 12:00 UTC

---

### **Executive Summary**

* **Runtime Risk Persistence:**
  Introduced persistent `RiskState` snapshots and adaptive limits through the new `RiskAutotuner` engine (`execution/risk_autotune.py`), now integrated with executor heartbeats (`executor_live.py:123, 1652`).

* **Router Health Analytics:**
  Added a dedicated analytics module (`dashboard/router_health.py`) and integrated cached PnL / hit-rate curves into Streamlit (`app.py:590, 1263, 1528`).
  Provides per-signal / per-symbol performance and rolling NAV + drawdown overlays.

* **ML Reactivation:**
  Revived lightweight live feature extraction (`execution/ml/features_live.py`) and asynchronous scoring (`execution/ml/predict.py`, `signal_generator.py:28`).
  Added **ML Insights** dashboard tab with cached predictions.

* **Infra Hardening:**
  Unified Firestore handling (`utils/firestore_client.py:259`) with safe fallbacks and `ENV=dev` defaults.
  Guarded all CLI/cache writers (`scripts/cron_doctor_cache.py`, `scripts/fs_doctor.py`, `execution/state_publish.py`) against prod writes unless explicitly allowed.
  Added exponential back-off and jitter to Binance HTTP requests (`exchange_utils.py:129`).

* **Tests & Lint:**
  Regression coverage in `tests/test_infra_v5_5.py`.
  `pytest` ✅ pass, `ruff` ✅ clean.
  `mypy --strict` still reports legacy issues under `scripts/` and `dashboard/` but no new regressions in modified modules.

---

### **Key Modules Added / Updated**

| Module                                                | Purpose                                                    |
| ----------------------------------------------------- | ---------------------------------------------------------- |
| **`execution/risk_autotune.py`**                      | Adaptive risk thresholds + state persistence               |
| **`execution/risk_limits.py` (404+)**                 | Snapshots and restoration of `RiskState`                   |
| **`execution/executor_live.py`**                      | Autotuner integration + ENV safeguards + heartbeat metrics |
| **`dashboard/router_health.py`**                      | PnL / hit-rate aggregation and per-symbol metrics          |
| **`dashboard/app.py`**                                | Cached doctor snapshot, new tabs, drawdown overlay         |
| **`execution/ml/features_live.py` & `ml/predict.py`** | Live ML features + async scoring                           |
| **`utils/firestore_client.py`**                       | Centralized client + ENV guardrails                        |
| **`scripts/cron_doctor_cache.py`**                    | Periodic doctor snapshot caching                           |
| **`execution/exchange_utils.py`**                     | Resilient HTTP retry/back-off wrapper                      |

---

### **Validation Checklist**

| Check                                       | Result                               |
| ------------------------------------------- | ------------------------------------ |
| `pytest -q tests/test_infra_v5_5.py`        | ✅ Pass                               |
| `ruff check --fix execution/ dashboard/`    | ✅ Clean                              |
| `mypy --strict execution/... dashboard/...` | ⚠️ Legacy issues only                |
| Supervisor reload                           | ✅ Executor + Dashboard healthy       |
| Streamlit load time                         | ≤ 2 s (avg ≈ 1.3 s)                  |
| Router Health tab                           | ✅ Displays PnL + Hit-rate + Drawdown |
| Risk Autotune                               | ✅ Heartbeats log adaptive limits     |
| ML Insights tab                             | ✅ Populated from cache               |

---

### **Next Steps (v5.6 Preview)**

* Extend **RiskAutotuner** with rolling Sharpe / volatility normalization.
* Add **Doctor Confidence Telemetry** overlay to Router Health chart.
* Begin **v5.6 AI Research Track Phase II** — RL / Monte Carlo adaptive sizers.
* Migrate Streamlit caching → async background tasks (`asyncio` or `cron_worker`).