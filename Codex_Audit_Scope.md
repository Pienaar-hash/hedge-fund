# GPT Hedge — Codex Audit Scope (v5.9 Prep)
**Purpose:** Guide Codex through a safe repo hygiene sweep — identifying legacy, redundant, and obsolete modules while keeping all active runtime and ML infrastructure intact.  
**Phase:** Sprint 5.9 — Infrastructure & Portfolio Analytics Hardening  
**Date:** 2025-11-08  

---

## 🧭 1. Active Modules — **KEEP**
These are in production or tied to current sprints and must remain untouched.

### Core Runtime
| Path | Purpose |
|------|----------|
| `execution/executor_live.py` | Main trading runtime (live & testnet) |
| `execution/sync_state.py` | Firestore + local state sync loop |
| `execution/utils.py` | Canonical utility layer (fetch, rounding, PnL, ZAR rate) |
| `execution/firestore_utils.py` | Telemetry + NAV + heartbeat publisher |
| `execution/risk_limits.py` | Risk caps, vetoes, and drawdown guards |
| `execution/order_router.py` | Unified order entry with retries and ID normalization |
| `execution/rules_sl_tp.py` | TP/SL + trailing logic |
| `execution/signal_generator.py` | Strategy signal core (momentum/vol/relative value) |
| `execution/telegram_utils.py` | Alerts + heartbeat notifications |
| `execution/leaderboard_sync.py` | (Active if used by dashboard — keep for now) |

### ML Subsystem — **KEEP**
| Path | Purpose |
|------|----------|
| `execution/ml/data.py` | Feature engineering + fetch pipeline |
| `execution/ml/features.py` | Derived indicators |
| `execution/ml/train.py` | Nightly model retrain cron |
| `execution/ml/predict.py` | Runtime inference |
| `cron/ml_retrain` | Linked cron / supervisor job |
| `execution/orderbook_features.py` | Future ML integration — keep |

### Dashboard & Telemetry
| Path | Purpose |
|------|----------|
| `dashboard/app.py` | Main Streamlit dashboard |
| `dashboard/live_helpers.py` | Position + Firestore adapters |
| `dashboard/nav_helpers.py` | NAV and reserve computation |
| `dashboard/dashboard_utils.py` | Shared Firestore readers and telemetry aggregation |
| `dashboard/router_health.py` | Deduped router telemetry (v2) |
| `docs/` | Sprint docs, audits, plans, changelogs |
| `scripts/doctor.py` | CLI diagnostics + dashboard subprocess |
| `utils/` | Shared metric, treasury, and test helpers |
| `tests/` | All test suites active in v5.8 |

---

## 🧹 2. Candidate for Removal or Archive
Codex should identify and archive these under `/archive/deprecated_v5.7/`.

| Path | Reason |
|------|---------|
| `execution/flatten_all.py` | Legacy batch flattener — replaced by `utils.py` |
| `execution/universe_resolver.py` | Outdated asset list generator |
| `execution/pipeline_probe.py` | Diagnostic prototype — never invoked |
| `execution/portfolio_probe.py` | Obsolete portfolio sampler — replaced by `doctor.py` |
| `execution/leaderboard_sync.py` | Only keep if referenced in dashboard; else archive |
| `execution/hedge_sync.py` | Old sync daemon superseded by `sync_state.py` |
| `execution/nav.py` | Superseded by unified equity logic |
| `execution/state_publish.py` | Duplicates Firestore publisher logic |
| `execution/signal_doctor.py` | Legacy diagnostic replaced by `scripts/doctor.py` |
| `execution/risk_autotune.py` | Deprecated prototype; audit and possibly archive |
| `execution/telegram_report.py` | Legacy Telegram formatters — replaced by `telegram_utils.py` |
| `execution/leaderboard_sync.py` | Archive if not used in app |
| `execution/ml/__old__/` | Any old prototypes under ML |
| `dashboard/router_health_v1.py` | Replaced by v2 |
| `dashboard/legacy_*/` | Any `legacy_` UI scripts |
| `old_`, `_bak`, `_copy`, `_tmp` | Temporary or backup artifacts |

---

## ⚙️ 3. Retention Rules
1. **Keep ML fully intact** — data, features, train, predict, and any ML cron.
2. **Keep all Firestore integrations** across `execution/firestore_utils.py`, `scripts/doctor.py`, and `dashboard/*`.
3. **Preserve current equity computation logic** per `infra_v5.7_audit.md` and `sprint_5_8_plan.md`.
4. **Retain test coverage** for dashboard, utils, and execution pipelines.
5. **Do not alter `.env`, supervisor.conf`, or `requirements.txt`.**
6. **Move removed code to `/archive/deprecated_v5.7/`**, not delete permanently.

---

## 📦 4. Expected Output from Codex
After the hygiene run, Codex must output:

Repo Hygiene Summary

Archived files: [list]

Remaining structure tree

Broken imports (if any)

Suggested import remaps


and commit:


git add .
git commit -m "[v5.9-prep] repo hygiene sweep"


---

## ✅ 5. Verification
After Codex completes:
- `pytest` passes on all tests.
- `python3 -m scripts.doctor -v` runs without missing imports.
- `sudo supervisorctl restart hedge:*` completes cleanly with no module errors.

---

**End of Codex Audit Scope.**
