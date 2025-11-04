# 🧭 Quant Infrastructure Patch — v5.6  
**Commit:** `v5.6 ML Confidence + Sharpe Normalization + RL Sizer Pilot`  
**Author:** Codex (`codex@example.com`)  
**Date:** 2025-11-07 (target)  

---

## 🚀 Executive Summary
Sprint v5.6 extends the v5.x intelligence stack toward **adaptive and learning-based execution**.  
The patch introduces confidence-weighted analytics, Sharpe normalization for volatility-aware metrics, and a reinforcement learning research pilot for position sizing.

**Core Upgrades**
- **Router Health 2.0** — Doctor-weighted PnL and hit-rate overlays; rolling Sharpe visualizations.
- **Sharpe Normalization Engine** — Standardized Sharpe/Sortino across symbols and lookbacks.
- **ML Confidence Telemetry** — Live model confidence + feature attribution feed surfaced in dashboard.
- **RL Sizer Pilot** — Prototype reinforcement-learning environment for adaptive sizing research.
- **Cross-Module Feedback Loop** — RiskAutotuner now ingests normalized metrics and ML confidence data.

---

## 🧩 File-Level Additions & Modifications

| Module | Purpose / Key Changes |
|---------|-----------------------|
| **`dashboard/router_health.py`** | Adds doctor-confidence weighting, rolling Sharpe computation, and normalized per-symbol stats. |
| **`dashboard/app.py`** | Integrates new “ML Confidence” and “RL Pilot” tabs; overlays rolling Sharpe and hit-rate curves. |
| **`execution/metrics_normalizer.py`** | New module computing normalized Sharpe ratios, volatility scaling, and factor weighting. |
| **`execution/risk_autotune.py`** | Updated to consume normalized Sharpe and ML confidence for adaptive threshold scaling. |
| **`ml/telemetry.py`** | New component capturing model confidence, top features, and attribution importance → cached JSON. |
| **`research/rl_sizer/`** | RL training environment (`SizingEnv`), agent interface (PPO/DQN), and episodic logging to `/logs/research/rl_runs/`. |
| **`tests/test_metrics_normalizer.py`** | Unit coverage for Sharpe/volatility normalization. |
| **`tests/test_router_health_v2.py`** | Integration tests for doctor-weighted Router Health analytics. |
| **`tests/test_rl_sizer_env.py`** | Simulation validation of RL sizing environment. |

---

## 🧪 Validation Checklist

| Check | Result / Expectation |
|--------|----------------------|
| `pytest -q tests/test_metrics_normalizer.py` | ✅ Pass |
| `pytest -q tests/test_router_health_v2.py` | ✅ Pass |
| `pytest -q tests/test_rl_sizer_env.py` | ✅ Pass |
| `ruff check dashboard/ execution/ research/ ml/` | ✅ Clean |
| `mypy execution/metrics_normalizer.py execution/risk_autotune.py research/rl_sizer` | ✅ Clean |
| Streamlit startup latency | ≤ 2 s |
| RL pilot (dry-run) | Executes 1 episode per cron cycle without impacting live trading |
| RiskAutotuner feedback | Logs `autotune adjustments={...}` with normalized Sharpe inputs |

---

## 🔒 Production Safety
- Executor remains production-safe (`ENV=prod`, `ALLOW_PROD_WRITE=1`).
- RL Pilot confined to **research mode** (`ENV=dev` or `research` only).
- All new modules follow existing Firestore + logging circuit-breaker patterns.
- Cached JSON artefacts stored under `logs/cache/` with rotation policy.

---

## 📈 Metrics Preview
**Dashboard Tabs**
1. **Router Health 2.0** — Confidence-weighted PnL curves, rolling Sharpe, per-symbol normalized metrics.  
2. **ML Confidence** — Live model confidence chart, top-N feature bars.  
3. **RL Pilot** — Episode reward, drawdown, and Sharpe progression.

---

## 🧠 Post-Patch Verification
```bash
# Verify environment
sudo supervisorctl restart hedge:executor
sudo tail -n 40 /var/log/supervisor/executor.log | grep autotune

# Run research pilot dry-run
python3 -m research.rl_sizer.runner --episodes 10 --dry-run

# Regenerate Router Health metrics
python3 -m scripts.doctor -v | tail -n 20
