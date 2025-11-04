# 🧭 Quant Infrastructure Patch — v5.7  
**Commit:** `v5.7 Factor Fusion + Dynamic Capital Allocation + Async Caching`  
**Author:** Codex (`codex@example.com`)  
**Date:** 2025-11-21 (target)  

---

## 🚀 Executive Summary
Sprint v5.7 transitions the GPT Hedge system from isolated strategy analytics to **portfolio intelligence**.  
It introduces multi-strategy correlation monitoring, dynamic capital weighting, a Factor Fusion layer combining ML and TA signals, and asynchronous caching for sub-second dashboard performance.

**Core Upgrades**
- **Multi-Strategy Correlation Analysis** — quantifies dependencies across strategies to reveal clustering and diversification.
- **Dynamic Capital Allocation** — distributes capital adaptively based on Sharpe, volatility, and correlation penalties.
- **Factor Fusion Layer** — unifies ML confidence, classical indicators, and volatility/risk scores into fused alpha.
- **Async Caching Framework** — replaces blocking cache reads with background refresh tasks for smoother dashboards.

---

## 🧩 File-Level Additions & Modifications

| Module | Purpose / Key Changes |
|---------|-----------------------|
| **`research/correlation_matrix.py`** | Computes rolling Pearson/Spearman correlations across strategies and persists JSON snapshots for the dashboard. |
| **`execution/capital_allocator.py`** | Dynamic strategy weighting via inverse-vol × Sharpe × correlation penalties; emits `logs/cache/capital_allocation.json`. |
| **`research/factor_fusion/`** | Ridge-based factor fusion layer plus TA helpers (`prepare_factor_frame`, `compute_rsi`, etc.) to blend ML and classical signals. |
| **`dashboard/async_cache.py`** | Async gather/refresh primitives supplying non-blocking doctor, router, telemetry, correlation, and allocation caches. |
| **`dashboard/app.py`** | Wires async cache feeds, adds “Portfolio Correlation” & “Factor Fusion” tabs, and surfaces fused-alpha diagnostics. |
| **`tests/test_correlation_matrix.py`** | Asserts matrix symmetry, diagonal unity, and directory loader behaviour. |
| **`tests/test_capital_allocator.py`** | Guarantees weight normalization and correlation-penalty behaviour. |
| **`tests/test_factor_fusion.py`** | Validates positive IC / Sharpe and deterministic weights for synthetic data. |
| **`tests/test_async_cache.py`** | Exercises async gather + periodic refresh stop semantics. |

---

## 🧪 Validation Checklist

| Check | Result / Expectation |
|--------|----------------------|
| `pytest -q tests/test_correlation_matrix.py` | ✅ Pass |
| `pytest -q tests/test_capital_allocator.py` | ✅ Pass |
| `pytest -q tests/test_factor_fusion.py` | ✅ Pass |
| `pytest -q tests/test_async_cache.py` | ✅ Pass |
| `ruff check --fix execution/ dashboard/ research/` | ✅ Clean |
| `mypy --strict execution/... research/...` | ⚠️ Legacy ignores acceptable |
| Streamlit latency | < 1 s render |
| Capital Allocator dry-run | Writes valid weight JSON without order routing |
| RL/Risk feedback | Autotune scales correctly with normalized Sharpe |

---

## 🔒 Production Safety
- Capital allocation operates in **dry-run** until validated (`ENABLE_DYNAMIC_ALLOC=0`).
- Factor Fusion and RL layers restricted to **research** environments (`ENV=dev` or `research`).
- Async caching isolated from execution threads; fails silently on error.
- All I/O writes gated by existing Firestore and ENV guards.

---

## 📊 Dashboard Additions
1. **Portfolio Correlation** — Rolling inter-strategy correlation matrix and diversification metrics.  
2. **Factor Fusion** — Blended alpha curves, feature attribution bars, and ensemble diagnostics.  
3. **Async Telemetry** — Non-blocking updates for doctor snapshot, router health, and ML confidence feeds.

---

## 🧠 Post-Patch Verification
```bash
# Restart stack
sudo supervisorctl restart hedge:dashboard hedge:executor

# Validate correlations (writes logs/cache/strategy_correlation.json)
python3 -m research.correlation_matrix logs/strategy_returns --window 200

# Inspect allocation output
cat logs/cache/capital_allocation.json | jq .

# Confirm async cache helpers
python3 -m dashboard.async_cache

# Run core tests
pytest -q tests/test_correlation_matrix.py tests/test_capital_allocator.py tests/test_factor_fusion.py tests/test_async_cache.py
