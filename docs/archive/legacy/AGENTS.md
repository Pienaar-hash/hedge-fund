# AGENTS.md — GPT Hedge v5.6

## Overview
GPT Hedge v5.6 introduces a complete **Execution → Fill → PnL** integrity chain.  
ACKs and FILLs are now distinct; realized PnL is computed from confirmed fills;  
and a dedicated **Backfill Agent** can rebuild historical gaps.  

The system operates through three cooperating agent layers:

1. **Execution Agents** — manage intents, risk gating, routing, and fill confirmation.  
2. **Doctor Agents** — audit router health, latency, slippage, and equity integrity.  
3. **Backfill Agents** — reconcile historical fills / PnL from Binance REST.

---

## Agent Roles

### 🧠 Execution Agent
- Consumes intents → routes orders via the router.  
- Logs `order_ack` on `NEW` status only.  
- Waits for confirmed `order_fill` (via WS userData or REST).  
- Computes realized PnL using FIFO lot tracking.  
- Emits `order_close` when exposure → 0.  
- Publishes to Firestore for dashboard + doctor telemetry.

### 🧪 Doctor Agent
- Aggregates execution metrics:
  - Fill-rate %
  - Median latency (ACK → FILL)
  - Median slippage vs baseline
  - Total realized PnL and fees
- Flags SLO breaches (red / yellow / green) on the dashboard.

### 🩺 Backfill Agent
- Script `scripts/backfill_fills_pnl.py`  
- Reconstructs missing fills / PnL using:
  - `/fapi/v1/order`
  - `/fapi/v1/userTrades`
- Outputs `logs/execution/orders_events_backfilled.jsonl`.  
- Dry-run by default — use `--apply` to persist.

---

## Guardrails
- **Never** commit credentials.  
- Enforce NAV / exposure caps in `execution/risk_limits.py`.  
- Maintain strict ACK ↔ FILL separation — no premature logging.  
- Restart supervisor after structural updates:
  - `executor`, `sync_state`, `dashboard`, `doctor`.  
- Run `pytest + ruff + mypy` before commit.  
- Supervisor logs rotate daily via `supervisord.conf`.

---

## Acceptance for PRs
- ✅ `pytest -q` green  
- ✅ `ruff check .` clean (or auto-fix)  
- ✅ `mypy .` clean (or documented suppressions)  
- ✅ Doctor / Router Health dashboards show non-zero fills & PnL

---

## Version Reference
- **v5.6 Execution Integrity Hotfix** — November 2025  
- **Auditor:** GPT-5 Codex Quant  
- **Artifacts:**  
  - `/mnt/data/repo_audit_summary.md`  
  - `/mnt/data/repo_audit_findings.csv`  
  - `/mnt/data/repo_patch_suggestions.json`
