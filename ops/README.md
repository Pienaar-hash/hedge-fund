# Hedge Fund Ops Guide — v7.6

## Overview

Supervisor runs **three** processes for the v7.6 runtime:

1. `hedge-executor` — Executes trades, writes local `logs/state/*.json`
2. `hedge-sync_state` — Mirrors local state to Firestore (when enabled)
3. `hedge-dashboard` — Streamlit dashboard reading local state

Local JSON under `logs/state/` is canonical; Firestore is optional and gated by env.

> **Note:** Legacy pipeline processes (`hedge-pipeline-shadow-heartbeat`, `hedge-pipeline-compare`) were removed in v7.6.

---

## State Surfaces

| Surface | Path | Description |
|---------|------|-------------|
| `nav_state.json` | `logs/state/` | NAV, exposure, drawdown |
| `positions_state.json` | `logs/state/` | Open futures positions |
| `positions_ledger.json` | `logs/state/` | Unified positions + TP/SL |
| `risk_snapshot.json` | `logs/state/` | Risk engine state |
| `router_health.json` | `logs/state/` | Router metrics |
| `offchain_assets.json` | `logs/state/` | Off-exchange treasury holdings |
| `offchain_yield.json` | `logs/state/` | Treasury yield rates |
| `kpis_v7.json` | `logs/state/` | Dashboard KPIs |
| `engine_metadata.json` | `logs/state/` | Engine version info |

---

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `ENV` | Environment label | `prod` |
| `DRY_RUN` | Skip actual order placement | `1` |
| `BINANCE_TESTNET` | Use testnet endpoints | `0` |
| `PYTHONPATH` | Module path | `/root/hedge-fund` |
| `FIRESTORE_ENABLED` | Enable Firestore client | `0` |
| `INTEL_V6_ENABLED` | Enable v6 intel loop | `1` |
| `RISK_ENGINE_V6_ENABLED` | Use v6 risk engine | `1` |
| `ROUTER_AUTOTUNE_V6_ENABLED` | Router intel on | `1` |
| `ROUTER_AUTOTUNE_V6_APPLY_ENABLED` | Router auto-apply (0=safe) | `0` |

---

## Process Control

### Restart All
\`\`\`bash
sudo supervisorctl restart hedge:
\`\`\`

### Individual Processes
\`\`\`bash
sudo supervisorctl restart hedge:hedge-executor
sudo supervisorctl restart hedge:hedge-sync_state
sudo supervisorctl restart hedge:hedge-dashboard
\`\`\`

### Check Status
\`\`\`bash
sudo supervisorctl status
\`\`\`

---

## Logs

### Executor
\`\`\`bash
tail -f /var/log/hedge-executor.out.log
tail -f /var/log/hedge-executor.err.log
\`\`\`

### Dashboard
\`\`\`bash
tail -f /var/log/hedge-dashboard.err.log
\`\`\`

---

## Dashboard Access

\`\`\`
http://{{DASHBOARD_HOST}}:8501
\`\`\`

---

## Deployment

### Update Code
\`\`\`bash
cd /root/hedge-fund
git fetch --all
git checkout <tag-or-commit>
sudo supervisorctl restart hedge:
\`\`\`

### Reboot & Recover
\`\`\`bash
sudo reboot

# After SSH reconnects:
cd /root/hedge-fund
sudo supervisorctl restart all
sudo supervisorctl status
tail -fn50 /var/log/hedge-executor.out.log
\`\`\`

---

## Troubleshooting

### Orders Being Vetoed?
\`\`\`bash
tail -100 logs/execution/risk_vetoes.jsonl | jq '{reason: .veto_reason}'
\`\`\`

### NAV Stale?
\`\`\`bash
cat logs/state/nav_state.json | jq '.age_s'
\`\`\`

### Dashboard Not Loading?
\`\`\`bash
sudo supervisorctl status hedge:hedge-dashboard
tail -50 /var/log/hedge-dashboard.err.log
\`\`\`

---

## Version

- **Runtime**: v7.6
- **Dashboard**: layout_v7_6.py + app_v7_6.py
- **Engine**: Risk Engine v6, Hybrid Alpha v2, Vol Regime Model
