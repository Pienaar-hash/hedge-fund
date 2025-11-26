# Hedge Fund Ops Guide — v6.0-rc

## Overview
Supervisor runs **five** processes for the v6 runtime:

1. `hedge-executor` — Executes trades, writes local `logs/state/*.json`.
2. `hedge-sync_state` — Mirrors local state to Firestore (when enabled).
3. `hedge-dashboard` — Streamlit dashboard reading local state.
4. `hedge-pipeline-shadow-heartbeat` — Maintains `pipeline_v6_shadow_head.json`.
5. `hedge-pipeline-compare` — Maintains `pipeline_v6_compare_summary.json`.

Local JSON under `logs/state/` is canonical; Firestore is optional and gated by env.

---

## Environment Variables

| Variable                    | Purpose                                           | Example |
|-----------------------------|---------------------------------------------------|---------|
| `ENV`                       | Environment label                                  | `prod`  |
| `PYTHONPATH`                | Module path                                        | `/root/hedge-fund` |
| `ALLOW_PROD_WRITE/SYNC`     | Gate Firestore writes                              | `1`     |
| `FIRESTORE_ENABLED`         | Enable Firestore client                            | `1`     |
| `FIRESTORE_CREDS_PATH`      | Firestore credentials                              | `/root/hedge-fund/config/firebase_creds.json` |
| `INTEL_V6_ENABLED`          | Enable v6 intel loop                               | `1`     |
| `RISK_ENGINE_V6_ENABLED`    | Use v6 risk engine                                 | `1`     |
| `PIPELINE_V6_SHADOW_ENABLED`| Pipeline shadow on                                | `1`     |
| `ROUTER_AUTOTUNE_V6_ENABLED`| Router intel on                                   | `1`     |
| `FEEDBACK_ALLOCATOR_V6_ENABLED` | Allocator intel on                           | `1`     |
| `ROUTER_AUTOTUNE_V6_APPLY_ENABLED` | Router auto-apply (0 safe / 1 live)       | `0`     |

---

## Process Control

### Restart all
```bash
sudo supervisorctl restart hedge:

# individual processes
sudo supervisorctl restart hedge:hedge-executor
sudo supervisorctl restart hedge:hedge-sync_state
sudo supervisorctl restart hedge:hedge-dashboard
sudo supervisorctl restart hedge:hedge-pipeline-shadow-heartbeat
sudo supervisorctl restart hedge:hedge-pipeline-compare


tail -f /var/log/hedge-executor.out.log
tail -f /var/log/hedge-executor.err.log
tail -f /var/log/hedge-dashboard.err.log

tail -f /var/log/supervisor/sync_state.err
tail -f /var/log/supervisor/sync_state.out

sudo supervisorctl status

http://{{DASHBOARD_HOST}}:8501

cd /root/hedge-fund
git fetch --all
git checkout <tag-or-commit>

sudo supervisorctl restart hedge:

### Reboot & Tail Logs
Reboots are rarely required, but when kernel patches or driver updates land you can bounce the whole node and rehydrate processes quickly.

```bash
sudo reboot

# once SSH is back:
cd /root/hedge-fund
sudo supervisorctl restart all
sudo supervisorctl status

# follow critical logs
tail -fn200 /var/log/hedge-executor.out.log
tail -fn200 /var/log/hedge-executor.err.log
tail -fn200 /var/log/hedge-dashboard.err.log
```

The executor will emit `[v6] flags ...` on startup; confirm v6 flags match your intended config after any restart.
sudo supervisorctl status
