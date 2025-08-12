# Hedge — Ops Notes

## Processes (managed by Supervisor)
- **hedge-executor**: main trading loop (`execution.executor_live`)
- **hedge-sync**: Firestore sync loop (`execution.sync_state`)
- **hedge-dashboard** (optional): Streamlit UI on port 8501

### Start / Stop / Restart
```bash
sudo supervisorctl status
sudo supervisorctl start hedge-executor
sudo supervisorctl stop hedge-executor
sudo supervisorctl restart hedge-executor

sudo supervisorctl restart hedge-sync
sudo supervisorctl start hedge-dashboard   # if enabled

# Ops Runbook

## Start/Stop/Restart
```bash
sudo supervisorctl reread && sudo supervisorctl update
sudo supervisorctl status hedge-executor hedge-sync hedge-dashboard
sudo supervisorctl start  hedge-executor
sudo supervisorctl stop   hedge-executor
sudo supervisorctl restart hedge-executor
