## Hedge Fund Ops Guide

### Supervisor Commands
sudo supervisorctl status
sudo supervisorctl restart hedge-sync
sudo supervisorctl restart hedge-executor

### Logs
sudo tail -f /var/log/hedge/sync.out.log
sudo tail -f /var/log/hedge/executor.out.log
