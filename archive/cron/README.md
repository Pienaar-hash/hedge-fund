# Cron Jobs

Placeholder for v5.9 cron definitions (ml_retrain, sync, etc.). Keep tasks under this directory so repo layout stays normalized.

## Execution Log Rotation

Use `cron/rotate_exec_logs.sh` to prune and archive `logs/execution/*.jsonl`. This wraps `scripts/ops_cleanup.py --rotate` with sane defaults (5 MB threshold, keep 10 archives).

Example crontab (run every hour):

```
0 * * * * /bin/bash /root/hedge-fund/cron/rotate_exec_logs.sh >> /var/log/hedge/rotate.log 2>&1
```

Supervisor snippet:

```
[program:hedge-log-rotate]
command=/bin/bash /root/hedge-fund/cron/rotate_exec_logs.sh
directory=/root/hedge-fund
autostart=true
autorestart=true
startsecs=0
stopwaitsecs=5
stderr_logfile=/var/log/hedge/log-rotate.err
stdout_logfile=/var/log/hedge/log-rotate.out
```

Override `MAX_ROTATE_BYTES` or `MAX_ROTATE_ARCHIVES` env vars to tune behavior. Running without `--rotate` still performs the legacy cleanup (veto logs, etc.).
