# Mainnet Cutover Checklist (VS Agent Only)

## Preflight
- [ ] Branch/tag: `release/v4.2-mainnet` / `v4.2-mainnet-rc`
- [ ] Tests/linters pass locally
- [ ] `strategy_config.json`: whitelist=["BTCUSDT"], small capital_per_trade, TP/SL/ATR-trail on
- [ ] `execution/risk_limits.py` present and imported by executor
- [ ] Firestore creds path correct; Telegram tokens loaded (optional)

## Env (Supervisor or .env)
- [ ] ENV=prod, BINANCE_TESTNET=0, DRY_RUN=1, HEDGE_MODE=1
- [ ] PYTHONPATH=/root/hedge-fund, PYTHONUNBUFFERED=1
- [ ] FIREBASE_CREDS_PATH set
- [ ] TELEGRAM_ENABLED=1 (optional)

## Dry Run (one-shot)
- [ ] Executor runs 1 loop cleanly (no tracebacks)
- [ ] nav_log.json has non-zero values
- [ ] peak_state.json created/updated
- [ ] Firestore shows NAV/positions
- [ ] Telegram heartbeat received

## Services
- [ ] Supervisor reread/update done
- [ ] hedge-executor and hedge-dashboard running
- [ ] NGINX reverse proxy working; dashboard reachable

## Go Live
- [ ] DRY_RUN=0 (tiny capital; BTCUSDT only)
- [ ] 30–60 min soak: no errors, orders bounded by risk
- [ ] Kill-switch verified (stop supervisor group)

## Post-Cutover
- [ ] Tag `v4.2-mainnet-cutover`
- [ ] Gradually raise capital / add next symbol
- [ ] Daily ops: morning check, telegram alerts, log health, drawdown alerts
