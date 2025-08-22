docs/QuantWizardPrompt.md
docs/Runbook.md                        # quick-start + routine ops
docs/Investors_Access.md               # how to reach the dashboard (Basic Auth), scope, read-only
docs/Mainnet_Cutover_Checklist.md      # the checklist above, as a standalone doc
docs/Telemetry_Format.md               # heartbeat schema & examples

config/strategy_config.json            # clean, three strategies (BTC breakout, ETH momentum, SOL fast lane)
config/pairs_universe.json             # whitelisted symbols + per-symbol overrides (min_notional, leverage)
config/risk_limits.json                # KILL_DD_PCT, KILL_DAILY_LOSS_PCT, MAX_POSITIONS, MAX_LEVERAGE
config/dashboard.yml                   # small UI tuneables (tail length=10, kpi thresholds)
config/firebase_creds.json.example     # placeholder (real JSON lives off-repo)
.env.example                           # BINANCE_TESTNET, API keys, FIREBASE path, TELEGRAM tokens

execution/executor_live.py             # fresh minimal executor (hedge-mode aware, robust logs)
execution/signal_screener.py           # emits intents; breadcrumbs include z/rsi/atr/cross flags
execution/exchange_utils.py            # single source for price/klines/filters/dualSide; signed helpers
execution/rules_sl_tp.py               # pure functions: calc SL/TP triggers & exits
execution/pipeline_probe.py            # one-shot: consume screener -> place one order (for diagnostics)
execution/flatten_all.py               # safe “panic close” hedge positions (per symbol or all)

dashboard/app.py                       # Streamlit: single Overview tab per spec
dashboard/dashboard_utils.py           # Firestore + local fallback readers, compaction helpers
dashboard/theme.css                    # light minimal style (optional)

telegram/heartbeat.py                  # 8h NAV + trend + recent PnL summary sender
telegram/telegram_utils.py             # safe wrapper; retries; rate-limit; dry-run switch

scripts/quick_checks.sh                # one-liners: supervisor, NGINX, Firestore env, dualSide, klines
scripts/leverage_once.py               # set leverage per symbol safely (idempotent)
scripts/margin_mode_once.py            # set marginType=CROSSED once per symbol
scripts/seed_universe.py               # add symbols to universe from exchangeInfo (read-only)

tests/test_rules_sl_tp.py              # unit tests for SL/TP module
tests/test_filters_rounding.py         # validates stepSize/minQty/minNotional rounding logic

ops/hedge.conf                         # supervisor program group (executor + dashboard only)
ops/nginx_site.conf                    # site conf (reverse proxy + Basic Auth)
ops/grep_patterns.txt                  # for log tails (“[screener]”, “[decision]”, “[screener->executor]”, errors)
