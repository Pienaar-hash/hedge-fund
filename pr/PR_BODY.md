Title: AI micro-notional futures: 20× risk guardrails, signal doctor, FS health, dash tweaks, NGINX, TG mini-report

Summary
- Micro-notional stack: targets $10–$20 gross notional per order with up to 20× leverage (tiny margin). Deterministic logic; AI used only for brief market notes.
- Signal diagnostics: `execution/signal_doctor.py` prints sizing, exchange min, leverage, and veto stack per whitelisted symbol.
- Risk updates: global leverage cap 20×; min_notional_usdt=10; tighter trade/portfolio caps with burst/day-loss brakes; per-symbol tiny caps.
- Orderbook gate: lightweight REST depth snapshot → imbalance veto/boost (rate-limited) behind feature flag.
- Firestore health: `scripts/fs_doctor.py` checks counts and env/testnet mixing; dashboard shows FS OK/Mixed marker.
- Telegram mini-report: tiny 7d NAV PNG + 80–120 word AI market note; dry-run friendly.
- Dashboard: reads only `hedge/{ENV}/state/*`, `trades`, `risk`; shows open positions and 24h tables; compact recency banner with FS status.
- NGINX: ready server block to reverse proxy Streamlit on 8501 with Basic Auth + TLS notes.

Changes
- New: `execution/signal_doctor.py`, `execution/orderbook_features.py`, `execution/telegram_report.py`.
- New: `strategies/momentum_vol.py`, `strategies/market_maker.py` (MM OFF by default).
- New: `scripts/fs_doctor.py`.
- New: `deploy/nginx/hedge-dashboard.conf`.
- Tweaks: `execution/executor_live.py` (DRY_RUN default=1; interpret capital_per_trade as GROSS; correct order sizing margin=gross/lev).
- Tweaks: `execution/signal_screener.py` (DEBUG_SIGNALS; orderbook gate; correct min-notional checks).
- Tweaks: `execution/risk_limits.py` (accept `max_portfolio_gross_nav_pct`).
- Config: `config/risk_limits.json` (20× caps; min_notional_usdt=10; whitelist & tiny caps); `config/strategy_config.json` (per-symbol capital_per_trade=10.0; lev=20; features).
- Dashboard: `dashboard/app.py` FS status marker.
- Docs: `docs/CloudEnvironment.md` NGINX + TLS + Basic Auth; sizing math; DRY_RUN flip instructions.

Validation (local)
- pytest -q: [paste output]
- ruff check .: [paste output]
- mypy .: [paste output]
- Signals doctor: `python -m execution.signal_doctor --env prod --symbols BTCUSDT,ETHUSDT,SOLUSDT,LTCUSDT,LINKUSDT --once`
  - Shows at least one `would_emit` at $10 notional with lev=20; clear veto reasons for others.
- FS doctor: `python -m scripts.fs_doctor --env prod` → OK, no mixing; prints counts for nav/positions/trades/risk.
- Telegram report: `python -m execution.telegram_report --dry-run` → writes `/tmp/nav.png` and logs composed AI note (no advice).
- Dashboard: Firestore source, open positions only; trades/risk tables limited to last 24h; FS status marker.
- NGINX: conf present; no installation performed.

Notes & toggles
- DRY_RUN default is 1. Supervisor flip later (see docs).
- Feature flags: `DEBUG_SIGNALS=1` for compact per-candidate logs; `MARKET_MAKER_ENABLED=1` (default OFF); orderbook gate via strategy config features.
- Example risk-block JSON (executor log): `{ "phase":"blocked", "symbol":"BTCUSDT", "side":"BUY", "reason":"below_min_notional", "notional":9.0, "nav":1000.0, ... }`

Reviewers
- @codex — please review risk guardrails, FS paths, and dashboard status markers.
