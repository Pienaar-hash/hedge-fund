# PROJECT BRIEF

**Version:** v7.9  
**Last updated:** 2026-06-05  
**Status:** Testnet (not yet live)

---

## What This Project Is

A systematic, multi-strategy crypto futures trading system targeting risk-adjusted returns on digital assets. The system runs a fully automated pipeline from signal generation through execution, with layered risk controls and a real-time investor dashboard.

The primary design goal is **proof, logging, and risk governance** — every signal, veto, fill, and NAV snapshot is logged in a reconstructable form before strategies are scaled.

---

## NAV / Capital

| Item | Value |
|------|-------|
| Inception baseline NAV | $10,760 (soft reference) |
| NAV scope (trading) | Binance UM Futures equity only |
| NAV scope (AUM) | Trading NAV + off-exchange BTC/ETH/USDC holdings |
| Target investor-ready AUM | TBD (pending live validation) |

Treasury (off-exchange) holdings tracked in `config/offexchange_holdings.json`. Mark pricing via Binance spot or CoinGecko fallback.

---

## Exchanges

| Exchange | Status | Purpose |
|----------|--------|---------|
| Binance UM Futures | Active (testnet) | All live/testnet trading |
| KuCoin | Not implemented | No active integration |

All trading is Binance USDT-margined perpetual futures. KuCoin appears only in archived/legacy code.

---

## Execution Mode

| Setting | Current value | Source |
|---------|--------------|--------|
| `ENV` | `production` (supervisor: `prod`) | `config/runtime.yaml`, `deploy/supervisor/hedge.conf` |
| `BINANCE_TESTNET` | `1` — exchange still routes to testnet | `deploy/supervisor/hedge.conf` |
| `dry_run` | `false` (real testnet orders placed) | `config/runtime.yaml` |
| `FIRESTORE_ENABLED` | `0` | `.env` |
| `TELEGRAM_ENABLED` | `0` | `.env` / `config/telegram_v7.json` |
| Trading window | 06:00–17:00 UTC, weekdays only | `config/runtime.yaml` |

The environment is labelled `production` / `prod` in the runtime and supervisor configs, but `BINANCE_TESTNET=1` keeps all exchange traffic on the Binance testnet. Transitioning to the live exchange requires flipping `BINANCE_TESTNET=0`, rotating credentials, enabling Firestore + Telegram, and completing the live execution audit.

---

## Three Managed Services

| Service | Supervisor name | Role |
|---------|----------------|------|
| Trading engine | `hedge-executor` | Signal → execution loop (5s cycle) |
| State sync | `hedge-sync_state` | Publishes local state to Firestore (optional) |
| Dashboard | `hedge-dashboard` | Streamlit investor UI on port 8501 |

All three are managed by Supervisor with `autorestart=true`.

---

## Investor-Readiness Checklist

- [ ] Execution audit: confirm signals become intended orders on testnet
- [ ] Backtest validity audit: no lookahead, fees correct, leverage assumptions match live
- [ ] Risk policy audit: all halt conditions reachable and tested
- [ ] Logging audit: every trade reconstructable from JSONL logs
- [ ] Dashboard truth audit: NAV/drawdown match execution state within 30s
- [ ] 30 consecutive days testnet without uncontrolled drawdown
- [ ] Live credential rotation + pre-flight checklist
- [ ] Investor report cadence: weekly, template at `INVESTOR_REPORT_TEMPLATE.md`
