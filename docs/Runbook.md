# docs/Runbook.md

## Hedge Fund Bot — Runbook (Quick-start & Routine Ops)

### 0) What this system is

A lightweight, read-only **Streamlit** dashboard fronted by **NGINX** with Basic Auth, and a single **executor** that turns screener signals into hedge-mode USD-M futures orders on **Binance**. State lives in **Firestore** (`hedge/prod/state/{nav,positions,...}`), with local fallbacks.

* Server: Ubuntu @ `{{DASHBOARD_HOST}}`
* Reverse proxy: NGINX → Streamlit on `127.0.0.1:8501` (Basic Auth)
* Supervisor programs: `hedge-executor`, `hedge-dashboard` (no extras)
* Firestore pathing: `hedge/prod/state/*` (docs: `nav`, `positions`, `leaderboard` if present)
* Local read-only state: `peak_state.json`, `synced_state.json`, optional `trade_log.json`
* Exchange mode: **Hedge (dual-side)**; `positionSide` set on every order
* Reserve: **0.013 BTC** shown independently (do **not** inject into NAV)

---

## 1) Quick start

### 1.1 First-time / after reboot

```bash
# ensure venv active when running Python directly
cd /root/hedge-fund
source venv/bin/activate

# confirm supervisor programs
supervisorctl reread && supervisorctl update
supervisorctl restart hedge-executor hedge-dashboard
supervisorctl status
```

### 1.2 Health one-liners (idempotent)

```bash
# Supervisor health
supervisorctl status

# Dashboard reachability (Basic Auth prompt expected; replace USER:PASS and host)
curl -I -u USER:PASS http://{{DASHBOARD_HOST}}/

# Executor environment (Firestore / Google paths)
pid=$(supervisorctl pid hedge-executor); tr '\0' '\n' < /proc/$pid/environ | egrep 'GOOGLE|FIREBASE|BINANCE_|USE_FUTURES|ENV='

# Hedge vs one-way (should be True)
PYTHONPATH=/root/hedge-fund venv/bin/python - <<'PY'
from execution.exchange_utils import _is_dual_side
print("dualSide:", _is_dual_side())
PY

# Log tail (screener breadcrumbs + decisions + intent bridge)
tail -n 150 /var/log/hedge-executor.out.log | egrep '\[screener\]|\[decision\]|\[screener->executor\]|\[executor\]'
```

---

## 2) Routine ops

### 2.1 Daily AM check (≤ 2 min)

1. `supervisorctl status` — both RUNNING
2. Dashboard loads via NGINX; KPIs render (NAV, DD, 24h PnL)
3. Log tail shows periodic `[screener] attempted=… emitted=…` and occasional `[executor] ORDER_REQ 200`
4. Telegram heartbeat due every **8h** (check last post time)

### 2.2 Evening sweep (≤ 3 min)

* Open positions plausible vs strategy sizing
* `grep -E '\[ORDER_ERR\]|-1013|-1106|HTTPError' /var/log/hedge-executor.out.log | tail` (should be empty)
* Firestore doc counts increasing as NAV points append

### 2.3 Weekly

* Update `config/pairs_universe.json` if expanding symbols
* Verify leverage margin settings once per newly added symbol
* Rotate dashboard Basic Auth passwords if required (out-of-band)

---

## 3) Signal → Execution pipeline (what to expect)

* Screener line per symbol, with breadcrumbs:

  ```
  [screener] SOLUSDT 15m z=+2.57 rsi=53.3 atr=0.011 cross_up=True cross_down=False in_trade=False side=-
  [decision] {... "veto": [] ...}
  [screener->executor] {... "symbol":"SOLUSDT","signal":"BUY",...}
  ```
* Executor mirrors the intent and fires a hedge-mode order:

  ```
  [executor] INTENT {...}
  [executor] SEND_ORDER {... "positionSide":"LONG"|"SHORT", "quantity": ...}
  [executor] ORDER_REQ /fapi/v1/order 200 {... "status":"NEW"|"FILLED"...}
  ```
* Heartbeat:

  ```
  [executor] account OK — futures=True testnet=<bool> dry_run=False balances: [...] positions: N
  ```

---

## 4) Expanding the asset universe

1. Add symbol to whitelist:

```json
// config/pairs_universe.json
{
  "symbols": ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"],
  "overrides": {
    "BNBUSDT": { "target_leverage": 3, "min_notional": 5.0 }
  }
}
```

2. Add/enable a strategy block for the symbol in `config/strategy_config.json`.
3. Set leverage & margin once (idempotent helper script or console one-liner).
4. Restart executor: `supervisorctl restart hedge-executor`.

---

## 5) Risk, SL/TP & lifecycle

* SL/TP computed in pure rules (no API dependency).
* Circuit breakers (configure in `config/risk_limits.json`):

  * `KILL_DD_PCT`, `KILL_DAILY_LOSS_PCT`, `MAX_POSITIONS`, `MAX_LEVERAGE`.
* Per-symbol **CROSSED** margin, leverage set once.
* Panic close (testnet or mainnet): a single script sends MARKET closes with correct `positionSide`.

---

## 6) Common errors (fast fixes)

* **`-1013` LOT\_SIZE/minNotional** → qty rounding too small → increase `capital_per_trade` or round up to step.
* **`-1106 reduceonly not required`** (Hedge Mode, MARKET) → drop `reduceOnly`; positionSide + qty ≤ current size is enough.
* **`-2015` permission/IP** → wrong key type, incorrect IP whitelist, or using mainnet endpoints with testnet keys (and vice-versa).

---

## 7) Logs & grep

```
/var/log/hedge-executor.out.log
Patterns: [screener], [decision], [screener->executor], [executor] INTENT|SEND_ORDER|ORDER_REQ|ORDER_ERR
```

---

## 8) Shutdown / restart

```bash
supervisorctl stop hedge-executor hedge-dashboard
supervisorctl start hedge-executor hedge-dashboard
```

---

# docs/Investors\_Access.md

## Hedge Fund Dashboard — Investor Access (Read-Only)

**URL**
`http://167.235.205.126/` (Basic Auth)

**Access**

* You will receive a unique username/password out-of-band.
* Access is **read-only**. There are **no trading controls**.

**What you will see** (single “Overview” tab)

* **KPIs**: NAV, 24h PnL, Drawdown, Open Positions, Exposure
* **NAV chart** (Firestore primary; falls back to local `peak_state.json` if needed)
* **Positions** table (hedge-mode aware LONG/SHORT legs)
* **Signals** table (latest intents)
* **Trade log** (most recent 5)
* **Screener tail** (last 10 lines: z-score, RSI, ATR, cross flags)
* **BTC reserve** shown as **0.013 BTC** (informational; **not** added to NAV)

**Data freshness**

* NAV & positions update continuously (executor heartbeat).
* A Telegram summary is posted every **8 hours** (NAV, 24h delta, brief trend, open positions).

**Security**

* Basic Auth over HTTP; credentials can be rotated at any time.
* Dashboard is isolated behind NGINX; backend processes are not exposed.

**Glossary**

* **NAV**: Account equity estimate (USDT + Σ unrealized PnL; Firestore NAV series if present).
* **Drawdown**: Peak-to-current NAV % drop.
* **Hedge Mode**: Separate LONG and SHORT books per symbol; positions labeled with `positionSide`.

---

# docs/Mainnet\_Cutover\_Checklist.md

## Mainnet Cutover — Checklist

> Goal: Switch from testnet to **mainnet** with minimal moving parts, provable entry/exit, and safe initial sizing.

### A) Preconditions (testnet proven)

* [ ] Screener emits and executor places orders; logs show `[ORDER_REQ 200]`.
* [ ] At least one **entry & exit** per strategy recorded with SL/TP decisions in logs.
* [ ] Dashboard loads externally (NGINX) with all KPIs and tables.
* [ ] Telegram heartbeat posting every **8h**.
* [ ] Panic-close script verified on testnet.

### B) Exchange readiness (mainnet)

* [ ] Mainnet API key has **USD-M Futures** trading permissions; IP whitelist updated.
* [ ] Account in **Hedge Mode** (dual-side = true).
* [ ] Margin type **CROSSED** for all enabled symbols.
* [ ] Leverage set per symbol (e.g., 3× or 5×).
* [ ] Small USDT balance available (plus any native asset for fees if needed).

### C) Server & config

* [ ] Backup: `tar zcf /root/hedge-backup-$(date +%F).tgz /root/hedge-fund`
* [ ] Update `/root/hedge-fund/.env`:

  * `BINANCE_TESTNET=0`
  * Mainnet `BINANCE_API_KEY` / `BINANCE_API_SECRET`
  * Firestore creds unchanged
* [ ] Verify `.env` perms `chmod 600`.
* [ ] Restart:

  ```bash
  supervisorctl restart hedge-executor hedge-dashboard
  sleep 3
  tail -n 80 /var/log/hedge-executor.out.log | egrep 'account OK|positionSide'
  ```
* [ ] Confirm `dualSide: True` via helper in venv.

### D) Dry micro-checks on mainnet

* [ ] **Read-only** probes: balances, exchangeInfo, klines (no orders).
* [ ] Place **one micro trade per symbol** (minQty), then close it, verifying:

  * `[executor] SEND_ORDER` includes `positionSide`
  * `[ORDER_REQ 200]` and on-chain fills (via `userTrades`)
* [ ] Dashboard shows the fill + position/close correctly.

### E) Risk & sizing gates

* [ ] `config/risk_limits.json` set to production values:

  * `KILL_DD_PCT`, `KILL_DAILY_LOSS_PCT`, `MAX_POSITIONS`, `MAX_LEVERAGE`
* [ ] `config/strategy_config.json` capital\_per\_trade and leverage sized conservatively for day 1.
* [ ] Panic-close script updated to **mainnet base URL** if it is environment-aware (or uses env var).

### F) Investor comms

* [ ] Confirm dashboard access (Basic Auth) for both investors.
* [ ] Explain reporting cadence (dashboard live; Telegram every 8h).
* [ ] Provide a short “how to read the dashboard” blurb.

### G) Go-live

* [ ] Flip `BINANCE_TESTNET=0` (done) and confirm first two live orders + closures behave.
* [ ] Monitor for **24 hours**:

  * Logs for errors (`-1013`, `-1106`, timeouts)
  * NAV drift vs expectations
  * Heartbeats firing on schedule

Rollback plan: restore `.env` with testnet keys and `BINANCE_TESTNET=1`, `supervisorctl restart hedge-executor hedge-dashboard`.

---

# docs/Telemetry\_Format.md

## Telegram Heartbeat — Message Schema & Examples

**Cadence**: Every **8 hours** (cron).
**Audience**: Operators & (optionally) investors.
**Goal**: Compact, human-readable state: NAV, trend, recent PnL, open risk.

### 1) Text layout (default)

```
HEDGE — 8h Update (prod)
Time: 2025-08-22 10:00 SAST

NAV: 12,345.67 USDT  (Δ24h: +123.45, +1.01%)
Drawdown: 3.2%  |  Trend: ↑

Open Positions (top 5):
• ETHUSDT  SHORT 0.042  uPnL -0.29
• BTCUSDT  SHORT 0.001  uPnL +0.16
• SOLUSDT  LONG  1.000  uPnL -6.10  |  SHORT 1.000 uPnL +5.70 (hedged)

Recent PnL (last 5 trades):
• SOLUSDT +3.12
• ETHUSDT -0.58
• BTCUSDT +0.22
• …

Notes:
• Reserve: 0.013 BTC (not included in NAV)
• Screener: attempted=15, emitted=1 (last cycle)
```

### 2) JSON payload (for logging / future API)

```json
{
  "env": "prod",
  "timestamp": "2025-08-22T08:00:00Z",
  "nav": 12345.67,
  "nav_24h_delta_abs": 123.45,
  "nav_24h_delta_pct": 1.01,
  "drawdown_pct": 3.2,
  "trend": "up",         // one of: up, down, flat
  "reserve_btc": 0.013,
  "positions": [
    {"symbol":"ETHUSDT","side":"SHORT","qty":0.042,"uPnl":-0.289},
    {"symbol":"BTCUSDT","side":"SHORT","qty":0.001,"uPnl":0.157},
    {"symbol":"SOLUSDT","side":"LONG","qty":1.0,"uPnl":-6.099},
    {"symbol":"SOLUSDT","side":"SHORT","qty":-1.0,"uPnl":5.699}
  ],
  "recent_trades": [
    {"symbol":"SOLUSDT","pnl":3.12},
    {"symbol":"ETHUSDT","pnl":-0.58},
    {"symbol":"BTCUSDT","pnl":0.22}
  ],
  "screener": {"attempted": 15, "emitted": 1},
  "kpis": {"max_dd_pct": 9.0, "max_daily_loss_pct": 9.0}
}
```

### 3) Computation rules

* **NAV** = Firestore `nav` latest value; if series unavailable, compute `USDT + Σ unrealizedPnl` from `/fapi/v2/balance` + `positionRisk`.
* **Δ24h** = `NAV_now - NAV_24h_ago` (Firestore if available; otherwise compute from balance snapshots).
* **Trend**:

  * ↑ if `Δ24h_pct > +0.25%`
  * ↓ if `Δ24h_pct < −0.25%`
  * → otherwise
* **Positions summary**: group by `(symbol, positionSide)`; display up to top 5 by absolute uPnL.
* **Recent trades**: last 5 fills from `userTrades` or local `trade_log.json` if present.
* **Reserve**: always show **0.013 BTC** clearly; **do not** add to NAV.

### 4) Delivery

* Cron (as root):

  ```
  0 */8 * * * /root/hedge-fund/venv/bin/python /root/hedge-fund/telegram/heartbeat.py >> /var/log/hedge-telebot.log 2>&1
  ```
* The script should tolerate missing Firestore or network hiccups (send best-effort content; never crash).

---

If you want, I can also generate ready-to-use starter contents for:

* `config/pairs_universe.json`
* `config/strategy_config.json` (BTC breakout, ETH momentum, SOL fast lane)
* `execution/rules_sl_tp.py`
* `telegram/heartbeat.py`
* `dashboard/app.py`

Just say “generate the starter set” and I’ll drop them in a single pass.
