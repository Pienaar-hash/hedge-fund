# execution/exchange_utils.py
import json
import os
import requests
from binance.client import Client
import pandas as pd
import numpy as np

# === Config ===
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
TEST_CHAT_ID = os.getenv("TEST_CHAT_ID")

# === Binance Client with API keys from environment ===
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, testnet=TESTNET)

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print("⚠️ BINANCE_API_KEY / BINANCE_API_SECRET not set — private endpoints will fail.")

SYNCED_STATE_PATH = "synced_state.json"
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# === Fetch OHLCV Data ===
def fetch_ohlcv(client, symbol, interval="1m", limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    if not klines:
        raise ValueError(f"No klines returned for {symbol} {interval}")

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_vol', 'num_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['close', 'volume'], inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['momentum'] = df['log_return'].rolling(20).sum()
    df['vol'] = df['log_return'].rolling(20).std()
    df['zscore'] = df['momentum'] / df['vol']
    df['ema_fast'] = df['close'].ewm(span=10).mean()
    df['ema_slow'] = df['close'].ewm(span=50).mean()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError(f"Empty feature frame for {symbol} {interval}")

    return df

# === Telegram Sender ===
def send_telegram(message):
    chat_id = TEST_CHAT_ID if DRY_RUN and TEST_CHAT_ID else CHAT_ID
    if not BOT_TOKEN or not chat_id:
        print("❌ Telegram BOT_TOKEN or CHAT_ID not set.")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print(f"Telegram send failed: {response.text}")
    except Exception as e:
        print(f"Telegram error: {e}")

# === Portfolio Summary Sender ===
def send_portfolio_summary_with_pnl(state_map: dict, symbols: list | None = None, interval: str = "1m"):
    try:
        nav_path = "nav_log.json"
        if not os.path.exists(nav_path):
            send_telegram("⚠️ No NAV log available.")
            return

        with open(nav_path, "r") as f:
            logs = json.load(f)
        if not logs:
            send_telegram("⚠️ NAV log is empty.")
            return

        latest = logs[-1]
        timestamp = latest.get("timestamp", "—")
        equity = latest.get("equity", {})
        if isinstance(equity, (int, float)):
            equity = {"TOTAL": float(equity)}

        account_info = client.get_account()
        balances = {item['asset']: float(item.get('free', 0.0)) for item in account_info.get('balances', []) if float(item.get('free', 0.0)) > 0}

        if symbols is None:
            symbols = list(state_map.keys()) if state_map else []

        lines = [f"📊 *Portfolio Summary* — {timestamp}"]
        total_equity = 0.0
        per_symbol_equity = {}

        synced_state = {}
        for sym in symbols:
            asset = sym.replace("USDT", "")
            qty = float(balances.get(asset, 0.0))
            try:
                price = float(fetch_ohlcv(client, sym, interval=interval, limit=2).iloc[-1]['close'])
            except Exception:
                price = 0.0
            entry = float(state_map.get(sym, {}).get("entry", price or 0.0))
            pnl = (price - entry) * qty if qty and price and entry else 0.0
            value = qty * price if qty and price else 0.0

            if qty > 0.0000001 and price > 0:
                merged = {**state_map.get(sym, {}), "in_position": True, "qty": qty, "latest_price": price, "pnl": pnl}
                synced_state[sym] = merged
                per_symbol_equity[sym] = value

        if synced_state:
            with open(SYNCED_STATE_PATH, "w") as f:
                json.dump(synced_state, f, indent=2)

        merged_equity = {**equity, **per_symbol_equity}
        for sym, eq in merged_equity.items():
            total_equity += float(eq)
            lines.append(f"{sym}: ${float(eq):,.2f}")
        lines.append(f"\n🧾 *Total Equity*: ${total_equity:,.2f}")

        realized = latest.get("realized")
        unrealized = latest.get("unrealized")
        balance = latest.get("balance")
        extras = []
        if realized is not None:
            extras.append(f"Realized: ${float(realized):,.2f}")
        if unrealized is not None:
            extras.append(f"Unrealized: ${float(unrealized):,.2f}")
        if balance is not None:
            extras.append(f"USDT: ${float(balance):,.2f}")
        if extras:
            lines.append("💵 " + " | ".join(extras))

        send_telegram("\n".join(lines))
    except Exception as e:
        send_telegram(f"❌ Portfolio summary error: {e}")


# execution/executor_live.py
import os
import json
from datetime import datetime, timezone
from pathlib import Path

from execution.exchange_utils import (
    client,
    send_telegram,
    send_portfolio_summary_with_pnl,
)

# ---- Config ----
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
HAS_KEYS = bool(os.getenv("BINANCE_API_SECRET"))
NAV_LOG_PATH = Path("nav_log.json")
SYNCED_STATE_PATH = Path("synced_state.json")
PEAK_STATE_PATH = Path("peak_state.json")

# ---- Tiny JSON helpers ----
def _read_json(path: Path, default):
    try:
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    except Exception as e:
        print(f"WARN: failed to read {path}: {e}")
    return default


def _write_json(path: Path, obj) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"WARN: failed to write {path}: {e}")


# ---- Peak state (tiny read/write) ----
def load_peak_state():
    return _read_json(PEAK_STATE_PATH, {"peak_total_equity": 0.0, "timestamp": None})


def update_peak_state(total_equity: float):
    peak = load_peak_state()
    if total_equity > float(peak.get("peak_total_equity", 0.0)):
        peak["peak_total_equity"] = float(total_equity)
        peak["timestamp"] = datetime.now(timezone.utc).isoformat()
        _write_json(PEAK_STATE_PATH, peak)
        return True
    return False


# ---- Core NAV computation ----
def compute_nav(state_map: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    # Default shell entry
    entry = {
        "timestamp": now,
        "realized": 0.0,
        "unrealized": 0.0,
        "balance": 0.0,
        "equity": {},
    }

    if DRY_RUN or not HAS_KEYS:
        # Public-only / no keys path — we cannot call private endpoints
        # Keep equity empty; downstream summary will still run with any synced_state data
        return entry

    # With keys: call private endpoint safely
    try:
        account = client.get_account()
        balances = {
            b["asset"]: float(b.get("free", 0.0))
            for b in account.get("balances", [])
            if float(b.get("free", 0.0)) > 0
        }

        # Track USDT cash explicitly if available
        entry["balance"] = float(balances.get("USDT", 0.0))

        # Compute per-symbol equity from synced state if we have quantities stored there
        synced_state = _read_json(SYNCED_STATE_PATH, {})
        for sym, meta in synced_state.items():
            qty = float(meta.get("qty", 0.0))
            price = float(meta.get("latest_price", 0.0))
            if qty and price:
                entry["equity"][sym] = qty * price

        return entry
    except Exception as e:
        print(f"ERROR compute_nav: {e}")
        return entry


# ---- Main runner ----
def main():
    state_map = _read_json(SYNCED_STATE_PATH, {})  # use any prior synced state if present

    nav = compute_nav(state_map)

    # Append to nav_log.json
    nav_log = _read_json(NAV_LOG_PATH, [])
    nav_log.append(nav)
    _write_json(NAV_LOG_PATH, nav_log)

    # Peak tracker
    total_equity = float(sum(nav.get("equity", {}).values() or [0.0]))
    if update_peak_state(total_equity):
        print(f"🌟 New peak equity recorded: {total_equity:.2f}")

    # Telegram summary (safe in DRY_RUN thanks to exchange_utils)
    try:
        send_portfolio_summary_with_pnl(state_map)
    except Exception as e:
        print(f"WARN summary: {e}")


if __name__ == "__main__":
    main()
