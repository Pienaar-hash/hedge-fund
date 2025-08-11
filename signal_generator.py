# execution/signal_generator.py
import time
from datetime import datetime, timezone
from execution.exchange_utils import fetch_ohlcv, send_telegram, client

# ==========================
# CONFIG (4h conservative)
# ==========================
CONFIG = {
    "interval": "4h",
    "z_buy": 1.25,
    "z_sell": -1.25,
    "z_flip_buffer": 0.15,      # hysteresis buffer before reversing
    "vol_mult": 1.25,           # vol > vol_sma20 * vol_mult
    "rsi_buy_min": 48, "rsi_buy_max": 68,
    "rsi_sell_min": 32, "rsi_sell_max": 52,
    "cooldown_bars": 1,         # 1 x 4h bar after any trade signal
    "symbols": ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","LTCUSDT"],
}

STATE_FILE = "signal_state.json"

# ==========================
# Helpers: state + RSI
# ==========================
def load_state():
    import json, os
    return json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}

def save_state(state):
    import json
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================
# Core decision logic
# ==========================
def decide_signal(row, df, cfg, last_sig=None, bars_since=None):
    z = row["zscore"]
    trend_up = row["ema_fast"] > row["ema_slow"]
    mom = row["momentum"]
    vol = row["volume"]
    rsi = row["rsi"]

    vol_ok = vol > (df["volume"].rolling(20).mean().iloc[-1] * cfg["vol_mult"]) / cfg["vol_mult"]
    # (equivalent to vol > SMA20 * 1.25; written this way to avoid chained float issues below)
    vol_ok = vol > df["volume"].rolling(20).mean().iloc[-1] * cfg["vol_mult"]

    mom_ok = (mom > 0) if trend_up else (mom < 0)

    buy_cond  = (z > cfg["z_buy"])  and trend_up and vol_ok and mom_ok and (cfg["rsi_buy_min"]  <= rsi <= cfg["rsi_buy_max"]) 
    sell_cond = (z < cfg["z_sell"]) and (not trend_up) and vol_ok and mom_ok and (cfg["rsi_sell_min"] <= rsi <= cfg["rsi_sell_max"]) 

    # cooldown
    if bars_since is not None and bars_since < cfg["cooldown_bars"]:
        return "HOLD"

    # hysteresis: make reversals harder by adding buffer
    if last_sig == "BUY" and (z > cfg["z_sell"] + cfg["z_flip_buffer"]):
        sell_cond = False
    if last_sig == "SELL" and (z < cfg["z_buy"] - cfg["z_flip_buffer"]):
        buy_cond = False

    if buy_cond:
        return "BUY"
    if sell_cond:
        return "SELL"
    return "HOLD"

# ==========================
# Signal generation per symbol
# ==========================
def generate_signal(symbol):
    df = fetch_ohlcv(client, symbol, interval=CONFIG["interval"], limit=100)
    # compute indicators (RSI once)
    rsi_series = compute_rsi(df["close"], window=14)
    df = df.assign(rsi=rsi_series)
    latest = df.iloc[-1]

    # state for cooldown / hysteresis
    state = load_state()
    st_key = f"{symbol}_state"
    last = state.get(st_key, {"last_signal": None, "last_bar": None})
    bars_since = None
    if last.get("last_bar") is not None:
        try:
            bars_since = max(0, len(df) - 1 - int(last["last_bar"]))
        except Exception:
            bars_since = None

    sig = decide_signal(latest, df, CONFIG, last_sig=last.get("last_signal"), bars_since=bars_since)
    # persist
    state[st_key] = {"last_signal": sig, "last_bar": len(df) - 1}
    save_state(state)

    zscore = latest["zscore"]
    trend = "UP" if latest["ema_fast"] > latest["ema_slow"] else "DOWN"
    momentum = latest["momentum"]
    volume = latest["volume"]
    rsi = latest["rsi"]

    return {
        "timestamp": latest.name.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "close": latest["close"],
        "zscore": float(zscore),
        "trend": trend,
        "momentum": float(momentum),
        "volume": float(volume),
        "rsi": float(rsi),
        "signal": sig,
    }

# ==========================
# Main loop (4h cadence)
# ==========================
def main_loop():
    while True:
        now = datetime.now(timezone.utc)
        print(f"\n📡 Signal Generator Tick @ {now.isoformat()} ")
        messages = []
        for symbol in CONFIG["symbols"]:
            try:
                s = generate_signal(symbol)
                formatted = (
                    f"🕒 {s['timestamp']} | {symbol} | Close: {s['close']:.2f} | "
                    f"Mom: {s['momentum']:+.4f} | Vol: {s['volume']:.0f} | "
                    f"Z: {s['zscore']:+.2f} | RSI: {s['rsi']:.1f} | Trend: {s['trend']} | "
                    f"Signal: {'✅ ' + s['signal'] if s['signal'] in ['BUY','SELL'] else '—'}"
                )
                messages.append(formatted)
                print(formatted)
            except Exception as e:
                print(f"❌ Error generating signal for {symbol}: {e}")
        if messages:
            send_telegram("\n".join(messages))
        time.sleep(60 * 60 * 4)  # every 4 hours

if __name__ == "__main__":
    main_loop()
