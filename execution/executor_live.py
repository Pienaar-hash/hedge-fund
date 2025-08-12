import os, json, time
from datetime import datetime, timezone
from typing import Tuple, Set

from execution.exchange_utils import execute_trade, get_balances, get_price
from execution.signal_screener import generate_signals_from_config
from execution.sync_state import sync_portfolio_state
from execution.telegram_utils import send_trade_alert, send_drawdown_alert, send_telegram
from execution.utils import load_json, save_json, log_trade

NAV_LOG = "nav_log.json"
PEAK_STATE = "peak_state.json"
STATE_FILE = "synced_state.json"
CFG_FILE = "config/strategy_config.json"

def load_config() -> dict:
    if not os.path.exists(CFG_FILE):
        return {"execution": {"poll_seconds": 60}, "alerts": {"dd_alert_pct": 0.1}, "strategies": []}
    with open(CFG_FILE, "r") as f:
        return json.load(f)

def load_list_json(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

def append_nav(entry: dict):
    data = load_list_json(NAV_LOG)
    data.append(entry)
    with open(NAV_LOG, "w") as f:
        json.dump(data, f, indent=2)

def build_asset_whitelist(cfg: dict) -> Set[str]:
    """Allow only assets we actually trade (prevents TRY/ZAR/etc noise)."""
    allowed = {"USDT"}  # always keep cash leg
    for s in cfg.get("strategies", []):
        p = s.get("params", {})
        # momentum symbols like BTCUSDT
        for sym in p.get("symbols", []):
            if sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        # vol-target assets list
        for a in p.get("assets", []):
            sym = a.get("symbol", "")
            if sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        # relative value pairs
        for sym in p.get("pairs", []):
            if sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        base = p.get("base")
        if base and base.endswith("USDT"):
            allowed.add(base.replace("USDT", ""))
    return allowed

def compute_nav_and_dd(balances: dict, positions: dict, peak: float, allowed_assets: Set[str]) -> Tuple[dict, float, float]:
    usdt = float(balances.get("USDT", 0.0))
    equity = usdt

    # value only allowed assets
    for asset, qty in balances.items():
        if asset in ("USDT",) or asset not in allowed_assets:
            continue
        sym = f"{asset}USDT"
        px = get_price(sym)
        if px > 0:
            equity += float(qty) * float(px)

    # unrealized from tracked positions
    unreal = 0.0
    for sym, pos in positions.items():
        qty = float(pos.get("qty", 0.0))
        entry = float(pos.get("entry", 0.0))
        last = float(pos.get("latest_price", 0.0)) or get_price(sym)
        unreal += (last - entry) * qty

    realized = 0.0  # TODO: wire realized pnl from accounting
    new_peak = max(peak, equity) if peak else equity
    dd = 0.0 if new_peak == 0 else (equity - new_peak) / new_peak
    nav_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "realized": realized,
        "unrealized": unreal,
        "balance": usdt,
        "equity": equity,
        "drawdown_pct": dd
    }
    return nav_entry, new_peak, dd

def get_capital_for_strategy(config: dict, strategy_name: str, usdt_balance: float) -> float:
    fallback = max(10.0, min(usdt_balance * 0.01, 500.0))
    for s in config.get("strategies", []):
        if s.get("name") == strategy_name:
            capital = s.get("params", {}).get("capital_per_trade")
            try:
                return float(capital) if capital is not None else fallback
            except Exception:
                return fallback
    return fallback

def main():
    print("🚀 Executor Live (Phase 2)")
    config = load_config()
    poll = int(config.get("execution", {}).get("poll_seconds", 60))
    dd_alert = float(config.get("alerts", {}).get("dd_alert_pct", 0.10))
    one_shot = os.getenv("ONE_SHOT", "0").lower() in ("1", "true", "yes")

    peak_state = load_json(PEAK_STATE) or {"portfolio": {"peak_equity": 0.0}}
    allowed_assets = build_asset_whitelist(config)

    while True:
        try:
            balances = get_balances()
            positions = load_json(STATE_FILE) or {}

            signals = list(generate_signals_from_config(config))
            print(f"🔎 Signals: {'none' if not signals else signals[:2] + (['...'] if len(signals)>2 else [])}")

            # execute signals
            for sig in signals:
                strat_name = sig.get("strategy_name", "unknown")
                symbol = sig["symbol"]
                side = sig["signal"]
                capital = get_capital_for_strategy(config, strat_name, float(balances.get("USDT", 0.0)))

                base_asset = symbol.replace("USDT", "")
                if side == "SELL" and float(balances.get(base_asset, 0.0)) <= 0.0:
                    print(f"ℹ️ Skip SELL {symbol}: no holdings.")
                    continue

                result = execute_trade(symbol=symbol, side=side, capital=capital, balances=balances)
                if "error" in result:
                    print(f"❌ Execution error: {result['error']}")
                    continue

                trade_entry = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "side": side,
                    "price": result.get("price"),
                    "qty": result.get("qty"),
                    "order_id": result.get("order_id"),
                    "strategy": sig.get("strategy"),
                    "z_score": sig.get("z_score"),
                    "rsi": sig.get("rsi"),
                    "momentum": sig.get("momentum")
                }
                log_trade(trade_entry, path="logs/trade_log.json")

            # NAV + drawdown
            peak = float(peak_state.get("portfolio", {}).get("peak_equity", 0.0))
            nav_entry, new_peak, dd = compute_nav_and_dd(balances, positions, peak, allowed_assets)
            append_nav(nav_entry)
            peak_state["portfolio"]["peak_equity"] = new_peak
            save_json(PEAK_STATE, peak_state)

            # alerts
            if abs(dd) >= dd_alert:
                send_drawdown_alert("portfolio", dd, nav_entry["equity"])

            try:
                tlog = load_json("logs/trade_log.json")
                if tlog:
                    last_ts = sorted(tlog.keys())[-1]
                    last_trade = tlog[last_ts]
                    last_trade.update({
                        "realized": nav_entry["realized"],
                        "unrealized": nav_entry["unrealized"],
                        "equity": nav_entry["equity"],
                        "drawdown_pct": nav_entry["drawdown_pct"]
                    })
                    send_trade_alert(last_trade, silent=False)
            except Exception as e:
                print(f"⚠️ Trade alert enrichment skipped: {e}")

            # sync firebase (soft-fail if creds missing)
            sync_portfolio_state()

        except Exception as e:
            print(f"❌ Unexpected executor error: {e}")
            try:
                send_telegram(f"❌ Unexpected executor error: {e}")
            except Exception:
                pass

        if one_shot:
            break
        time.sleep(poll)

if __name__ == "__main__":
    main()

