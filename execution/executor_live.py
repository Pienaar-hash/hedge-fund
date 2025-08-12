import os, json, time
from datetime import datetime, timezone
from typing import Tuple, Set, Dict, DefaultDict
from collections import defaultdict

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
        return {"execution": {"poll_seconds": 60, "telegram_enabled": False}, "alerts": {"dd_alert_pct": 0.1}, "strategies": []}
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
    allowed = {"USDT"}
    for s in cfg.get("strategies", []):
        p = s.get("params", {})
        for sym in p.get("symbols", []):
            if sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        for a in p.get("assets", []):
            sym = a.get("symbol", "")
            if sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        for sym in p.get("pairs", []):
            if sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        base = p.get("base")
        if base and base.endswith("USDT"):
            allowed.add(base.replace("USDT", ""))
    return allowed

def refresh_positions_latest_prices(positions: Dict[str, dict]) -> None:
    for sym, pos in positions.items():
        try:
            last = get_price(sym)
            pos["latest_price"] = float(last) if last else 0.0
            qty = float(pos.get("qty", 0.0))
            entry = float(pos.get("entry", 0.0))
            pos["pnl"] = (pos["latest_price"] - entry) * qty
        except Exception:
            pass

def compute_nav_and_dd(balances: dict, positions: dict, peak: float, allowed_assets: Set[str]) -> Tuple[dict, float, float]:
    usdt = float(balances.get("USDT", 0.0))
    equity = usdt
    for asset, qty in balances.items():
        if asset in ("USDT",) or asset not in allowed_assets:
            continue
        sym = f"{asset}USDT"
        px = get_price(sym)
        if px > 0:
            equity += float(qty) * float(px)

    unreal = 0.0
    for sym, pos in positions.items():
        qty = float(pos.get("qty", 0.0))
        entry = float(pos.get("entry", 0.0))
        last = float(pos.get("latest_price", 0.0)) or get_price(sym)
        unreal += (last - entry) * qty

    realized = 0.0  # will be filled in main()
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

def compute_asset_dd(symbol: str, positions: dict, asset_peaks: dict) -> Tuple[float, float]:
    pos = positions.get(symbol, {})
    qty = float(pos.get("qty", 0.0))
    last = float(pos.get("latest_price", 0.0)) or get_price(symbol)
    value = qty * last
    peak = float(asset_peaks.get(symbol, 0.0))
    new_peak = max(peak, value)
    dd = 0.0 if new_peak == 0 else (value - new_peak) / new_peak
    return new_peak, dd

def compute_strategy_values(positions: dict) -> Dict[str, float]:
    """Aggregate current MTM values per strategy key (last-touch wins on symbol)."""
    values: DefaultDict[str, float] = defaultdict(float)
    for sym, pos in positions.items():
        strat_key = pos.get("strategy")
        if not strat_key:
            continue
        qty = float(pos.get("qty", 0.0))
        last = float(pos.get("latest_price", 0.0)) or get_price(sym)
        values[strat_key] += qty * last
    return dict(values)

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

def update_position_after_trade(positions: dict, symbol: str, side: str, price: float, qty: float, strategy_key: str) -> float:
    """Update local position book, tag with strategy_key, and return realized PnL."""
    state = positions.get(symbol, {"qty": 0.0, "entry": 0.0, "in_position": False})
    q_old = float(state.get("qty", 0.0))
    e_old = float(state.get("entry", 0.0))
    realized = 0.0

    if side == "BUY":
        total_cost = e_old * q_old + price * qty
        q_new = q_old + qty
        e_new = total_cost / q_new if q_new > 0 else 0.0
        state.update({"qty": q_new, "entry": e_new, "in_position": q_new > 0})
    else:  # SELL
        sell_qty = min(q_old, qty)
        realized = (price - e_old) * sell_qty
        q_new = max(0.0, q_old - sell_qty)
        state.update({"qty": q_new, "entry": (e_old if q_new > 0 else 0.0), "in_position": q_new > 0})

    state["latest_price"] = price
    state["strategy"] = strategy_key  # last-touch ownership
    positions[symbol] = state
    return realized

def main():
    print("🚀 Executor Live (Phase 2)")
    config = load_config()
    os.environ["TELEGRAM_ENABLED"] = "1" if bool(config.get("execution", {}).get("telegram_enabled", False)) else "0"

    poll = int(config.get("execution", {}).get("poll_seconds", 60))
    dd_alert = float(config.get("alerts", {}).get("dd_alert_pct", 0.10))
    one_shot = os.getenv("ONE_SHOT", "0").lower() in ("1", "true", "yes")

    peak_state = load_json(PEAK_STATE) or {"portfolio": {"peak_equity": 0.0}, "assets": {}, "strategies": {}}
    peak_state.setdefault("assets", {})
    peak_state.setdefault("strategies", {})

    allowed_assets = build_asset_whitelist(config)

    while True:
        try:
            balances = get_balances()
            positions = load_json(STATE_FILE) or {}

            # Refresh MTM for open positions
            refresh_positions_latest_prices(positions)

            signals = list(generate_signals_from_config(config))
            print(f"🔎 Signals: {'none' if not signals else signals[:2] + (['...'] if len(signals)>2 else [])}")

            realized_total = 0.0
            # Execute signals
            for sig in signals:
                strat_name = sig.get("strategy_name", "unknown")          # e.g., 'momentum'
                strategy_key = sig.get("strategy") or strat_name          # e.g., 'momentum_btcusdt'
                symbol = sig["symbol"]
                side = sig["signal"]
                capital = get_capital_for_strategy(config, strat_name, float(balances.get("USDT", 0.0)))

                base_asset = symbol.replace("USDT", "")
                if side == "SELL" and float(positions.get(symbol, {}).get("qty", 0.0)) <= 0.0 and float(balances.get(base_asset, 0.0)) <= 0.0:
                    print(f"ℹ️ Skip SELL {symbol}: no holdings.")
                    continue

                result = execute_trade(symbol=symbol, side=side, capital=capital, balances=balances)
                if "error" in result:
                    print(f"❌ Execution error: {result['error']}")
                    continue

                # Update positions & compute realized PnL
                px = float(result.get("price", 0.0))
                qty = float(result.get("qty", 0.0))
                realized = update_position_after_trade(positions, symbol, side, px, qty, strategy_key)
                realized_total += realized

                # Compute *strategy* drawdown snapshot at trade time (before peak update)
                # Current value for this symbol only (approx for the strategy)
                sym_qty = float(positions.get(symbol, {}).get("qty", 0.0))
                sym_val = sym_qty * px
                strat_peak_prev = float(peak_state["strategies"].get(strategy_key, 0.0))
                strat_dd_now = 0.0 if max(strat_peak_prev, sym_val) == 0 else (sym_val - max(strat_peak_prev, sym_val)) / max(strat_peak_prev, sym_val)

                trade_entry = {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "symbol": symbol,
                    "side": side,
                    "price": px,
                    "qty": qty,
                    "order_id": result.get("order_id"),
                    "strategy": strategy_key,
                    "strategy_name": strat_name,
                    "strategy_drawdown_pct": strat_dd_now,
                    "z_score": float(sig.get("z_score") or 0.0),
                    "rsi": float(sig.get("rsi") or 0.0),
                    "momentum": float(sig.get("momentum") or 0.0)
                }
                log_trade(trade_entry, path="logs/trade_log.json")

            # Persist positions
            save_json(STATE_FILE, positions)

            # NAV + portfolio DD
            peak = float(peak_state.get("portfolio", {}).get("peak_equity", 0.0))
            nav_entry, new_peak, dd = compute_nav_and_dd(balances, positions, peak, allowed_assets)
            nav_entry["realized"] = float(realized_total)
            append_nav(nav_entry)
            peak_state["portfolio"]["peak_equity"] = new_peak

            # Per-asset peaks & alerts
            for sym in positions.keys():
                new_sym_peak, sym_dd = compute_asset_dd(sym, positions, peak_state["assets"])
                prev = float(peak_state["assets"].get(sym, 0.0))
                peak_state["assets"][sym] = max(prev, new_sym_peak)
                if abs(sym_dd) >= dd_alert:
                    try:
                        send_drawdown_alert(sym, sym_dd, new_sym_peak)
                    except Exception:
                        pass

            # Per-strategy peaks & alerts (aggregate across symbols tagged to each strategy)
            strat_values = compute_strategy_values(positions)
            for strat_key, value in strat_values.items():
                prev_peak = float(peak_state["strategies"].get(strat_key, 0.0))
                new_peak = max(prev_peak, value)
                peak_state["strategies"][strat_key] = new_peak
                strat_dd = 0.0 if new_peak == 0 else (value - new_peak) / new_peak
                if abs(strat_dd) >= dd_alert:
                    try:
                        send_drawdown_alert(strat_key, strat_dd, value)
                    except Exception:
                        pass

            save_json(PEAK_STATE, peak_state)

            # Send last-trade alert w/ NAV & DD context (only if any realized)
            try:
                tlog = load_json("logs/trade_log.json")
                if tlog and realized_total != 0.0:
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

            # Sync Firebase (soft-fail ok)
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
