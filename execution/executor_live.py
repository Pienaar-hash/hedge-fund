import os, json, time
from datetime import datetime, timezone
from typing import Tuple, Set, Dict, DefaultDict
from collections import defaultdict

from execution.exchange_utils import get_balances, get_price as _raw_get_price, place_market_order, get_positions
from execution.signal_screener import generate_signals_from_config
from utils.firestore_client import get_db
from execution.sync_state import sync_leaderboard, sync_nav, sync_positions
from execution.telegram_utils import send_telegram, send_trade_alert, send_drawdown_alert, should_send_summary
from execution.utils import load_json, save_json, log_trade

NAV_LOG = "nav_log.json"
PEAK_STATE = "peak_state.json"
STATE_FILE = "synced_state.json"
CFG_FILE = "config/strategy_config.json"

# --- Local shims & helpers (use our new exchange utils safely) ---
def get_price(symbol: str) -> float:
    """Shim around exchange_utils.get_price returning a float price."""
    try:
        r = _raw_get_price(symbol)
        if isinstance(r, dict):
            return float(r.get("price", 0.0))
        return float(r or 0.0)
    except Exception:
        return 0.0

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def execute_trade(symbol: str, side: str, capital: float, balances: dict,
                  desired_qty: float | None, min_notional_usdt: float) -> dict:
    """
    Place a market order using place_market_order.
    - If desired_qty is provided use it; else qty = capital / price.
    - Enforces min_notional_usdt.
    Returns {price, qty, order_id} or {error}.
    """
    try:
        side = side.upper()
        price = get_price(symbol)
        if price <= 0:
            return {"error": "price_unavailable"}
        qty = float(desired_qty) if desired_qty is not None else max(0.0, float(capital) / price)
        if qty * price < float(min_notional_usdt):
            return {"error": "below_min_notional"}
        res = place_market_order(symbol, side, qty)
        if not res or not res.get("ok"):
            return {"error": (res or {}).get("_error", "order_failed")}
        return {"price": price, "qty": qty, "order_id": res.get("order_id")}
    except Exception as e:
        return {"error": str(e)}

def load_config() -> dict:
    if not os.path.exists(CFG_FILE):
        return {"execution": {"poll_seconds": 60, "telegram_enabled": False},
                "alerts": {"dd_alert_pct": 0.1},
                "trade_defaults": {"sell_close_pct": 1.0, "min_notional_usdt": 10.0},
                "strategies": []}
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
            if isinstance(sym, str) and sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        for a in p.get("assets", []):
            sym = a.get("symbol", "")
            if isinstance(sym, str) and sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        for sym in p.get("pairs", []):
            if isinstance(sym, str) and sym.endswith("USDT"):
                allowed.add(sym.replace("USDT", ""))
        base = p.get("base")
        if isinstance(base, str) and base.endswith("USDT"):
            allowed.add(base.replace("USDT", ""))
    return allowed

def prune_empty_positions(positions: dict) -> None:
    for k in list(positions.keys()):
        try:
            q = float(positions[k].get("qty", 0.0))
            if q <= 0.0 and not positions[k].get("in_position", False):
                positions.pop(k, None)
        except Exception:
            pass

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
    realized = 0.0
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

def get_trade_knobs(config: dict, strategy_name: str) -> tuple[float, float]:
    defaults = config.get("trade_defaults", {})
    d_pct = float(defaults.get("sell_close_pct", 1.0))
    d_min = float(defaults.get("min_notional_usdt", 10.0))
    for s in config.get("strategies", []):
        if s.get("name") == strategy_name:
            p = s.get("params", {})
            return float(p.get("sell_close_pct", d_pct)), float(p.get("min_notional_usdt", d_min))
    return d_pct, d_min

def update_position_after_trade(positions: dict, symbol: str, side: str, price: float, qty: float, strategy_key: str) -> float:
    state = positions.get(symbol, {"qty": 0.0, "entry": 0.0, "in_position": False})
    q_old = float(state.get("qty", 0.0))
    e_old = float(state.get("entry", 0.0))
    realized = 0.0
    if side == "BUY":
        total_cost = e_old * q_old + price * qty
        q_new = q_old + qty
        e_new = total_cost / q_new if q_new > 0 else 0.0
        state.update({"qty": q_new, "entry": e_new, "in_position": q_new > 0})
    else:
        sell_qty = min(q_old, qty)
        realized = (price - e_old) * sell_qty
        q_new = max(0.0, q_old - sell_qty)
        state.update({"qty": q_new, "entry": (e_old if q_new > 0 else 0.0), "in_position": q_new > 0})
    state["latest_price"] = price
    state["strategy"] = strategy_key
    positions[symbol] = state
    return realized

def main():
    print("🚀 Executor Live (Phase 2)")
    config = load_config()
    tele_cfg = bool(config.get("execution", {}).get("telegram_enabled", False))
    os.environ["TELEGRAM_ENABLED"] = "1" if tele_cfg else "0"
    poll = int(config.get("execution", {}).get("poll_seconds", 60))
    dd_alert = float(config.get("alerts", {}).get("dd_alert_pct", 0.10))
    one_shot = os.getenv("ONE_SHOT", "0").lower() in ("1", "true", "yes")
    peak_state = load_json(PEAK_STATE) or {"portfolio": {"peak_equity": 0.0}, "assets": {}, "strategies": {}}
    peak_state.setdefault("assets", {})
    peak_state.setdefault("strategies", {})
    allowed_assets = build_asset_whitelist(config)

    # Firestore init
    ENV = os.environ.get("ENV", "dev")
    try:
        db = get_db()
    except Exception as e:
        print(f"❌ Firestore init failed: {e}")
        db = None
    # Trading mode toggle: Futures vs Spot
    use_futures = os.getenv("USE_FUTURES", "0").lower() in ("1", "true", "yes")

    while True:
        try:
            balances = get_balances()
            positions = load_json(STATE_FILE) or {}
            if use_futures:
                # Override/merge with live futures positions from the exchange
                live = get_positions() or []
                for p in live:
                    sym = p.get("symbol")
                    if not sym:
                        continue
                    qty = abs(float(p.get("qty", 0.0)))
                    entry = float(p.get("entry_price", 0.0))
                    latest = get_price(sym)
                    positions[sym] = {
                        "qty": qty,
                        "entry": entry,
                        "latest_price": latest,
                        "in_position": qty > 0,
                        "strategy": positions.get(sym, {}).get("strategy"),
                        "leverage": int(p.get("leverage", 1)),
                        "pnl": (latest - entry) * qty,
                    }

            refresh_positions_latest_prices(positions)
            prune_empty_positions(positions)
            signals = list(generate_signals_from_config(config))
            preview = signals[:2] + (["..."] if len(signals) > 2 else [])
            print(f"🔎 Signals: {'none' if not signals else preview}")
            realized_total = 0.0

            for sig in signals:
                strat_name = sig.get("strategy_name", "unknown")
                strategy_key = sig.get("strategy") or strat_name
                symbol = sig["symbol"]
                side = sig["signal"].upper()

                # Trade knobs per strategy (needed in all branches)
                sell_close_pct, min_notional = get_trade_knobs(config, strat_name)

                # --- Basic exit logic ---
                # If FLAT → close all. If SELL and we are long → close configured pct.
                if side == "FLAT":
                    pos_qty = float((positions.get(symbol) or {}).get("qty", 0.0))
                    if pos_qty > 0:
                        result = execute_trade(symbol=symbol, side="SELL", capital=0.0, balances=balances,
                                               desired_qty=pos_qty, min_notional_usdt=min_notional)
                        if "error" not in result:
                            px = float(result.get("price", 0.0)); qty = float(result.get("qty", 0.0))
                            realized = update_position_after_trade(positions, symbol, "SELL", px, qty, strategy_key)
                            realized_total += realized
                            log_trade({"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                                       "symbol": symbol, "side": "SELL", "price": px, "qty": qty,
                                       "order_id": result.get("order_id"), "strategy": strategy_key,
                                       "strategy_name": strat_name})
                    continue  # handled FLAT; skip buy/sell below

                if side == "SELL":
                    pos_qty = float((positions.get(symbol) or {}).get("qty", 0.0))
                    desired_qty = max(0.0, pos_qty * sell_close_pct)
                    if desired_qty <= 0.0:
                        continue
                    result = execute_trade(symbol=symbol, side=side, capital=0.0, balances=balances, desired_qty=desired_qty, min_notional_usdt=min_notional)
                else:
                    capital = get_capital_for_strategy(config, strat_name, float(balances.get("USDT", 0.0)))
                    if capital < min_notional:
                        continue
                    result = execute_trade(symbol=symbol, side=side, capital=capital, balances=balances, desired_qty=None, min_notional_usdt=min_notional)

                if "error" in result:
                    continue

                px = float(result.get("price", 0.0))
                qty = float(result.get("qty", 0.0))
                realized = update_position_after_trade(positions, symbol, side, px, qty, strategy_key)
                realized_total += realized
                log_trade({"timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "symbol": symbol, "side": side, "price": px, "qty": qty, "order_id": result.get("order_id"), "strategy": strategy_key, "strategy_name": strat_name})

            prune_empty_positions(positions)
            save_json(STATE_FILE, positions)
            peak = float(peak_state.get("portfolio", {}).get("peak_equity", 0.0))
            nav_entry, new_peak, dd = compute_nav_and_dd(balances, positions, peak, allowed_assets)
            nav_entry["realized"] = float(realized_total)
            append_nav(nav_entry)
            peak_state["portfolio"]["peak_equity"] = new_peak

            strat_values = compute_strategy_values(positions)
            for strat_key, value in strat_values.items():
                prev_peak = float(peak_state["strategies"].get(strat_key, 0.0))
                new_sp = max(prev_peak, value)
                peak_state["strategies"][strat_key] = new_sp

            save_json(PEAK_STATE, peak_state)

            # ---- Granular Firestore syncs ----
            try:
                if db is not None:
                    # NAV payload
                    nav_payload = {
                        "series": [{"ts": nav_entry["timestamp"], "equity": float(nav_entry.get("equity", 0.0))}],
                        "total_equity": float(nav_entry.get("equity", 0.0)),
                        "realized_pnl": float(nav_entry.get("realized", 0.0)),
                        "unrealized_pnl": float(nav_entry.get("unrealized", 0.0)),
                        "peak_equity": float(peak_state.get("portfolio", {}).get("peak_equity", 0.0)),
                        "drawdown": float(nav_entry.get("drawdown_pct", 0.0)),
                    }
                    sync_nav(db, nav_payload, ENV)

                    # Positions payload
                    pos_items = []
                    now_iso = datetime.now(timezone.utc).isoformat()
                    for sym, pos in positions.items():
                        qty = float(pos.get("qty", 0.0))
                        entry = float(pos.get("entry", 0.0))
                        last = float(pos.get("latest_price", 0.0)) or 0.0
                        side = "LONG" if qty > 0 else "FLAT"
                        pos_items.append({
                            "symbol": sym,
                            "side": side,
                            "qty": qty,
                            "entry_price": entry,
                            "mark_price": last,
                            "pnl": float(pos.get("pnl", (last - entry) * qty)),
                            "leverage": int(pos.get("leverage", 1)),
                            "notional": abs(qty * last),
                            "ts": now_iso,
                        })
                    sync_positions(db, {"items": pos_items}, ENV)

                    # Leaderboard payload (lightweight placeholder from strategy values)
                    lb_items = []
                    rank = 1
                    for strat_key, value in sorted(strat_values.items(), key=lambda x: x[1], reverse=True):
                        lb_items.append({
                            "strategy": str(strat_key),
                            "cagr": 0.0,
                            "sharpe": 0.0,
                            "mdd": 0.0,
                            "win_rate": 0.0,
                            "trades": 0,
                            "pnl": 0.0,
                            "equity": float(value),
                            "rank": rank,
                        })
                        rank += 1
                    if lb_items:
                        sync_leaderboard(db, {"items": lb_items}, ENV)

                    # Telegram summary (rate-limited)
                    if should_send_summary():
                        eq = float(nav_payload["total_equity"])
                        dd_pct = float(nav_payload["drawdown"]) * 100
                        send_telegram(f"📊 Sync OK — Equity {eq:,.2f} | DD {dd_pct:.2f}% | Positions {len(pos_items)}")
            except Exception as e:
                send_telegram(f"⚠️ Sync failed: {e}")

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
