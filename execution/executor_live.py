# execution/executor_live.py — Sprint Phase 2
# Continuous runner, config-driven capital, NAV + drawdown, Telegram alerts
# Uses signal_screener.screen_strategy for single source of truth (per-asset overrides)

import os
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List

# Local imports
from execution.exchange_utils import execute_trade, get_balances, get_price
from execution.signal_screener import screen_strategy
from execution.sync_state import sync_portfolio_state
from execution.telegram_utils import (
    send_nav_summary,
    send_dd_breach,
    send_trade_alert,
    send_sync_error,
    send_executor_error,
    send_telegram,
)
from execution.utils import load_json, save_json, log_trade

CONFIG_PATH = os.getenv("STRATEGY_CONFIG_PATH", "config/strategy_config.json")
NAV_LOG_PATH = os.getenv("NAV_LOG_PATH", "nav_log.json")
STATE_PATH = os.getenv("STATE_PATH", "synced_state.json")
PEAK_PATH = os.getenv("PEAK_PATH", "peak_state.json")

POLL_SECONDS = int(os.getenv("EXEC_POLL_SECONDS", "60"))

# ---------- Helpers ----------

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def ensure_json_file(path: str, default):
    if not os.path.exists(path):
        save_json(path, default)
    return load_json(path)


def append_nav_log(entry: Dict[str, Any]):
    data: List[Dict[str, Any]] = []
    if os.path.exists(NAV_LOG_PATH):
        try:
            with open(NAV_LOG_PATH, "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception:
            data = []
    data.append(entry)
    with open(NAV_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def compute_equity_from_balances(balances: Dict[str, float]) -> float:
    equity = 0.0
    for asset, qty in balances.items():
        if asset.upper() == "USDT":
            equity += float(qty)
        else:
            symbol = f"{asset.upper()}USDT"
            px = get_price(symbol)
            equity += float(qty) * float(px)
    return equity


def compute_nav() -> Dict[str, Any]:
    balances = get_balances()
    usdt = float(balances.get("USDT", 0.0))
    equity = compute_equity_from_balances(balances)
    unrealized = equity - usdt
    realized = 0.0  # placeholder for closed PnL when available
    return {
        "timestamp": utcnow_iso(),
        "realized": realized,
        "unrealized": unrealized,
        "balance": usdt,
        "equity": equity,
    }


def update_peak_and_drawdown(equity: float) -> Dict[str, Any]:
    peak_state = ensure_json_file(PEAK_PATH, {"peak_equity": equity})
    peak_equity = float(peak_state.get("peak_equity", equity))
    if equity > peak_equity:
        peak_equity = equity
        peak_state["peak_equity"] = peak_equity
        save_json(PEAK_PATH, peak_state)
    drawdown = 0.0 if peak_equity == 0 else (equity - peak_equity) / peak_equity
    return {"peak_equity": peak_equity, "drawdown": drawdown}


def determine_capital_for_trade(strategy_params: Dict[str, Any], equity: float) -> float:
    # Priority: explicit capital_per_trade (abs USDT) → percent_of_equity → env → default 500
    if isinstance(strategy_params.get("capital_per_trade"), (int, float)):
        return float(strategy_params["capital_per_trade"])  # absolute USDT
    pct = strategy_params.get("percent_of_equity")
    if isinstance(pct, (int, float)) and pct > 0:
        return max(10.0, float(equity) * float(pct))
    env_cap = os.getenv("CAPITAL_PER_TRADE_USDT")
    if env_cap:
        try:
            return max(10.0, float(env_cap))
        except Exception:
            pass
    return 500.0

# ---------- Main loop ----------

def run_once():
    cfg = load_config()

    # Compute NAV first (for capital sizing + drawdown checks)
    nav = compute_nav()
    peak = update_peak_and_drawdown(nav["equity"])
    nav.update(peak)
    append_nav_log(nav)

    # Always send compact NAV summary (silenced by bot settings if needed)
    try:
        send_nav_summary(nav, silent=True)
    except Exception:
        pass

    # Drawdown breach alert
    dd_threshold = float(os.getenv("DD_ALERT_THRESHOLD", "0.15"))  # 15%
    if nav["drawdown"] <= -dd_threshold:
        try:
            send_dd_breach(nav["drawdown"], nav["equity"])
        except Exception:
            pass

    # Iterate strategies and use screen_strategy (single source of truth for thresholds/overrides)
    strategies = cfg.get("strategies", [])
    for strat in strategies:
        name = strat.get("name", "?")
        params = strat.get("params", {})

        # Per-strategy capital sizing
        capital_usdt = determine_capital_for_trade(params, nav["equity"])

        # Screen all symbols for this strategy via signal_screener
        try:
            signals = screen_strategy(strat)  # returns list of dicts with keys incl. 'signal'
        except Exception as e:
            # If screening fails for a strategy, continue with others
            try:
                send_executor_error(f"screen_strategy failed for {name}: {e}")
            except Exception:
                pass
            continue

        for sig in signals or []:
            if not sig or sig.get("signal") is None:
                continue

            symbol = sig.get("symbol", "BTCUSDT")
            side = sig.get("signal")

            # Fetch balances and execute
            balances = get_balances()
            result = execute_trade(symbol=symbol, side=side, capital=capital_usdt, balances=balances)

            # Build trade entry for logs/alerts
            trade_entry = {
                "timestamp": utcnow_iso(),
                "strategy": sig.get("strategy", name),
                "symbol": symbol,
                "side": side,
                "price": sig.get("price"),
                "z_score": sig.get("z_score"),
                "rsi": sig.get("rsi"),
                "momentum": sig.get("momentum"),
                "timeframe": sig.get("timeframe"),
                "capital": capital_usdt,
                "drawdown": nav["drawdown"],
                "peak_equity": nav["peak_equity"],
                **({k: v for k, v in (result or {}).items()}),
            }

            # Persist trade log
            try:
                log_trade(trade_entry, path="logs/trade_log.json")
            except Exception:
                print("[WARN] Failed to write trade_log.json — ensure logs/ exists.")

            # Human-friendly alert + compact trade alert
            msg = (
                f"🚀 Trade Executed
"
                f"🧠 Strategy: {trade_entry['strategy']}
"
                f"📈 {symbol} | {side}
"
                f"⏱ TF: {trade_entry.get('timeframe','—')}
"
                f"💰 Capital: ${capital_usdt:,.2f}
"
                f"💵 Price: {trade_entry.get('price', 0):.2f} USDT
"
                f"🧮 Qty: {trade_entry.get('qty', '—')}
"
                f"📊 z:{trade_entry.get('z_score')} rsi:{trade_entry.get('rsi')} mom:{trade_entry.get('momentum')}
"
                f"💼 Equity: ${nav['equity']:,.2f} | Peak: ${nav['peak_equity']:,.2f} | DD: {nav['drawdown']*100:.2f}%"
            )
            try:
                send_telegram(msg)
            except Exception:
                pass
            try:
                send_trade_alert(trade_entry, nav, silent=True)
            except Exception:
                pass

    # Sync portfolio state at end of cycle
    try:
        sync_portfolio_state()
    except Exception as e:
        try:
            send_sync_error(str(e))
        except Exception:
            pass


def main():
    print("🚀 Executor Live — continuous runner")
    while True:
        try:
            run_once()
        except Exception as e:
            print(f"[ERROR] {e}")
            try:
                send_executor_error(str(e))
            except Exception:
                pass
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
