from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from execution.risk_limits import RiskGate
from execution.utils import load_json
from execution.exchange_utils import get_balances, get_price


def _read_strategy_cfg() -> Dict[str, Any]:
    cfg = load_json("config/strategy_config.json")
    return cfg if isinstance(cfg, dict) else {}


def _read_risk_cfg() -> Dict[str, Any]:
    cfg = load_json("config/risk_limits.json")
    return cfg if isinstance(cfg, dict) else {}


def get_nav_snapshot(nav_path: str | None = None) -> Dict[str, float]:
    """
    Return a coarse NAV snapshot sourced from logs/nav.jsonl when available.
    Falls back to strategy config based NAV calculations; never raises.
    """
    default_path = os.getenv("NAV_LOG_PATH", "logs/nav.jsonl")
    path = Path(nav_path or default_path)
    now_ts = time.time()
    result = {"nav": 0.0, "equity": 0.0, "ts": float(now_ts)}

    try:
        if path.exists():
            lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            if lines:
                payload = json.loads(lines[-1])
                nav_val = float(payload.get("nav") or payload.get("wallet", 0.0) or 0.0)
                equity_val = float(payload.get("equity") or nav_val)
                ts_val = float(payload.get("t") or payload.get("ts") or payload.get("timestamp") or now_ts)
                result.update({"nav": nav_val, "equity": equity_val, "ts": ts_val})
                return result
    except Exception:
        pass

    try:
        cfg = _read_strategy_cfg()
        gate = RiskGate(cfg)
        nav_val = float(gate._portfolio_nav())  # type: ignore[attr-defined]
        if nav_val > 0:
            result.update({"nav": nav_val, "equity": nav_val})
    except Exception:
        pass
    return result


def get_caps() -> Dict[str, float]:
    """
    Return key risk caps in a lightweight dict for dashboard display.
    """
    caps = {
        "max_trade_nav_pct": 0.0,
        "max_gross_exposure_pct": 0.0,
        "max_symbol_exposure_pct": 0.0,
        "min_notional": 0.0,
    }
    try:
        cfg = _read_risk_cfg()
        gate = RiskGate(cfg)
        sizing = gate.sizing
        caps["max_trade_nav_pct"] = float(sizing.get("max_trade_nav_pct") or 0.0)
        gross_pct = sizing.get("max_gross_exposure_pct") or sizing.get("max_portfolio_gross_nav_pct")
        caps["max_gross_exposure_pct"] = float(gross_pct or 0.0)
        caps["max_symbol_exposure_pct"] = float(sizing.get("max_symbol_exposure_pct") or 0.0)
        caps["min_notional"] = float(getattr(gate, "min_notional", 0.0) or 0.0)
    except Exception:
        pass
    return caps


def get_veto_counts(log_dir: str = "logs", max_lines: int = 100) -> Dict[str, int]:
    """
    Aggregate veto reasons from recent veto_exec*.json logs.
    """
    counts: Counter[str] = Counter()
    try:
        base = Path(log_dir)
        if not base.exists():
            return {}
        files = sorted(base.glob("veto_exec_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        processed = 0
        for file_path in files:
            try:
                lines = file_path.read_text().splitlines()
            except Exception:
                continue
            for line in reversed(lines[-max_lines:]):
                if processed >= max_lines:
                    break
                processed += 1
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                reasons = payload.get("reasons") or payload.get("veto")
                if not reasons:
                    continue
                if isinstance(reasons, str):
                    counts[reasons] += 1
                elif isinstance(reasons, (list, tuple, set)):
                    for reason in reasons:
                        if reason:
                            counts[str(reason)] += 1
                else:
                    counts[str(reasons)] += 1
            if processed >= max_lines:
                break
    except Exception:
        return {}
    return dict(counts)


def _extract_balance(balances: Any, asset: str) -> float:
    target = asset.upper()
    try:
        if isinstance(balances, dict):
            value = balances.get(target)
            if isinstance(value, (int, float, str)):
                return float(value or 0.0)
            if isinstance(value, dict):
                for key in ("free", "balance", "walletBalance", target):
                    if key in value:
                        return float(value.get(key) or 0.0)
        if isinstance(balances, list):
            for entry in balances:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("asset", "")).upper() == target:
                    for key in ("balance", "free", "walletBalance", "total"):
                        if key in entry and entry[key] not in (None, ""):
                            return float(entry[key])
        if hasattr(balances, "get"):
            value = balances.get(target)  # type: ignore[attr-defined]
            if value:
                return float(value)
    except Exception:
        return 0.0
    return 0.0


def _safe_price(symbol: str) -> float:
    try:
        return float(get_price(symbol) or 0.0)
    except Exception:
        return 0.0


def get_treasury() -> Dict[str, Any]:
    """
    Return BTC/XAUT balances with USD valuations for dashboard display.
    """
    try:
        balances = get_balances()
    except Exception:
        balances = {}

    assets: List[Dict[str, float | str]] = []
    total_usd = 0.0

    for asset in ("BTC", "XAUT"):
        balance = _extract_balance(balances, asset)
        price = _safe_price(f"{asset}USDT")
        usd_value = balance * price
        total_usd += usd_value
        assets.append(
            {
                "asset": asset,
                "balance": float(balance),
                "price": float(price),
                "usd": float(usd_value),
            }
        )

    return {"assets": assets, "total_usd": float(total_usd)}


__all__ = ["get_nav_snapshot", "get_caps", "get_veto_counts", "get_treasury"]
