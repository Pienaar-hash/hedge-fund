#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Mapping

from . import nav as nav_mod
from .risk_limits import RiskState, check_order
from .universe_resolver import symbol_tier, is_listed_on_futures
from .nav import is_nav_fresh
from execution.risk_loader import load_risk_config

try:
    from .exchange_utils import get_price, get_symbol_filters
except Exception:
    # Allow running without network: stub minimal behavior
    def get_price(symbol: str) -> float:
        return 0.0

    def get_symbol_filters(symbol: str) -> Dict[str, Any]:
        raise RuntimeError("exchange access unavailable")


WL_DEFAULT = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "LINKUSDT",
    "BNBUSDT",
    "SUIUSDT",
    "LTCUSDT",
    "WIFUSDT",
    "DOGEUSDT",
    "ARBUSDT",
    "OPUSDT",
    "AVAXUSDT",
    "APTUSDT",
    "INJUSDT",
    "RNDRUSDT",
    "SEIUSDT",
    "TONUSDT",
    "XRPUSDT",
    "ADAUSDT",
]


def _load_cfg() -> Dict[str, Any]:
    try:
        cfg = load_risk_config()
    except Exception:
        cfg = {}
    return cfg if isinstance(cfg, dict) else {"global": {}, "per_symbol": {}}


def evaluate_signal(signal: str, symbol: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fast heuristic to vet signals before execution."""

    def _to_float(value: Any) -> float | None:
        try:
            if value in (None, "", [], {}):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    reasons: List[str] = []
    base_conf = _to_float(payload.get("confidence")) or 0.6
    base_conf = max(0.0, min(1.0, base_conf))

    recent_fail = int(float(payload.get("recent_fail_streak", 0) or 0))
    fail_threshold = int(float(payload.get("fail_streak_limit", 3) or 3))
    if fail_threshold > 0 and recent_fail >= fail_threshold:
        reasons.append("recent_fail_streak")

    nav_threshold = (
        _to_float(payload.get("nav_threshold_s"))
        or _to_float(payload.get("nav_stale_threshold_s"))
        or 180.0
    )
    if nav_threshold and nav_threshold > 0 and not is_nav_fresh(nav_threshold):
        reasons.append("nav_stale")

    spread_bps = _to_float(payload.get("spread_bps"))
    spread_cap = _to_float(payload.get("max_spread_bps")) or 7.5
    if spread_bps is not None and spread_cap > 0 and spread_bps > spread_cap:
        reasons.append("wide_spread")

    funding_bps = _to_float(payload.get("funding_rate_bps"))
    funding_cap = _to_float(payload.get("max_funding_bps"))
    if funding_cap is None:
        funding_cap = 20.0
    if funding_bps is not None and abs(funding_bps) > abs(funding_cap):
        reasons.append("funding_window")

    age_ms = _to_float(payload.get("latency_ms")) or _to_float(payload.get("age_ms"))
    max_latency_ms = _to_float(payload.get("max_latency_ms")) or 15_000.0
    if age_ms is not None and age_ms > max_latency_ms:
        reasons.append("stale_signal")

    ok = len(reasons) == 0
    confidence = base_conf if ok else max(0.05, min(base_conf, 0.2))

    return {"ok": ok, "reasons": reasons, "confidence": float(confidence)}


def _nav_snapshot_fallback() -> float:
    try:
        snap = nav_mod.PortfolioSnapshot()
        nav_val = float(snap.current_nav_usd())
        if nav_val > 0:
            return nav_val
        confirmed = nav_mod.get_confirmed_nav()
        nav_val = float(confirmed.get("nav") or confirmed.get("nav_usd") or 0.0)
        if nav_val > 0:
            return nav_val
    except Exception:
        return 0.0
    return 0.0


def diagnose_symbols(env: str, testnet: bool, symbols: List[str]) -> int:
    cfg = _load_cfg()
    g = cfg.get("global") or {}
    wl = g.get("whitelist") or WL_DEFAULT
    wl_set = {str(x).upper() for x in wl}
    now = time.time()
    nav = _nav_snapshot_fallback()
    st = RiskState()

    print(f"[doctor] ENV={env} testnet={int(testnet)} nav={nav:.2f}")
    exit_code = 0
    for sym in symbols:
        s = str(sym).upper()
        if s not in wl_set:
            print(
                json.dumps(
                    {
                        "symbol": s,
                        "would_emit": False,
                        "would_block": True,
                        "reasons": ["not_whitelisted"],
                    }
                )
            )
            continue
        # Defaults
        exch_min_notional = 0.0
        price = 0.0
        listed = is_listed_on_futures(s)
        if listed:
            try:
                price = float(get_price(s))
                f = get_symbol_filters(s)
                exch_min_notional = float(
                    (f.get("MIN_NOTIONAL", {}) or {}).get("notional", 0) or 0.0
                )
            except Exception:
                listed = False

        # Strategy leverage and capital from strategy_config.json (fallbacks)
        cap = 10.0
        lev = float(g.get("max_leverage", 20) or 20)
        try:
            scfg = json.load(open("config/strategy_config.json"))
            arr = scfg.get("strategies", []) if isinstance(scfg, dict) else []
            for row in arr:
                if not isinstance(row, dict):
                    continue
                if str(row.get("symbol", "")).upper() != s:
                    continue
                params = row.get("params") if isinstance(row.get("params"), Mapping) else {}
                try:
                    cap = float((params or {}).get("capital_per_trade", cap) or cap)
                except Exception:
                    cap = cap
                try:
                    lev = float((params or {}).get("leverage", lev) or lev)
                except Exception:
                    lev = lev
                break
        except Exception:
            pass

        vetoes: List[str] = []
        if not listed:
            vetoes.append("not_listed")

        # Risk check (gross notional = cap)
        # compute current gross for portfolio/tier (doctor is stateless; pass 0)
        tier = symbol_tier(s) or "UNKNOWN"
        risk_veto, details = check_order(
            symbol=s,
            side="BUY",
            requested_notional=cap,
            price=price,
            nav=nav,
            open_qty=0.0,
            now=now,
            cfg=cfg,
            state=st,
            current_gross_notional=0.0,
            lev=lev,
            open_positions_count=0,
            tier_name=tier,
            current_tier_gross_notional=0.0,
        )
        reasons = list(details.get("reasons", [])) if isinstance(details, dict) else []
        vetoes.extend([r for r in reasons if r not in vetoes])
        would_emit = listed and (not risk_veto) and (
            cap >= max(exch_min_notional, float(g.get("min_notional_usdt", 0) or 0))
        )
        # Include budget info for readout
        tcfg = (g.get("tiers") or {}).get(tier, {}) if isinstance(g, dict) else {}
        out = {
            "symbol": s,
            "price": price,
            "cap": cap,
            "exch_min_notional": exch_min_notional,
            "lev": lev,
            "tier": tier,
            "tier_budget_pct": float(tcfg.get("per_symbol_nav_pct", 0.0) or 0.0),
            "veto": vetoes,
            "would_emit": bool(would_emit),
            "would_block": not bool(would_emit),
        }
        print(json.dumps(out))
        if not would_emit:
            exit_code = 0  # doctor itself should not hard-fail; keep 0
    return exit_code


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Signal doctor: diagnostics for whitelist symbols")
    p.add_argument("--env", default=os.getenv("ENV", "dev"))
    p.add_argument("--testnet", type=int, default=int(os.getenv("BINANCE_TESTNET", "0")))
    p.add_argument("--symbols", type=str, default=",".join(WL_DEFAULT))
    p.add_argument("--once", action="store_true", help="Run once and exit")
    args = p.parse_args(argv)
    if str(args.env).lower() == "prod":
        allow = os.getenv("ALLOW_PROD_WRITE", "0").strip().lower()
        if allow not in {"1", "true", "yes"}:
            raise RuntimeError("signal_doctor refuses to run with ENV=prod without ALLOW_PROD_WRITE=1")
    syms = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    rc = diagnose_symbols(args.env, bool(args.testnet), syms)
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
