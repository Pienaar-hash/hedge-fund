from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

from execution.risk_limits import RiskGate
from execution.utils import load_json
from execution.exchange_utils import get_price

STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD"}


def _try_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


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


def _safe_price(symbol: str) -> float:
    try:
        return float(get_price(symbol) or 0.0)
    except Exception:
        return 0.0


def _to_float(value: Any) -> float:
    try:
        if value in (None, "", "null"):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _format_asset_entry(asset: str, balance: float, price: float, usd_value: float) -> Dict[str, float | str]:
    price = float(price)
    usd_value = float(usd_value)
    return {
        "asset": asset,
        "balance": float(balance),
        "price": price,
        "price_usdt": price,
        "usd": usd_value,
        "usd_value": usd_value,
    }


def _normalize_treasury_payload(payload: Dict[str, Any], source: str) -> Dict[str, Any] | None:
    assets_acc: Dict[str, Dict[str, float]] = {}

    def ingest_node(node: Any, hint: str | None = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = str(key).lower()
                if key_lower in {"total_usd", "total_treasury_usdt", "updated_at", "ts"}:
                    continue
                if key_lower in {"assets", "treasury"}:
                    ingest_node(value)
                    continue
                ingest_entry(str(key), value)
        elif isinstance(node, list):
            for entry in node:
                ingest_entry(hint, entry)

    def ingest_entry(hint: str | None, entry: Any) -> None:
        asset_name = ""
        qty = 0.0
        price = 0.0
        usd_value = 0.0

        if isinstance(entry, dict):
            candidate = entry.get("asset") or entry.get("Asset") or entry.get("symbol") or entry.get("code")
            asset_name = str(hint or candidate or "").upper()
            qty = _to_float(entry.get("qty") or entry.get("Units") or entry.get("units") or entry.get("balance") or entry.get("amount"))
            price = _to_float(entry.get("px") or entry.get("price") or entry.get("price_usdt"))
            usd_value = _to_float(
                entry.get("val_usdt")
                or entry.get("USD Value")
                or entry.get("usd_value")
                or entry.get("value_usd")
                or entry.get("usd")
            )
        else:
            asset_name = str(hint or "").upper()
            qty = _to_float(entry)

        if not asset_name or asset_name in {"TOTAL_USD", "TOTAL_TREASURY_USDT"}:
            return

        acc = assets_acc.setdefault(asset_name, {"balance": 0.0, "price": 0.0, "usd_value": 0.0})
        if qty:
            acc["balance"] = float(qty)
        if price:
            acc["price"] = float(price)
        if usd_value:
            acc["usd_value"] = float(usd_value)

    candidates: List[Any] = []
    if isinstance(payload.get("assets"), (list, dict)):
        candidates.append(payload["assets"])
    if isinstance(payload.get("treasury"), dict):
        treas = payload["treasury"]
        candidates.append(treas)
        if isinstance(treas.get("assets"), (list, dict)):
            candidates.append(treas["assets"])
    if isinstance(payload.get("breakdown"), dict):
        breakdown = payload["breakdown"]
        treas_breakdown = breakdown.get("treasury")
        if isinstance(treas_breakdown, (list, dict)):
            candidates.append(treas_breakdown)
        if isinstance(treas_breakdown, dict) and isinstance(treas_breakdown.get("assets"), (list, dict)):
            candidates.append(treas_breakdown["assets"])

    for node in candidates:
        ingest_node(node)

    if not assets_acc:
        return None

    assets: List[Dict[str, Any]] = []
    total_payload = _to_float(
        payload.get("total_usd")
        or payload.get("treasury_usdt")
        or payload.get("total_treasury_usdt")
        or payload.get("treasury_total")
    )

    total_usd = 0.0
    for asset, info in assets_acc.items():
        balance = float(info.get("balance") or 0.0)
        price = float(info.get("price") or 0.0)
        usd_value = float(info.get("usd_value") or 0.0)

        if balance and usd_value and price <= 0.0:
            price = usd_value / balance if balance else 0.0

        if price <= 0.0:
            if asset in STABLES:
                price = 1.0
            else:
                price = _safe_price(f"{asset}USDT")

        if usd_value <= 0.0:
            usd_value = balance * price if balance else 0.0

        assets.append(_format_asset_entry(asset, balance, price, usd_value))
        total_usd += usd_value

    total = total_payload if total_payload > 0 else total_usd
    result: Dict[str, Any] = {
        "assets": assets,
        "total_usd": float(total),
        "source": source,
    }
    timestamp = payload.get("updated_at") or payload.get("ts")
    if timestamp:
        result["updated_at"] = timestamp
    return result


def _read_treasury_sources(root: str = ".") -> List[Dict[str, Any]]:
    """Return treasury source candidates with freshness metadata."""
    files = [
        ("logs/treasury.json", Path(root) / "logs" / "treasury.json"),
    ]
    sources: List[Dict[str, Any]] = []
    now = time.time()

    for label, path in files:
        try:
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        updated_raw = payload.get("updated_at") or payload.get("ts")
        updated_ts = None
        freshness = None
        if isinstance(updated_raw, str):
            try:
                dt = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
                updated_ts = dt.timestamp()
            except Exception:
                updated_ts = None
        if updated_ts is not None:
            freshness = now - updated_ts
        sources.append(
            {
                "label": label,
                "payload": payload,
                "updated_at": updated_ts,
                "freshness_seconds": freshness,
            }
        )

    try:
        cfg_path = Path(root) / "config" / "reserves.json"
        if cfg_path.exists():
            payload = json.loads(cfg_path.read_text())
            if isinstance(payload, dict):
                sources.append(
                    {
                        "label": "config/reserves.json",
                        "payload": payload,
                        "updated_at": None,
                        "freshness_seconds": None,
                    }
                )
    except Exception:
        pass
    return sources


def _select_canonical_treasury(
    sources: List[Dict[str, Any]], freshness_limit: float = 600.0
) -> tuple[List[Dict[str, Any]], float, str, List[str]]:
    """Select canonical treasury data from available sources."""
    sources_seen: List[str] = []
    priority = ["logs/treasury.json", "config/reserves.json"]
    chosen_payload: Dict[str, Any] = {}
    source_used = "config/reserves.json"

    for label in priority:
        for entry in sources:
            if entry.get("label") != label:
                continue
            sources_seen.append(label)
            freshness = entry.get("freshness_seconds")
            if label == "config/reserves.json" or (freshness is not None and freshness <= freshness_limit):
                chosen_payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
                source_used = label
                break
        if chosen_payload:
            break

    assets: List[Dict[str, Any]] = []
    total_usd = 0.0
    if source_used == "config/reserves.json":
        stable_set = {"USDT", "USDC", "DAI", "FDUSD", "TUSD"}
        for symbol, qty in (chosen_payload or {}).items():
            try:
                balance = float(qty)
            except Exception:
                continue
            if balance == 0:
                continue
            asset = str(symbol).upper()
            price = 1.0 if asset in stable_set else _safe_price(f"{asset}USDT")
            if price <= 0 and asset.endswith("USDT"):
                price = 1.0
            usd_value = balance * price if price > 0 else balance
            total_usd += usd_value
            assets.append(_format_asset_entry(asset, balance, price, usd_value))
    else:
        normalized = _normalize_treasury_payload(chosen_payload, source_used)
        if normalized:
            assets = normalized.get("assets", [])
            total_usd = float(normalized.get("total_usd") or 0.0)
    assets = sorted(assets, key=lambda item: item.get("usd_value", 0.0), reverse=True)
    return assets, float(total_usd), source_used, sources_seen


def get_treasury() -> Dict[str, Any]:
    """
    Return BTC/XAUT balances with USD valuations for dashboard display.
    """
    sources = _read_treasury_sources(Path("."))
    assets, total_usd, source_used, sources_seen = _select_canonical_treasury(sources)
    return {
        "assets": assets,
        "total_usd": float(total_usd),
        "source": source_used,
        "sources_seen": sources_seen,
    }


__all__ = ["get_nav_snapshot", "get_caps", "get_veto_counts", "get_treasury"]
