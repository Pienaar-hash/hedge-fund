"""Execution Hardening — KPI tiles."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from execution.risk_limits import RiskGate
from execution.utils import load_json
from execution.utils.metrics import (
    fill_notional_7d,
    submitted_notional_7d,
    gross_realized_7d,
    fees_7d,
    realized_slippage_bps_7d,
    hourly_expectancy as metrics_hourly_expectancy,
    router_effectiveness_7d,
)
from execution.utils.execution_health import compute_execution_health
from execution.utils.expectancy import rolling_expectancy
from execution.exchange_utils import get_price
from execution.intel.symbol_score import compute_symbol_score
from execution.intel.expectancy_map import hourly_expectancy as intel_hourly_expectancy
from execution.risk_loader import load_risk_config

LOG = logging.getLogger("dash.live_helpers")

STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD"}
STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
NAV_STATE_PATH = Path(os.getenv("NAV_STATE_PATH") or (STATE_DIR / "nav_state.json"))
SYNCED_STATE_PATH = Path(os.getenv("SYNCED_STATE_PATH") or (STATE_DIR / "synced_state.json"))
UNIVERSE_STATE_PATH = Path(os.getenv("UNIVERSE_STATE_PATH") or (STATE_DIR / "universe.json"))
ROUTER_HEALTH_STATE_PATH = Path(os.getenv("ROUTER_HEALTH_STATE_PATH") or (STATE_DIR / "router_health.json"))
EXPECTANCY_STATE_PATH = Path(os.getenv("EXPECTANCY_STATE_PATH") or (STATE_DIR / "expectancy_v6.json"))
SYMBOL_SCORES_STATE_PATH = Path(os.getenv("SYMBOL_SCORES_STATE_PATH") or (STATE_DIR / "symbol_scores_v6.json"))
PIPELINE_COMPARE_STATE_PATH = Path(os.getenv("PIPELINE_V6_COMPARE_STATE_PATH") or (STATE_DIR / "pipeline_v6_compare_summary.json"))
PIPELINE_COMPARE_LOG_PATH = Path(os.getenv("PIPELINE_V6_COMPARE_LOG_PATH") or "logs/pipeline_v6_compare.jsonl")
RISK_ALLOCATOR_STATE_PATH = Path(os.getenv("RISK_ALLOC_STATE_PATH") or (STATE_DIR / "risk_allocation_suggestions_v6.json"))
ROUTER_POLICY_STATE_PATH = Path(
    os.getenv("ROUTER_POLICY_STATE_PATH") or (STATE_DIR / "router_policy_state_v6.json")
)
ROUTER_SUGGESTIONS_STATE_PATH = Path(
    os.getenv("ROUTER_SUGGESTIONS_STATE_PATH") or (STATE_DIR / "router_policy_suggestions_v6.json")
)
PIPELINE_SHADOW_HEAD_STATE_PATH = Path(os.getenv("PIPELINE_SHADOW_HEAD_STATE_PATH") or (STATE_DIR / "pipeline_v6_shadow_head.json"))
RUNTIME_PROBE_STATE_PATH = Path(os.getenv("RUNTIME_PROBE_STATE_PATH") or (STATE_DIR / "v6_runtime_probe.json"))


def kpi_tiles(symbol: str | None = None) -> Dict[str, Any]:
    submitted = submitted_notional_7d(symbol)
    filled = fill_notional_7d(symbol)
    fees = fees_7d(symbol) or 0.0
    gpnl = gross_realized_7d(symbol) or 0.0
    fee_ratio = (fees / gpnl) if gpnl else None
    efficiency = (filled / submitted) if submitted else None
    slip = realized_slippage_bps_7d(symbol)
    expectancy = rolling_expectancy(symbol) if symbol else None
    return {
        "fill_eff": efficiency,
        "fee_pnl_ratio": fee_ratio,
        "slip_bps": slip,
        "expectancy": expectancy,
        "hourly_expectancy": metrics_hourly_expectancy(symbol) if symbol else None,
    }


def execution_kpis(symbol: str | None = None) -> Dict[str, Any]:
    base = kpi_tiles(symbol)
    eff = router_effectiveness_7d(symbol)
    if isinstance(eff, dict):
        base.update(
            {
                "maker_fill_ratio": eff.get("maker_fill_ratio"),
                "fallback_ratio": eff.get("fallback_ratio"),
                "slip_q25": eff.get("slip_q25"),
                "slip_q50": eff.get("slip_q50"),
                "slip_q75": eff.get("slip_q75"),
            }
        )
    return base


def execution_health(symbol: str | None = None) -> Dict[str, Any]:
    """
    Thin wrapper so the dashboard can pull execution health snapshots.
    """
    return compute_execution_health(symbol)


def _try_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _load_state_json(path: Path, default: Any | None = None) -> Any:
    default_obj = {} if default is None else copy.deepcopy(default)
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text())
    except Exception as exc:
        LOG.debug("[dash] failed to load state file %s: %s", path, exc, exc_info=True)
    return default_obj


def _ensure_dict(payload: Any) -> Dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _ensure_list(payload: Any) -> List[Any]:
    return payload if isinstance(payload, list) else []


def _resolve_nav_state_path() -> Path:
    """
    Resolve the canonical v6 NAV state path.
    No legacy nav.json fallback is allowed in v6.
    """
    return NAV_STATE_PATH


def ensure_timestamp(ts: Any) -> str:
    """Best-effort conversion to ISO UTC string; returns "N/A" if unknown."""
    if ts is None:
        return "N/A"
    try:
        val = float(ts)
        if val > 1e12:
            val /= 1000.0
        return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
    except Exception:
        pass
    if isinstance(ts, str):
        text = ts.strip()
        if not text:
            return "N/A"
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).astimezone(timezone.utc).isoformat()
        except Exception:
            return "N/A"
    return "N/A"


def load_universe_state() -> List[Dict[str, Any]]:
    payload = _load_state_json(UNIVERSE_STATE_PATH, {})
    if isinstance(payload, dict) and isinstance(payload.get("universe"), list):
        return [dict(item) for item in payload["universe"] if isinstance(item, dict)]
    return []


def load_expectancy_v6_state() -> Dict[str, Any]:
    payload = _load_state_json(EXPECTANCY_STATE_PATH, {})
    return _ensure_dict(payload)


def load_expectancy_v6() -> Dict[str, Any]:
    payload = _load_state_json(EXPECTANCY_STATE_PATH, {})
    return _ensure_dict(payload)


def load_symbol_scores_v6_state() -> Dict[str, Any]:
    payload = _load_state_json(SYMBOL_SCORES_STATE_PATH, {})
    return _ensure_dict(payload)


def load_symbol_scores_v6() -> Dict[str, Any]:
    payload = _load_state_json(SYMBOL_SCORES_STATE_PATH, {})
    return _ensure_dict(payload)


def load_risk_allocator_v6() -> Dict[str, Any]:
    payload = _load_state_json(RISK_ALLOCATOR_STATE_PATH, {})
    return _ensure_dict(payload)


def load_router_policy_v6() -> Dict[str, Any]:
    payload = _load_state_json(ROUTER_POLICY_STATE_PATH, {})
    return _ensure_dict(payload)


def load_router_health_state() -> Dict[str, Any]:
    payload = _load_state_json(ROUTER_HEALTH_STATE_PATH, {})
    return _ensure_dict(payload)


def load_router_suggestions_v6() -> Dict[str, Any]:
    payload = _load_state_json(ROUTER_SUGGESTIONS_STATE_PATH, {})
    result: Dict[str, Any] = payload if isinstance(payload, dict) else {}
    ts_raw = result.get("generated_ts")
    ts_float: Optional[float] = None
    if isinstance(ts_raw, (int, float)):
        ts_float = float(ts_raw)
    elif isinstance(ts_raw, str):
        ts_clean = ts_raw.strip()
        if ts_clean:
            try:
                ts_float = float(ts_clean)
            except Exception:
                try:
                    ts_dt = datetime.fromisoformat(ts_clean.replace("Z", "+00:00"))
                    ts_float = ts_dt.timestamp()
                except Exception:
                    ts_float = None
    result["generated_ts"] = ts_float
    if ts_float is not None:
        result["generated_at"] = datetime.fromtimestamp(ts_float, tz=timezone.utc).isoformat()
        result["stale"] = (time.time() - ts_float) > 3600
    else:
        result["generated_at"] = "N/A"
        result["stale"] = True
    return result


def load_pipeline_v6_compare_summary() -> Dict[str, Any]:
    payload = _load_state_json(PIPELINE_COMPARE_STATE_PATH, {})
    return _ensure_dict(payload)


def load_compare_summary() -> Dict[str, Any]:
    payload = _load_state_json(PIPELINE_COMPARE_STATE_PATH, {})
    return _ensure_dict(payload)


def load_pipeline_v6_compare_events(limit: int = 200) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = PIPELINE_COMPARE_LOG_PATH
    if not path.exists() or limit <= 0:
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except Exception:
        return records
    lines = lines[-limit:]
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def load_shadow_head() -> Dict[str, Any]:
    payload = _load_state_json(PIPELINE_SHADOW_HEAD_STATE_PATH, {})
    return _ensure_dict(payload)


def load_synced_state() -> Dict[str, Any]:
    payload = _load_state_json(SYNCED_STATE_PATH, {})
    return _ensure_dict(payload)


def get_symbol_intel_table(limit: int = 20) -> List[Dict[str, Any]]:
    scores_snapshot = load_symbol_scores_v6_state()
    expectancy_snapshot = load_expectancy_v6_state()
    expectancy_map = expectancy_snapshot.get("symbols") if isinstance(expectancy_snapshot.get("symbols"), dict) else {}
    rows: List[Dict[str, Any]] = []
    for entry in scores_snapshot.get("symbols", []) if isinstance(scores_snapshot.get("symbols"), list) else []:
        symbol = entry.get("symbol")
        if not symbol:
            continue
        exp_entry = expectancy_map.get(symbol)
        row = {
            "symbol": symbol,
            "score": entry.get("score"),
            "expectancy": (exp_entry or {}).get("expectancy"),
            "hit_rate": (exp_entry or {}).get("hit_rate"),
            "max_drawdown": (exp_entry or {}).get("max_drawdown"),
            "components": entry.get("components"),
        }
        rows.append(row)
    rows.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    if limit > 0:
        rows = rows[:limit]
    return rows


def load_runtime_probe() -> Dict[str, Any]:
    payload = _load_state_json(RUNTIME_PROBE_STATE_PATH, {})
    data = _ensure_dict(payload)

    def _flag(key: str, alt: str | None = None) -> bool:
        return bool(data.get(key) or (data.get(alt) if alt else False))

    flags = {
        "INTEL_V6": _flag("INTEL_V6_ENABLED", "intel_v6_enabled"),
        "RISK_ENGINE_V6": _flag("RISK_ENGINE_V6_ENABLED", "risk_v6_enabled"),
        "PIPELINE_V6_SHADOW": _flag("PIPELINE_V6_SHADOW_ENABLED", "pipeline_v6_enabled"),
        "ROUTER_AUTOTUNE_V6": _flag("ROUTER_AUTOTUNE_V6_ENABLED", "router_autotune_v6_enabled"),
        "FEEDBACK_ALLOCATOR_V6": _flag("FEEDBACK_ALLOCATOR_V6_ENABLED", "feedback_allocator_v6_enabled"),
    }

    result: Dict[str, Any] = {
        "engine_version": data.get("engine_version"),
        "nav_age_ms": data.get("nav_age_ms"),
        "loop_latency_ms": data.get("loop_latency_ms"),
        "ts": data.get("ts"),
        "flags": flags,
    }
    result["generated_at"] = ensure_timestamp(data.get("ts"))
    # Backwards compatible aliases
    result.update(
        {
            "risk_v6_enabled": flags["RISK_ENGINE_V6"],
            "router_autotune_v6_enabled": flags["ROUTER_AUTOTUNE_V6"],
            "pipeline_v6_enabled": flags["PIPELINE_V6_SHADOW"],
            "intel_v6_enabled": flags["INTEL_V6"],
            "feedback_allocator_v6_enabled": flags["FEEDBACK_ALLOCATOR_V6"],
        }
    )
    return result


def _read_risk_cfg() -> Dict[str, Any]:
    try:
        cfg = load_risk_config()
    except Exception:
        cfg = {}
    return cfg if isinstance(cfg, dict) else {}


def get_nav_snapshot(nav_path: str | None = None) -> Dict[str, float]:
    """
    Return a NAV snapshot sourced from logs/state/nav_state.json; never raises.
    """
    _ = nav_path  # legacy param; state source is authoritative
    now_ts = time.time()
    result = {"nav": 0.0, "equity": 0.0, "ts": float(now_ts)}

    state_payload = _load_state_json(_resolve_nav_state_path(), {}) or {}
    if isinstance(state_payload, dict):
        series = state_payload.get("series")
        latest_entry = None
        if isinstance(series, list):
            for item in reversed(series):
                if isinstance(item, dict):
                    latest_entry = item
                    break
        try:
            nav_val = float(
                (latest_entry or {}).get("equity")
                or (latest_entry or {}).get("nav")
                or state_payload.get("total_equity")
                or state_payload.get("nav")
                or 0.0
            )
        except Exception:
            nav_val = 0.0
        try:
            equity_val = float(state_payload.get("total_equity") or nav_val)
        except Exception:
            equity_val = nav_val

        ts_val = None
        for candidate in (
            (latest_entry or {}).get("t"),
            (latest_entry or {}).get("ts"),
            state_payload.get("updated_at"),
            state_payload.get("ts"),
        ):
            if candidate is None:
                continue
            try:
                ts_raw = float(candidate)
                ts_val = ts_raw / 1000.0 if ts_raw > 1e12 else ts_raw
                break
            except Exception:
                try:
                    ts_val = datetime.fromisoformat(str(candidate).replace("Z", "+00:00")).timestamp()
                    break
                except Exception:
                    continue
        result.update({"nav": nav_val, "equity": equity_val, "ts": float(ts_val or now_ts)})
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
    universe_entries = load_universe_state()
    if universe_entries:
        min_vals: List[float] = []
        for entry in universe_entries:
            try:
                raw = entry.get("min_notional")
                if raw is None:
                    continue
                min_vals.append(float(raw))
            except Exception:
                continue
        positive = [val for val in min_vals if val > 0]
        target_vals = positive or min_vals
        if target_vals:
            caps["min_notional"] = min(target_vals)
    try:
        cfg = _read_risk_cfg()
        gate = RiskGate(cfg)
        sizing = gate.sizing
        caps["max_trade_nav_pct"] = float(sizing.get("max_trade_nav_pct") or 0.0)
        gross_pct = sizing.get("max_gross_exposure_pct") or sizing.get("max_portfolio_gross_nav_pct")
        caps["max_gross_exposure_pct"] = float(gross_pct or 0.0)
        caps["max_symbol_exposure_pct"] = float(sizing.get("max_symbol_exposure_pct") or 0.0)
        if not caps["min_notional"]:
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


def get_symbol_score(symbol: str) -> Dict[str, Any]:
    """
    Thin wrapper around execution.intel.symbol_score.compute_symbol_score.
    """
    return compute_symbol_score(symbol)


def get_hourly_expectancy(symbol: str | None = None) -> Dict[int, Dict[str, Any]]:
    """
    Wrapper around execution.intel.expectancy_map.hourly_expectancy.
    """
    return intel_hourly_expectancy(symbol)


__all__ = [
    "get_nav_snapshot",
    "get_caps",
    "get_veto_counts",
    "get_treasury",
    "get_symbol_score",
    "get_hourly_expectancy",
    "load_runtime_probe",
    "load_expectancy_v6",
    "load_symbol_scores_v6",
    "load_risk_allocator_v6",
    "load_router_policy_v6",
    "load_router_health_state",
    "load_router_suggestions_v6",
    "load_shadow_head",
    "load_compare_summary",
    "ensure_timestamp",
]
