"""v6 shadow execution pipeline (intel-only)."""

from __future__ import annotations

from collections import deque
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from execution.intel.router_policy import router_policy
from execution.intel.maker_offset import suggest_maker_offset_bps
from execution.log_utils import JsonlLogger
from execution.order_router import effective_px
from execution.risk_engine_v6 import OrderIntent, RiskDecision, RiskEngineV6
from execution.risk_limits import RiskState
from execution.universe_resolver import symbol_tier
from execution.utils import load_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SHADOW_LOG = PROJECT_ROOT / "logs" / "pipeline_v6_shadow.jsonl"


def _load_config(path: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _order_intent_from_signal(
    symbol: str,
    signal: Mapping[str, Any],
    nav_state: Mapping[str, Any],
    risk_cfg: Mapping[str, Any],
    positions_state: Mapping[str, Any],
) -> OrderIntent:
    def _abs_qty(value: Any) -> float:
        try:
            return abs(float(value))
        except Exception:
            return 0.0

    def _positions_from_state(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        if not isinstance(state, Mapping):
            return []
        for key in ("positions", "items", "rows"):
            raw = state.get(key)
            if isinstance(raw, list):
                return [entry for entry in raw if isinstance(entry, Mapping)]
        return []

    side = str(signal.get("side") or signal.get("signal") or "BUY").upper()
    notional = float(signal.get("notional") or signal.get("quote_notional") or 0.0)
    leverage = float(signal.get("leverage") or signal.get("lev") or 1.0)
    tier_name = signal.get("tier") or symbol_tier(symbol)
    open_positions_count = int(signal.get("open_positions_count") or 0)
    tier_gross = float(signal.get("tier_gross_notional") or 0.0)
    current_gross = float(signal.get("current_gross_notional") or nav_state.get("portfolio_gross_usd") or 0.0)
    symbol_open_qty = float(signal.get("symbol_open_qty") or nav_state.get("symbol_open_qty") or 0.0)
    nav_usd = float(nav_state.get("nav_usd") or nav_state.get("nav") or 0.0)
    price = float(signal.get("price") or 0.0)
    qty = float(signal.get("qty") or signal.get("quantity") or 0.0)
    positions_rows = _positions_from_state(positions_state)
    if positions_rows:
        if open_positions_count <= 0:
            open_positions_count = sum(1 for row in positions_rows if _abs_qty(row.get("qty") or row.get("positionAmt")) > 0.0)
        if symbol_open_qty <= 0.0:
            sym_u = symbol.upper()
            for row in positions_rows:
                if str(row.get("symbol") or "").upper() != sym_u:
                    continue
                symbol_open_qty = max(
                    symbol_open_qty,
                    _abs_qty(row.get("qty") or row.get("positionAmt")),
                )
    if qty <= 0.0 and price > 0.0:
        qty = notional / price
    return OrderIntent(
        symbol=symbol,
        side=side,
        qty=qty,
        quote_notional=notional,
        leverage=leverage,
        price=price,
        tier_name=tier_name,
        tier_gross_notional=tier_gross,
        current_gross_notional=current_gross,
        symbol_open_qty=symbol_open_qty,
        nav_usd=nav_usd,
        open_positions_count=open_positions_count,
        metadata={"signal": dict(signal), "nav_state": dict(nav_state)},
    )


def _router_decision(symbol: str, intent: OrderIntent) -> Dict[str, Any]:
    policy = router_policy(symbol)
    offset = suggest_maker_offset_bps(symbol)
    px = float(intent.price or intent.metadata.get("signal", {}).get("price_hint") or 0.0)
    eff_maker = effective_px(px, intent.side, True)
    eff_taker = effective_px(px, intent.side, False)
    return {
        "policy": {
            "maker_first": policy.maker_first,
            "taker_bias": policy.taker_bias,
            "quality": policy.quality,
        },
        "offset_bps": offset,
        "effective_px": {
            "maker": eff_maker,
            "taker": eff_taker,
        },
        "sized": {"gross_usd": intent.quote_notional, "qty": intent.qty},
    }


def run_pipeline_v6_shadow(
    symbol: str,
    signal: Mapping[str, Any],
    nav_state: Mapping[str, Any],
    positions_state: Mapping[str, Any],
    risk_limits_cfg: Mapping[str, Any],
    pairs_universe_cfg: Mapping[str, Any],
    sizing_cfg: Mapping[str, Any],
    *,
    risk_engine: Optional[RiskEngineV6] = None,
) -> Dict[str, Any]:
    engine = risk_engine or RiskEngineV6.from_configs(risk_limits_cfg, pairs_universe_cfg)
    intent = _order_intent_from_signal(symbol, signal, nav_state, risk_limits_cfg, positions_state)
    state = RiskState()
    decision = engine.check_order(intent, state)
    try:
        screener_gross = float(
            signal.get("screener_gross_usd")
            or signal.get("gross_usd")
            or signal.get("desired_gross_usd")
            or signal.get("notional")
            or 0.0
        )
    except Exception:
        screener_gross = 0.0
    result: Dict[str, Any] = {
        "symbol": symbol,
        "intent": asdict(intent),
        "risk_decision": {
            "allowed": decision.allowed,
            "clamped_qty": decision.clamped_qty,
            "reasons": decision.reasons,
            "hit_caps": decision.hit_caps,
            "diagnostics": decision.diagnostics,
        },
        "sizing": {
            "screener_gross_usd": screener_gross,
            "executor_gross_usd": float(intent.quote_notional),
            "sized_gross_usd": None,
        },
        "timestamp": time.time(),
    }
    if not decision.allowed:
        return result
    size_decision = {"gross_usd": intent.quote_notional, "qty": intent.qty}
    router_info = _router_decision(symbol, intent)
    result["size_decision"] = size_decision
    result["router_decision"] = router_info
    try:
        result["sizing"]["sized_gross_usd"] = float(intent.quote_notional or 0.0)
    except Exception:
        result["sizing"]["sized_gross_usd"] = None
    return result


def append_shadow_decision(row: Dict[str, Any]) -> None:
    """
    Append a single shadow decision to the v6 shadow log.
    Must use dynamic logger so monkeypatching PIPELINE_SHADOW_LOG works.
    """
    logger = JsonlLogger(PIPELINE_SHADOW_LOG, max_bytes=10_000_000, backup_count=5)
    logger.write(row)


def load_shadow_decisions(limit: int = 50) -> list[Mapping[str, Any]]:
    if limit <= 0 or not PIPELINE_SHADOW_LOG.exists():
        return []
    rows: deque[Mapping[str, Any]] = deque(maxlen=limit)
    try:
        with PIPELINE_SHADOW_LOG.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    rows.append(payload)
    except Exception:
        return []
    return list(rows)


def build_shadow_summary(decisions: Optional[list[Mapping[str, Any]]] = None) -> Dict[str, Any]:
    entries = decisions or []
    allowed = sum(1 for row in entries if row.get("risk_decision", {}).get("allowed"))
    vetoed = len(entries) - allowed
    return {
        "generated_ts": time.time(),
        "total": len(entries),
        "allowed": allowed,
        "vetoed": vetoed,
    }


__all__ = [
    "run_pipeline_v6_shadow",
    "append_shadow_decision",
    "load_shadow_decisions",
    "build_shadow_summary",
]
