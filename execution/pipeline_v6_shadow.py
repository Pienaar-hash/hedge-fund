"""v6 shadow execution pipeline (intel-only)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from collections import deque
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from execution.intel.router_policy import router_policy
from execution.intel.maker_offset import suggest_maker_offset_bps
from execution.log_utils import get_logger
from execution.order_router import effective_px
from execution.risk_engine_v6 import OrderIntent, RiskDecision, RiskEngineV6
from execution.risk_limits import RiskState
from execution.size_model import suggest_gross_usd
from execution.universe_resolver import symbol_tier
from execution.utils import load_json

PIPELINE_SHADOW_LOG = Path("logs/pipeline_v6_shadow.jsonl")
PIPELINE_SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
PIPELINE_SHADOW_LOGGER = get_logger(str(PIPELINE_SHADOW_LOG))


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


def _size_order(symbol: str, intent: OrderIntent, sizing_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    risk_payload = {
        "min_notional_usd": float(sizing_cfg.get("min_notional_usd", 0.0) or 0.0),
        "fallback_gross_usd": intent.quote_notional,
        "price": intent.price,
    }
    suggestion = suggest_gross_usd(
        symbol,
        intent.nav_usd,
        float(intent.metadata.get("signal", {}).get("signal_strength", 1.0) or 1.0),
        risk_payload,
    )
    return dict(suggestion)


def _router_decision(symbol: str, intent: OrderIntent, sized: Mapping[str, Any]) -> Dict[str, Any]:
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
        "sized": dict(sized),
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
        "timestamp": time.time(),
    }
    if not decision.allowed:
        return result
    sized = _size_order(symbol, intent, sizing_cfg)
    router_info = _router_decision(symbol, intent, sized)
    result["size_decision"] = sized
    result["router_decision"] = router_info
    return result


def append_shadow_decision(decision: Mapping[str, Any]) -> None:
    try:
        PIPELINE_SHADOW_LOGGER.write(decision)
    except Exception:
        pass


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
