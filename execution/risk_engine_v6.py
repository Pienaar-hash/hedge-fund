from __future__ import annotations

"""
RiskEngine v6 — parity wrapper around existing risk_limits logic.

Provides structured OrderIntent/RiskDecision helpers so higher-level modules
can reason about risk outcomes without directly invoking risk_limits.
"""

from dataclasses import dataclass, field
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from execution.risk_limits import RiskState, check_order
from execution.universe_resolver import universe_by_symbol
from execution.utils.execution_health import compute_execution_health


@dataclass
class OrderIntent:
    symbol: str
    side: str
    qty: float
    quote_notional: float
    leverage: float = 1.0
    price: float = 0.0
    tier_name: Optional[str] = None
    tier_gross_notional: Optional[float] = None
    current_gross_notional: Optional[float] = None
    symbol_open_qty: Optional[float] = None
    nav_usd: Optional[float] = None
    open_positions_count: Optional[int] = None
    strategy_id: Optional[str] = None
    account_mode: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecision:
    allowed: bool
    clamped_qty: float
    reasons: List[str] = field(default_factory=list)
    hit_caps: Dict[str, bool] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class RiskEngineV6:
    def __init__(self, risk_limits_config: Mapping[str, Any], pairs_universe_config: Mapping[str, Any]) -> None:
        self._risk_cfg = dict(risk_limits_config or {})
        self._pairs_cfg = dict(pairs_universe_config or {})

    @classmethod
    def from_configs(cls, risk_limits_config: Mapping[str, Any] | None, pairs_universe_config: Mapping[str, Any] | None) -> RiskEngineV6:
        return cls(risk_limits_config or {}, pairs_universe_config or {})

    @classmethod
    def from_files(cls, risk_limits_path: str, pairs_universe_path: str) -> RiskEngineV6:
        def _load(path: str) -> Mapping[str, Any]:
            try:
                payload = json.loads(Path(path).read_text())
            except Exception:
                payload = {}
            return payload if isinstance(payload, Mapping) else {}

        return cls(_load(risk_limits_path), _load(pairs_universe_path))

    @property
    def risk_config(self) -> Mapping[str, Any]:
        return self._risk_cfg

    def check_order(
        self,
        intent: OrderIntent,
        state: RiskState,
        *,
        nav_state: Optional[Mapping[str, Any]] = None,
        positions_state: Optional[Mapping[str, Any]] = None,
        now: Optional[float] = None,
    ) -> RiskDecision:
        """Mirror risk_limits.check_order and translate into RiskDecision."""

        req_notional = float(intent.quote_notional or 0.0)
        nav_val = float(intent.nav_usd if intent.nav_usd is not None else (nav_state or {}).get("nav_usd") or 0.0)
        symbol_open_qty = float(intent.symbol_open_qty or 0.0)
        price = float(intent.price or 0.0)
        tier_gross = float(intent.tier_gross_notional or 0.0)
        current_gross = float(intent.current_gross_notional or 0.0)
        now_ts = float(now if now is not None else time.time())
        tier_name = intent.tier_name
        open_positions_count = intent.open_positions_count

        veto, details = check_order(
            symbol=intent.symbol,
            side=intent.side,
            requested_notional=req_notional,
            price=price,
            nav=nav_val,
            open_qty=symbol_open_qty,
            now=now_ts,
            cfg=self._risk_cfg,
            state=state,
            current_gross_notional=current_gross,
            lev=float(intent.leverage or 0.0),
            open_positions_count=open_positions_count,
            tier_name=tier_name,
            current_tier_gross_notional=tier_gross,
        )
        diagnostics: Dict[str, Any]
        if isinstance(details, Mapping):
            diagnostics = dict(details)
        else:
            diagnostics = {"detail": details}
        reasons = [str(reason) for reason in diagnostics.get("reasons", []) if reason]
        hit_caps = {reason: True for reason in reasons}
        return RiskDecision(
            allowed=not veto,
            clamped_qty=float(intent.qty or 0.0),
            reasons=reasons,
            hit_caps=hit_caps,
            diagnostics=diagnostics,
        )

    def check_portfolio(
        self,
        nav_state: Optional[Mapping[str, Any]] = None,
        positions_state: Optional[Mapping[str, Any]] = None,
    ) -> RiskDecision:
        """Placeholder portfolio-level check (parity with existing snapshot logic)."""
        snapshot = self.build_risk_snapshot(nav_state=nav_state, positions_state=positions_state)
        return RiskDecision(
            allowed=True,
            clamped_qty=0.0,
            diagnostics=snapshot,
        )

    def build_risk_snapshot(
        self,
        *,
        nav_state: Optional[Mapping[str, Any]] = None,
        positions_state: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Centralized risk snapshot builder (mirrors executor telemetry)."""
        now = time.time()
        symbols: List[Dict[str, Any]] = []
        for sym in sorted(universe_by_symbol().keys()):
            try:
                entry = compute_execution_health(sym)
            except Exception:
                continue
            entry["updated_ts"] = now
            symbols.append(entry)
        return {"updated_ts": now, "symbols": symbols}


__all__ = ["RiskEngineV6", "OrderIntent", "RiskDecision"]
