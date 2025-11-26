from __future__ import annotations

"""
RiskEngine v6 — parity wrapper around existing risk_limits logic.

Provides structured OrderIntent/RiskDecision helpers so higher-level modules
can reason about risk outcomes without directly invoking risk_limits.
"""

from dataclasses import dataclass, field
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from execution import drawdown_tracker
from execution.risk_limits import RiskState, check_order, drawdown_snapshot, classify_drawdown_state
from execution.nav import nav_health_snapshot
from execution.risk_loader import load_risk_config
from execution.universe_resolver import universe_by_symbol
from execution.utils import metrics
from execution.utils import vol as vol_utils
from execution.utils.execution_health import compute_execution_health, summarize_atr_regimes


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


def _symbols_from_positions(positions_state: Optional[Mapping[str, Any] | List[Any]]) -> List[str]:
    symbols: List[str] = []
    if isinstance(positions_state, Mapping):
        entries = positions_state.get("items") or positions_state.get("positions") or positions_state.get("symbols")
        if isinstance(entries, Mapping):
            symbols.extend([str(key) for key in entries.keys()])
        elif isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, Mapping) and entry.get("symbol"):
                    symbols.append(str(entry.get("symbol")))
    elif isinstance(positions_state, list):
        for entry in positions_state:
            if isinstance(entry, Mapping) and entry.get("symbol"):
                symbols.append(str(entry.get("symbol")))
    return sorted({sym for sym in symbols if sym})


def _portfolio_atr_regime(nav_state: Optional[Mapping[str, Any]] = None, positions_state: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    try:
        getter = getattr(vol_utils, "get_atr_regime", None)
        if callable(getter):
            return getter(nav_state or {}, positions_state)
    except Exception:
        pass
    symbols = _symbols_from_positions(positions_state)
    if not symbols:
        try:
            symbols = sorted(universe_by_symbol().keys())
        except Exception:
            symbols = []
    try:
        return summarize_atr_regimes(symbols)
    except Exception:
        return {"atr_regime": "unknown", "median_ratio": None, "symbols": []}


def _dd_state_snapshot(risk_cfg: Mapping[str, Any] | None) -> Dict[str, Any]:
    try:
        getter = getattr(drawdown_tracker, "current_state", None)
        if callable(getter):
            state = getter()
            return state if isinstance(state, dict) else {"dd_state": state}
    except Exception:
        pass
    g_cfg = (risk_cfg or {}).get("global") if isinstance(risk_cfg, Mapping) else {}
    try:
        dd_snapshot = drawdown_snapshot(g_cfg)
    except Exception:
        dd_snapshot = {}
    drawdown_info = dd_snapshot.get("drawdown") if isinstance(dd_snapshot, Mapping) else {}
    dd_pct = None
    if isinstance(drawdown_info, Mapping):
        dd_pct = drawdown_info.get("pct")
    if dd_pct is None and isinstance(dd_snapshot, Mapping):
        dd_pct = dd_snapshot.get("dd_pct")
    try:
        alert_pct = float((g_cfg or {}).get("drawdown_alert_pct"))
    except Exception:
        alert_pct = None
    try:
        kill_pct = float((g_cfg or {}).get("max_nav_drawdown_pct") or (g_cfg or {}).get("daily_loss_limit_pct"))
    except Exception:
        kill_pct = None
    dd_state = classify_drawdown_state(dd_pct=dd_pct or 0.0, alert_pct=alert_pct, kill_pct=kill_pct)
    return {"dd_state": dd_state, "drawdown": dd_snapshot}


class RiskEngineV6:
    def __init__(self, risk_limits_config: Mapping[str, Any], pairs_universe_config: Mapping[str, Any]) -> None:
        self._risk_cfg = dict(risk_limits_config or load_risk_config())
        self._pairs_cfg = dict(pairs_universe_config or {})

    @classmethod
    def from_configs(cls, risk_limits_config: Mapping[str, Any] | None, pairs_universe_config: Mapping[str, Any] | None) -> RiskEngineV6:
        return cls(risk_limits_config or load_risk_config(), pairs_universe_config or {})

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
        if nav_val <= 0.0:
            nav_health = nav_health_snapshot()
            nav_val = float(nav_health.get("nav_total") or 0.0)
        symbol_open_qty = float(intent.symbol_open_qty or 0.0)
        price = float(intent.price or 0.0)
        tier_gross = float(intent.tier_gross_notional or 0.0)
        current_gross = float(intent.current_gross_notional or 0.0)
        now_ts = float(now if now is not None else time.time())
        tier_name = intent.tier_name
        open_positions_count = intent.open_positions_count

        try:
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
        except Exception as exc:
            logging.getLogger("risk_engine_v6").warning(
                "[risk] check_order_failed symbol=%s err=%s", intent.symbol, exc
            )
            veto = True
            details = {"reasons": ["risk_engine_error"], "error": str(exc)}
        diagnostics: Dict[str, Any]
        if isinstance(details, Mapping):
            diagnostics = dict(details)
        else:
            diagnostics = {"detail": details}
        diagnostics.setdefault("gate", diagnostics.get("gate") or "risk_engine_v6")
        if "thresholds" not in diagnostics or not isinstance(diagnostics.get("thresholds"), dict):
            diagnostics["thresholds"] = {}
        if "observations" not in diagnostics or not isinstance(diagnostics.get("observations"), dict):
            extra_obs = {k: v for k, v in diagnostics.items() if k not in {"reasons", "thresholds", "gate"}}
            diagnostics["observations"] = extra_obs if extra_obs else {}
        reasons = [str(reason) for reason in diagnostics.get("reasons", []) if reason]
        hit_caps = {reason: True for reason in reasons}
        if veto and not reasons:
            reasons = ["risk_engine_error"]
            hit_caps["risk_engine_error"] = True
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
        nav_snapshot = nav_state if isinstance(nav_state, Mapping) else nav_health_snapshot()
        try:
            dd_state = _dd_state_snapshot(self._risk_cfg)
        except Exception:
            dd_state = {"dd_state": "unknown"}
        try:
            atr_summary = _portfolio_atr_regime(nav_snapshot, positions_state)
        except Exception:
            atr_summary = {"atr_regime": "unknown", "median_ratio": None, "symbols": []}
        try:
            fee_snapshot = metrics.fee_pnl_ratio(symbol=None)
        except Exception:
            fee_snapshot = {"fee_pnl_ratio": None, "fees": 0.0, "pnl": 0.0, "window_days": 7}
        return {
            "updated_ts": now,
            "symbols": symbols,
            "dd_state": dd_state,
            "atr_regime": atr_summary.get("atr_regime"),
            "atr": atr_summary,
            "fee_pnl_ratio": fee_snapshot.get("fee_pnl_ratio"),
            "fee_pnl": fee_snapshot,
            "nav_health": nav_snapshot,
        }


__all__ = ["RiskEngineV6", "OrderIntent", "RiskDecision"]
