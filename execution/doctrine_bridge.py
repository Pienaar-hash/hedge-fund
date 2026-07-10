from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from execution.alpha_decay import AlphaDecayState, load_alpha_decay_state
from execution.doctrine_kernel import AlphaHealthSnapshot, PortfolioSnapshot
from execution.helpers import iso_to_ts
from execution.hydra_engine import HydraState, load_hydra_state

HYDRA_STATE_MAX_AGE_S = 900.0
ALPHA_DECAY_STATE_MAX_AGE_S = 900.0


@dataclass(frozen=True)
class DoctrineEntryInputs:
    portfolio: Optional[PortfolioSnapshot]
    alpha_health: Optional[AlphaHealthSnapshot]
    telemetry: Dict[str, Any]
    degraded_details: Optional[Dict[str, Any]] = None

    @property
    def degraded(self) -> bool:
        return self.degraded_details is not None


def _coerce_updated_ts(raw: Any) -> Optional[float]:
    if raw in (None, ""):
        return None
    if isinstance(raw, (int, float)):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None
    return iso_to_ts(str(raw))


def _state_health(
    *,
    label: str,
    path: Path,
    updated_ts: Any,
    now_ts: float,
    max_age_s: float,
) -> Dict[str, Any]:
    ts = _coerce_updated_ts(updated_ts)
    exists = path.exists()
    age_s = None if ts is None else max(0.0, now_ts - ts)
    fresh = bool(exists and ts is not None and age_s is not None and age_s <= max_age_s)
    status = "healthy" if fresh else "missing"
    code = None
    message = ""
    if not exists or ts is None:
        code = f"{label}_MISSING"
        message = f"{label.replace('_', ' ').title()} missing or unreadable"
    elif age_s is not None and age_s > max_age_s:
        status = "stale"
        code = f"{label}_STALE"
        message = (
            f"{label.replace('_', ' ').title()} stale: age={age_s:.1f}s max={max_age_s:.1f}s"
        )
    return {
        "path": str(path),
        "exists": exists,
        "updated_ts": updated_ts,
        "updated_epoch": ts,
        "age_s": age_s,
        "max_age_s": max_age_s,
        "fresh": fresh,
        "status": status,
        "code": code,
        "message": message,
    }


def _build_portfolio_snapshot(
    hydra_state: HydraState,
    *,
    head: str,
    total_exposure_pct: float,
    drawdown_pct: float,
    risk_mode: str,
) -> PortfolioSnapshot:
    remaining: Dict[str, float] = {}
    heads = set(hydra_state.head_budgets) | set(hydra_state.head_usage)
    for name in heads:
        budget = float(hydra_state.head_budgets.get(name, 0.0) or 0.0)
        used = float(hydra_state.head_usage.get(name, 0.0) or 0.0)
        remaining[name] = max(0.0, budget - used)
    remaining.setdefault(head, 0.0)
    return PortfolioSnapshot(
        head_budget_remaining=remaining,
        total_exposure_pct=total_exposure_pct,
        drawdown_pct=drawdown_pct,
        risk_mode=risk_mode,
    )


def _build_alpha_health_snapshot(symbol: str, decay_state: AlphaDecayState) -> AlphaHealthSnapshot:
    symbol_stats = decay_state.symbols.get(symbol)
    survival = decay_state.avg_symbol_survival
    carry_edge = 0.0
    if symbol_stats is not None:
        survival = float(symbol_stats.survival_prob)
        carry_edge = float(symbol_stats.last_edge_score)
    overall_health = max(0.0, min(1.0, float(decay_state.overall_alpha_health)))
    return AlphaHealthSnapshot(
        survival_probability=max(0.0, min(1.0, float(survival))),
        trend_strength=overall_health,
        carry_edge=carry_edge,
        execution_drag_bps=0.0,
        crossfire_spread_remaining=overall_health,
    )


def collect_doctrine_dependency_health(
    *,
    hydra_path: Path | str = Path("logs/state/hydra_state.json"),
    alpha_decay_path: Path | str = Path("logs/state/alpha_decay.json"),
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    current_ts = float(now_ts if now_ts is not None else time.time())
    hydra_state_path = Path(hydra_path)
    alpha_state_path = Path(alpha_decay_path)
    hydra_state = load_hydra_state(hydra_state_path)
    alpha_decay_state = load_alpha_decay_state(alpha_state_path)
    hydra = _state_health(
        label="HYDRA_STATE",
        path=hydra_state_path,
        updated_ts=hydra_state.updated_ts,
        now_ts=current_ts,
        max_age_s=HYDRA_STATE_MAX_AGE_S,
    )
    alpha = _state_health(
        label="ALPHA_DECAY_STATE",
        path=alpha_state_path,
        updated_ts=alpha_decay_state.updated_ts if alpha_decay_state is not None else None,
        now_ts=current_ts,
        max_age_s=ALPHA_DECAY_STATE_MAX_AGE_S,
    )
    return {
        "updated_ts": current_ts,
        "hydra": hydra,
        "alpha_decay": alpha,
        "healthy": hydra["fresh"] and alpha["fresh"],
    }


def resolve_doctrine_entry_inputs(
    *,
    symbol: str,
    head: str,
    total_exposure_pct: float,
    drawdown_pct: float,
    risk_mode: str,
    hydra_path: Path | str = Path("logs/state/hydra_state.json"),
    alpha_decay_path: Path | str = Path("logs/state/alpha_decay.json"),
    now_ts: Optional[float] = None,
) -> DoctrineEntryInputs:
    current_ts = float(now_ts if now_ts is not None else time.time())
    health = collect_doctrine_dependency_health(
        hydra_path=hydra_path,
        alpha_decay_path=alpha_decay_path,
        now_ts=current_ts,
    )
    for key in ("hydra", "alpha_decay"):
        part = health[key]
        if not part.get("fresh"):
            return DoctrineEntryInputs(
                portfolio=None,
                alpha_health=None,
                telemetry=health,
                degraded_details={
                    "code": part.get("code") or "DOCTRINE_INPUTS_UNAVAILABLE",
                    "reason": part.get("message") or "Doctrine dependency unhealthy",
                    "dependency": key,
                    "telemetry": health,
                },
            )

    hydra_state = load_hydra_state(hydra_path)
    alpha_decay_state = load_alpha_decay_state(alpha_decay_path)
    if alpha_decay_state is None:
        return DoctrineEntryInputs(
            portfolio=None,
            alpha_health=None,
            telemetry=health,
            degraded_details={
                "code": "ALPHA_DECAY_STATE_MISSING",
                "reason": "Alpha decay state missing or unreadable",
                "dependency": "alpha_decay",
                "telemetry": health,
            },
        )

    portfolio = _build_portfolio_snapshot(
        hydra_state,
        head=head,
        total_exposure_pct=total_exposure_pct,
        drawdown_pct=drawdown_pct,
        risk_mode=risk_mode,
    )
    alpha_health = _build_alpha_health_snapshot(symbol, alpha_decay_state)
    health["head"] = head
    health["symbol"] = symbol
    health["head_budget_remaining"] = portfolio.head_budget_remaining.get(head, 0.0)
    health["alpha_survival_probability"] = alpha_health.survival_probability
    health["overall_alpha_health"] = alpha_decay_state.overall_alpha_health
    return DoctrineEntryInputs(
        portfolio=portfolio,
        alpha_health=alpha_health,
        telemetry=health,
        degraded_details=None,
    )


__all__ = [
    "ALPHA_DECAY_STATE_MAX_AGE_S",
    "HYDRA_STATE_MAX_AGE_S",
    "DoctrineEntryInputs",
    "collect_doctrine_dependency_health",
    "resolve_doctrine_entry_inputs",
]
