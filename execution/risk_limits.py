from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class RiskConfig:
    max_notional_per_trade: float           # e.g., 200.0 USDT
    max_open_notional: float                # e.g., 1000.0 USDT
    max_positions: int                      # e.g., 5
    max_leverage: float                     # e.g., 5.0
    kill_switch_drawdown_pct: float         # e.g., -10.0 (portfolio)
    min_notional: float = 10.0              # exchange min

class RiskState:
    """Holds rolling state the executor can update each loop."""
    def __init__(self) -> None:
        self.open_notional: float = 0.0
        self.open_positions: int = 0
        self.portfolio_drawdown_pct: float = 0.0

def can_open_position(symbol: str, notional: float, lev: float,
                      cfg: RiskConfig, st: RiskState) -> (bool, str):
    if st.portfolio_drawdown_pct <= cfg.kill_switch_drawdown_pct:
        return False, "kill_switch_triggered"
    if notional < cfg.min_notional:
        return False, "below_min_notional"
    if notional > cfg.max_notional_per_trade:
        return False, "exceeds_per_trade_cap"
    if lev > cfg.max_leverage:
        return False, "exceeds_leverage_cap"
    if (st.open_notional + notional) > cfg.max_open_notional:
        return False, "exceeds_open_notional_cap"
    if st.open_positions >= cfg.max_positions:
        return False, "too_many_positions"
    return True, "ok"

def should_reduce_positions(st: RiskState, cfg: RiskConfig) -> bool:
    return st.portfolio_drawdown_pct <= cfg.kill_switch_drawdown_pct

def clamp_order_size(requested_qty: float, step_size: float) -> float:
    """Round down to exchange step size to avoid rejection."""
    if step_size <= 0:
        return requested_qty
    steps = int(requested_qty / step_size)
    return max(0.0, steps * step_size)

def explain_limits(cfg: RiskConfig) -> Dict[str, float]:
    return {
        "max_notional_per_trade": cfg.max_notional_per_trade,
        "max_open_notional": cfg.max_open_notional,
        "max_positions": cfg.max_positions,
        "max_leverage": cfg.max_leverage,
        "kill_switch_drawdown_pct": cfg.kill_switch_drawdown_pct,
        "min_notional": cfg.min_notional,
    }
