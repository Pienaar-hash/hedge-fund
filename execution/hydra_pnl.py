"""
Hydra PnL Attribution & Drawdown Engine — v7.9_P2

Gives each Hydra head its own PnL, equity curve, drawdown, and health state:

1. Per-head realized and unrealized PnL
2. Per-head equity curves and drawdowns
3. Per-head kill-switches and throttling
4. Performance-aware signals for Cerberus & Hydra
5. Head-level performance surfaces for dashboard

Architecture:
    Fills/Trades  ──┐
                    │
    Positions     ──┼──► Hydra PnL Engine ──► Head Stats ──► Cerberus/Hydra throttles
                    │                              ↓
    Head Contribs ──┘              logs/state/hydra_pnl.json

Hydra already routes intents. P2 makes performance observable and governable per head.

Single writer rule: Only executor may write hydra_pnl.json.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_STATE_PATH = Path("logs/state/hydra_pnl.json")
DEFAULT_EVENT_LOG_PATH = Path("logs/hydra/hydra_pnl_events.jsonl")
DEFAULT_CONFIG_PATH = Path("config/strategy_config.json")

_LOG = logging.getLogger(__name__)

# Canonical strategy heads (same as Hydra/Cerberus)
STRATEGY_HEADS = [
    "TREND",
    "MEAN_REVERT",
    "RELATIVE_VALUE",
    "CATEGORY",
    "VOL_HARVEST",
    "EMERGENT_ALPHA",
]

# Default config values
DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_MIN_TRADES = 30
DEFAULT_MAX_DRAWDOWN = 0.25
DEFAULT_MIN_WIN_RATE = 0.40
DEFAULT_MIN_R_MULTIPLE = -0.5
DEFAULT_COOLDOWN_CYCLES = 720
DEFAULT_DD_SOFT_THRESHOLD = 0.10
DEFAULT_DD_HARD_THRESHOLD = 0.20
DEFAULT_SOFT_SCALE = 0.5
DEFAULT_HARD_SCALE = 0.0


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class KillSwitchConfig:
    """Configuration for per-head kill switches."""

    max_drawdown: float = DEFAULT_MAX_DRAWDOWN
    min_win_rate: float = DEFAULT_MIN_WIN_RATE
    min_R_multiple: float = DEFAULT_MIN_R_MULTIPLE
    cooldown_cycles: int = DEFAULT_COOLDOWN_CYCLES


@dataclass
class ThrottlingConfig:
    """Configuration for per-head throttling based on drawdown."""

    dd_soft_threshold: float = DEFAULT_DD_SOFT_THRESHOLD
    dd_hard_threshold: float = DEFAULT_DD_HARD_THRESHOLD
    soft_scale: float = DEFAULT_SOFT_SCALE
    hard_scale: float = DEFAULT_HARD_SCALE


@dataclass
class HydraPnlConfig:
    """Configuration for Hydra PnL Attribution & Drawdown Engine."""

    enabled: bool = False
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    min_trades_for_stats: int = DEFAULT_MIN_TRADES
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    throttling: ThrottlingConfig = field(default_factory=ThrottlingConfig)

    def __post_init__(self) -> None:
        """Validate config values."""
        self.lookback_days = max(1, min(365, self.lookback_days))
        self.min_trades_for_stats = max(1, min(1000, self.min_trades_for_stats))


@dataclass
class HeadPnlStats:
    """PnL statistics for a single strategy head."""

    head: str
    equity: float = 0.0
    max_equity: float = 0.0
    drawdown: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    total_R: float = 0.0
    trades_with_R: int = 0
    avg_R: float = 0.0
    gross_exposure: float = 0.0
    veto_count: int = 0
    last_trade_ts: float = 0.0
    last_active_ts: float = 0.0
    kill_switch_active: bool = False
    cooldown_remaining: int = 0
    throttle_scale: float = 1.0

    def update_equity(self) -> None:
        """Update equity and drawdown from PnL."""
        self.equity = self.realized_pnl + self.unrealized_pnl
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        if self.max_equity > 0:
            self.drawdown = (self.max_equity - self.equity) / self.max_equity
        else:
            self.drawdown = 0.0

    def update_win_rate(self) -> None:
        """Update win rate from trade counts."""
        if self.trades > 0:
            self.win_rate = self.wins / self.trades
        else:
            self.win_rate = 0.0

    def update_avg_R(self) -> None:
        """Update average R multiple."""
        if self.trades_with_R > 0:
            self.avg_R = self.total_R / self.trades_with_R
        else:
            self.avg_R = 0.0

    def record_trade(self, pnl: float, R_multiple: Optional[float] = None) -> None:
        """Record a completed trade."""
        self.trades += 1
        self.realized_pnl += pnl
        if pnl > 0:
            self.wins += 1
        self.update_win_rate()

        if R_multiple is not None:
            self.total_R += R_multiple
            self.trades_with_R += 1
            self.update_avg_R()

        self.last_trade_ts = time.time()
        self.last_active_ts = time.time()
        self.update_equity()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "head": self.head,
            "equity": round(self.equity, 4),
            "max_equity": round(self.max_equity, 4),
            "drawdown": round(self.drawdown, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "trades": self.trades,
            "wins": self.wins,
            "win_rate": round(self.win_rate, 4),
            "avg_R": round(self.avg_R, 4),
            "gross_exposure": round(self.gross_exposure, 4),
            "veto_count": self.veto_count,
            "last_trade_ts": self.last_trade_ts,
            "last_active_ts": self.last_active_ts,
            "kill_switch_active": self.kill_switch_active,
            "cooldown_remaining": self.cooldown_remaining,
            "throttle_scale": round(self.throttle_scale, 4),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeadPnlStats":
        """Create from dictionary."""
        return cls(
            head=data.get("head", "UNKNOWN"),
            equity=data.get("equity", 0.0),
            max_equity=data.get("max_equity", 0.0),
            drawdown=data.get("drawdown", 0.0),
            realized_pnl=data.get("realized_pnl", 0.0),
            unrealized_pnl=data.get("unrealized_pnl", 0.0),
            trades=data.get("trades", 0),
            wins=data.get("wins", 0),
            win_rate=data.get("win_rate", 0.0),
            total_R=data.get("total_R", 0.0),
            trades_with_R=data.get("trades_with_R", 0),
            avg_R=data.get("avg_R", 0.0),
            gross_exposure=data.get("gross_exposure", 0.0),
            veto_count=data.get("veto_count", 0),
            last_trade_ts=data.get("last_trade_ts", 0.0),
            last_active_ts=data.get("last_active_ts", 0.0),
            kill_switch_active=data.get("kill_switch_active", False),
            cooldown_remaining=data.get("cooldown_remaining", 0),
            throttle_scale=data.get("throttle_scale", 1.0),
        )


@dataclass
class HydraPnlState:
    """Full state of the Hydra PnL engine."""

    updated_ts: str = ""
    heads: Dict[str, HeadPnlStats] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure all heads exist."""
        for head in STRATEGY_HEADS:
            if head not in self.heads:
                self.heads[head] = HeadPnlStats(head=head)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "updated_ts": self.updated_ts,
            "heads": {h: stats.to_dict() for h, stats in self.heads.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HydraPnlState":
        """Create from dictionary."""
        heads = {}
        heads_raw = data.get("heads", {})
        for head_name in STRATEGY_HEADS:
            if head_name in heads_raw:
                heads[head_name] = HeadPnlStats.from_dict(heads_raw[head_name])
            else:
                heads[head_name] = HeadPnlStats(head=head_name)
        return cls(
            updated_ts=data.get("updated_ts", ""),
            heads=heads,
            meta=data.get("meta", {}),
        )

    def get_total_realized_pnl(self) -> float:
        """Get total realized PnL across all heads."""
        return sum(h.realized_pnl for h in self.heads.values())

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized PnL across all heads."""
        return sum(h.unrealized_pnl for h in self.heads.values())

    def get_total_equity(self) -> float:
        """Get total equity across all heads."""
        return sum(h.equity for h in self.heads.values())

    def get_max_head_drawdown(self) -> float:
        """Get maximum drawdown across all heads."""
        return max((h.drawdown for h in self.heads.values()), default=0.0)

    def get_worst_heads(self, n: int = 3) -> List[str]:
        """Get heads with worst drawdown."""
        sorted_heads = sorted(self.heads.values(), key=lambda h: h.drawdown, reverse=True)
        return [h.head for h in sorted_heads[:n]]

    def get_best_heads(self, n: int = 3) -> List[str]:
        """Get heads with best equity."""
        sorted_heads = sorted(self.heads.values(), key=lambda h: h.equity, reverse=True)
        return [h.head for h in sorted_heads[:n]]


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------


def load_hydra_pnl_config(
    config_path: Path | str | None = None,
    strategy_config: Mapping[str, Any] | None = None,
) -> HydraPnlConfig:
    """
    Load Hydra PnL configuration from strategy_config.json.

    Args:
        config_path: Path to strategy_config.json
        strategy_config: Pre-loaded strategy config dict

    Returns:
        HydraPnlConfig instance
    """
    if strategy_config is None:
        cfg_path = Path(config_path or DEFAULT_CONFIG_PATH)
        if cfg_path.exists():
            try:
                strategy_config = json.loads(cfg_path.read_text())
            except (json.JSONDecodeError, IOError):
                return HydraPnlConfig()
        else:
            return HydraPnlConfig()

    pnl_cfg = strategy_config.get("hydra_pnl", {})
    if not pnl_cfg:
        return HydraPnlConfig()

    # Parse kill switch config
    ks_raw = pnl_cfg.get("kill_switch", {})
    kill_switch = KillSwitchConfig(
        max_drawdown=ks_raw.get("max_drawdown", DEFAULT_MAX_DRAWDOWN),
        min_win_rate=ks_raw.get("min_win_rate", DEFAULT_MIN_WIN_RATE),
        min_R_multiple=ks_raw.get("min_R_multiple", DEFAULT_MIN_R_MULTIPLE),
        cooldown_cycles=ks_raw.get("cooldown_cycles", DEFAULT_COOLDOWN_CYCLES),
    )

    # Parse throttling config
    th_raw = pnl_cfg.get("throttling", {})
    throttling = ThrottlingConfig(
        dd_soft_threshold=th_raw.get("dd_soft_threshold", DEFAULT_DD_SOFT_THRESHOLD),
        dd_hard_threshold=th_raw.get("dd_hard_threshold", DEFAULT_DD_HARD_THRESHOLD),
        soft_scale=th_raw.get("soft_scale", DEFAULT_SOFT_SCALE),
        hard_scale=th_raw.get("hard_scale", DEFAULT_HARD_SCALE),
    )

    return HydraPnlConfig(
        enabled=pnl_cfg.get("enabled", False),
        lookback_days=pnl_cfg.get("lookback_days", DEFAULT_LOOKBACK_DAYS),
        min_trades_for_stats=pnl_cfg.get("min_trades_for_stats", DEFAULT_MIN_TRADES),
        kill_switch=kill_switch,
        throttling=throttling,
    )


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def load_hydra_pnl_state(state_path: Path | str | None = None) -> HydraPnlState:
    """
    Load Hydra PnL state from disk.

    Args:
        state_path: Path to hydra_pnl.json

    Returns:
        HydraPnlState instance (empty if file missing/invalid)
    """
    path = Path(state_path or DEFAULT_STATE_PATH)
    if not path.exists():
        return HydraPnlState()

    try:
        data = json.loads(path.read_text())
        return HydraPnlState.from_dict(data)
    except (json.JSONDecodeError, IOError) as e:
        _LOG.warning("Failed to load hydra_pnl state: %s", e)
        return HydraPnlState()


def save_hydra_pnl_state(
    state: HydraPnlState,
    state_path: Path | str | None = None,
) -> bool:
    """
    Save Hydra PnL state to disk.

    Args:
        state: HydraPnlState to save
        state_path: Path to hydra_pnl.json

    Returns:
        True if successful
    """
    path = Path(state_path or DEFAULT_STATE_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2))
        return True
    except IOError as e:
        _LOG.error("Failed to save hydra_pnl state: %s", e)
        return False


# ---------------------------------------------------------------------------
# Event Logging
# ---------------------------------------------------------------------------


def log_pnl_event(
    head: str,
    event: str,
    reason: str,
    log_path: Path | str | None = None,
    extra: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Log a PnL event (kill switch on/off, throttle change, etc).

    Args:
        head: Head name
        event: Event type (KILL_SWITCH_ON, KILL_SWITCH_OFF, THROTTLE_CHANGE, etc)
        reason: Reason for the event
        log_path: Path to event log
        extra: Additional fields to include

    Returns:
        True if successful
    """
    path = Path(log_path or DEFAULT_EVENT_LOG_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "head": head,
            "event": event,
            "reason": reason,
        }
        if extra:
            entry.update(extra)
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return True
    except IOError as e:
        _LOG.error("Failed to log pnl event: %s", e)
        return False


# ---------------------------------------------------------------------------
# PnL Attribution
# ---------------------------------------------------------------------------


def attribute_fill_pnl(
    fill_pnl: float,
    fill_fee: float,
    head_contributions: Dict[str, float],
    state: HydraPnlState,
    R_multiple: Optional[float] = None,
) -> Dict[str, float]:
    """
    Attribute a fill's PnL and fees to heads based on contributions.

    Args:
        fill_pnl: PnL from the fill (after fees)
        fill_fee: Fee paid for the fill
        head_contributions: Dict of head -> weight (should sum to ~1.0)
        state: Current HydraPnlState to update
        R_multiple: Optional R-multiple for the trade

    Returns:
        Dict of head -> attributed PnL
    """
    attributed: Dict[str, float] = {}

    # Normalize contributions if needed
    total_weight = sum(head_contributions.values())
    if total_weight <= 0:
        return attributed

    for head, weight in head_contributions.items():
        if head not in state.heads:
            continue

        normalized_weight = weight / total_weight
        head_pnl = fill_pnl * normalized_weight
        head_R = R_multiple * normalized_weight if R_multiple is not None else None

        # Record the trade for this head
        state.heads[head].record_trade(head_pnl, head_R)
        attributed[head] = head_pnl

    return attributed


def update_unrealized_pnl(
    positions: List[Dict[str, Any]],
    head_contributions_by_symbol: Dict[str, Dict[str, float]],
    state: HydraPnlState,
) -> None:
    """
    Update unrealized PnL per head from current positions.

    Args:
        positions: List of position dicts with symbol, unrealized_pnl
        head_contributions_by_symbol: symbol -> head_contributions dict
        state: HydraPnlState to update
    """
    # Reset unrealized PnL for all heads
    for head_stats in state.heads.values():
        head_stats.unrealized_pnl = 0.0

    # Attribute unrealized PnL by position
    for pos in positions:
        symbol = pos.get("symbol", "")
        unrealized = pos.get("unrealized_pnl", 0.0)

        contributions = head_contributions_by_symbol.get(symbol, {})
        if not contributions:
            # Default: attribute to TREND if no specific contributions
            contributions = {"TREND": 1.0}

        total_weight = sum(contributions.values())
        if total_weight <= 0:
            continue

        for head, weight in contributions.items():
            if head not in state.heads:
                continue
            normalized = weight / total_weight
            state.heads[head].unrealized_pnl += unrealized * normalized

    # Update equity for all heads
    for head_stats in state.heads.values():
        head_stats.update_equity()


def update_gross_exposure(
    positions: List[Dict[str, Any]],
    head_contributions_by_symbol: Dict[str, Dict[str, float]],
    state: HydraPnlState,
    nav_usd: float,
) -> None:
    """
    Update gross exposure per head from current positions.

    Args:
        positions: List of position dicts with symbol, notional_usd
        head_contributions_by_symbol: symbol -> head_contributions dict
        state: HydraPnlState to update
        nav_usd: Current NAV in USD
    """
    if nav_usd <= 0:
        return

    # Reset exposure for all heads
    for head_stats in state.heads.values():
        head_stats.gross_exposure = 0.0

    # Attribute exposure by position
    for pos in positions:
        symbol = pos.get("symbol", "")
        notional = abs(pos.get("notional_usd", 0.0))

        contributions = head_contributions_by_symbol.get(symbol, {})
        if not contributions:
            contributions = {"TREND": 1.0}

        total_weight = sum(contributions.values())
        if total_weight <= 0:
            continue

        for head, weight in contributions.items():
            if head not in state.heads:
                continue
            normalized = weight / total_weight
            state.heads[head].gross_exposure += (notional * normalized) / nav_usd


# ---------------------------------------------------------------------------
# Kill Switch Logic
# ---------------------------------------------------------------------------


def check_kill_switch(
    head_stats: HeadPnlStats,
    cfg: HydraPnlConfig,
    log_path: Optional[Path] = None,
) -> bool:
    """
    Check if kill switch should be activated for a head.

    Args:
        head_stats: HeadPnlStats for the head
        cfg: HydraPnlConfig with kill switch settings
        log_path: Path to log events

    Returns:
        True if kill switch was activated or remains active
    """
    ks = cfg.kill_switch

    # Check if already active with cooldown
    if head_stats.kill_switch_active:
        if head_stats.cooldown_remaining > 0:
            head_stats.cooldown_remaining -= 1
            if head_stats.cooldown_remaining == 0:
                # Cooldown complete, deactivate
                head_stats.kill_switch_active = False
                log_pnl_event(
                    head=head_stats.head,
                    event="KILL_SWITCH_OFF",
                    reason="Cooldown completed",
                    log_path=log_path,
                )
            return head_stats.kill_switch_active
        else:
            # Active but no cooldown - auto-clear (config choice)
            head_stats.kill_switch_active = False
            return False

    # Need minimum trades to evaluate
    if head_stats.trades < cfg.min_trades_for_stats:
        return False

    # Check conditions
    reasons = []

    if head_stats.drawdown > ks.max_drawdown:
        reasons.append(f"drawdown {head_stats.drawdown:.2%} > {ks.max_drawdown:.2%}")

    if head_stats.win_rate < ks.min_win_rate:
        reasons.append(f"win_rate {head_stats.win_rate:.2%} < {ks.min_win_rate:.2%}")

    if head_stats.trades_with_R > 0 and head_stats.avg_R < ks.min_R_multiple:
        reasons.append(f"avg_R {head_stats.avg_R:.2f} < {ks.min_R_multiple:.2f}")

    if reasons:
        head_stats.kill_switch_active = True
        head_stats.cooldown_remaining = ks.cooldown_cycles
        log_pnl_event(
            head=head_stats.head,
            event="KILL_SWITCH_ON",
            reason="; ".join(reasons),
            log_path=log_path,
            extra={"cooldown_cycles": ks.cooldown_cycles},
        )
        return True

    return False


def evaluate_all_kill_switches(
    state: HydraPnlState,
    cfg: HydraPnlConfig,
    log_path: Optional[Path] = None,
) -> Dict[str, bool]:
    """
    Evaluate kill switches for all heads.

    Args:
        state: HydraPnlState to evaluate
        cfg: HydraPnlConfig with settings
        log_path: Path to log events

    Returns:
        Dict of head -> kill_switch_active
    """
    result = {}
    for head, stats in state.heads.items():
        result[head] = check_kill_switch(stats, cfg, log_path)
    return result


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


def compute_throttle_scale(
    drawdown: float,
    cfg: ThrottlingConfig,
) -> float:
    """
    Compute throttle scale based on drawdown.

    Args:
        drawdown: Current drawdown (0-1)
        cfg: ThrottlingConfig

    Returns:
        Scale factor 0-1
    """
    if drawdown < cfg.dd_soft_threshold:
        return 1.0
    elif drawdown < cfg.dd_hard_threshold:
        # Linear interpolation between soft and hard
        k = (drawdown - cfg.dd_soft_threshold) / (cfg.dd_hard_threshold - cfg.dd_soft_threshold)
        return 1.0 - k * (1.0 - cfg.soft_scale)
    else:
        return cfg.hard_scale


def update_throttle_scales(
    state: HydraPnlState,
    cfg: HydraPnlConfig,
    log_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Update throttle scales for all heads based on drawdown.

    Args:
        state: HydraPnlState to update
        cfg: HydraPnlConfig with settings
        log_path: Path to log events

    Returns:
        Dict of head -> throttle_scale
    """
    scales = {}
    for head, stats in state.heads.items():
        if stats.kill_switch_active:
            new_scale = 0.0
        else:
            new_scale = compute_throttle_scale(stats.drawdown, cfg.throttling)

        # Log significant scale changes
        old_scale = stats.throttle_scale
        if abs(new_scale - old_scale) > 0.1:
            log_pnl_event(
                head=head,
                event="THROTTLE_CHANGE",
                reason=f"Scale {old_scale:.2f} -> {new_scale:.2f} (dd={stats.drawdown:.2%})",
                log_path=log_path,
            )

        stats.throttle_scale = new_scale
        scales[head] = new_scale

    return scales


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------


def get_head_throttle_scales(
    state: HydraPnlState | None = None,
) -> Dict[str, float]:
    """
    Get current throttle scales for all heads.

    Args:
        state: HydraPnlState (loads from disk if None)

    Returns:
        Dict of head -> throttle_scale (1.0 = full, 0.0 = disabled)
    """
    if state is None:
        state = load_hydra_pnl_state()

    return {h: stats.throttle_scale for h, stats in state.heads.items()}


def get_head_kill_switch_status(
    state: HydraPnlState | None = None,
) -> Dict[str, bool]:
    """
    Get kill switch status for all heads.

    Args:
        state: HydraPnlState (loads from disk if None)

    Returns:
        Dict of head -> kill_switch_active
    """
    if state is None:
        state = load_hydra_pnl_state()

    return {h: stats.kill_switch_active for h, stats in state.heads.items()}


def is_head_active(
    head: str,
    state: HydraPnlState | None = None,
) -> bool:
    """
    Check if a head is currently active (not killed).

    Args:
        head: Head name
        state: HydraPnlState (loads from disk if None)

    Returns:
        True if head is active
    """
    if state is None:
        state = load_hydra_pnl_state()

    if head not in state.heads:
        return True  # Unknown heads default to active

    return not state.heads[head].kill_switch_active


def get_head_stats_summary(
    state: HydraPnlState | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Get summary stats for all heads.

    Args:
        state: HydraPnlState (loads from disk if None)

    Returns:
        Dict of head -> summary dict
    """
    if state is None:
        state = load_hydra_pnl_state()

    return {
        h: {
            "equity": stats.equity,
            "drawdown": stats.drawdown,
            "win_rate": stats.win_rate,
            "trades": stats.trades,
            "throttle_scale": stats.throttle_scale,
            "kill_switch_active": stats.kill_switch_active,
        }
        for h, stats in state.heads.items()
    }


def get_hydra_pnl_summary(
    state: HydraPnlState | None = None,
) -> Dict[str, Any]:
    """
    Get summary for edge insights integration.

    Args:
        state: HydraPnlState (loads from disk if None)

    Returns:
        Summary dict for edge_insights
    """
    if state is None:
        state = load_hydra_pnl_state()

    return {
        "worst_heads": state.get_worst_heads(3),
        "best_heads": state.get_best_heads(3),
        "max_head_dd": state.get_max_head_drawdown(),
        "total_realized_pnl": state.get_total_realized_pnl(),
        "total_unrealized_pnl": state.get_total_unrealized_pnl(),
        "heads_killed": sum(1 for h in state.heads.values() if h.kill_switch_active),
    }


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------


def run_hydra_pnl_step(
    cfg: HydraPnlConfig,
    fills: Optional[List[Dict[str, Any]]] = None,
    positions: Optional[List[Dict[str, Any]]] = None,
    head_contributions_by_symbol: Optional[Dict[str, Dict[str, float]]] = None,
    nav_usd: float = 0.0,
    state_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> HydraPnlState:
    """
    Run one Hydra PnL cycle: attribute PnL, update stats, evaluate kill switches.

    Args:
        cfg: HydraPnlConfig
        fills: List of new fill dicts with pnl, fee, head_contributions
        positions: List of current position dicts
        head_contributions_by_symbol: symbol -> head_contributions for positions
        nav_usd: Current NAV in USD
        state_path: Path to save state
        log_path: Path to log events

    Returns:
        Updated HydraPnlState
    """
    if not cfg.enabled:
        return HydraPnlState(
            updated_ts=datetime.now(timezone.utc).isoformat(),
            meta={"enabled": False},
        )

    # Load current state
    state = load_hydra_pnl_state(state_path)

    # Process new fills
    fills = fills or []
    for fill in fills:
        fill_pnl = fill.get("pnl", 0.0)
        fill_fee = fill.get("fee", 0.0)
        contributions = fill.get("head_contributions", {})
        R_multiple = fill.get("R_multiple")

        if contributions:
            attribute_fill_pnl(fill_pnl, fill_fee, contributions, state, R_multiple)

    # Update unrealized PnL from positions
    positions = positions or []
    head_contributions_by_symbol = head_contributions_by_symbol or {}
    update_unrealized_pnl(positions, head_contributions_by_symbol, state)

    # Update gross exposure
    update_gross_exposure(positions, head_contributions_by_symbol, state, nav_usd)

    # Evaluate kill switches
    evaluate_all_kill_switches(state, cfg, log_path)

    # Update throttle scales
    update_throttle_scales(state, cfg, log_path)

    # Update metadata
    state.updated_ts = datetime.now(timezone.utc).isoformat()
    state.meta = {
        "lookback_days": cfg.lookback_days,
        "total_realized_pnl": state.get_total_realized_pnl(),
        "total_unrealized_pnl": state.get_total_unrealized_pnl(),
        "total_equity": state.get_total_equity(),
        "max_head_drawdown": state.get_max_head_drawdown(),
        "heads_killed": sum(1 for h in state.heads.values() if h.kill_switch_active),
    }

    # Save state
    if state_path is not None or DEFAULT_STATE_PATH.parent.exists():
        save_hydra_pnl_state(state, state_path)

    return state


# ---------------------------------------------------------------------------
# Cerberus Integration
# ---------------------------------------------------------------------------


def apply_pnl_throttle_to_cerberus(
    cerberus_multipliers: Dict[str, float],
    pnl_state: HydraPnlState | None = None,
) -> Dict[str, float]:
    """
    Apply Hydra PnL throttle scales to Cerberus multipliers.

    Args:
        cerberus_multipliers: Current Cerberus head multipliers
        pnl_state: HydraPnlState (loads from disk if None)

    Returns:
        Adjusted multipliers with throttle applied
    """
    if pnl_state is None:
        pnl_state = load_hydra_pnl_state()

    adjusted = {}
    for head, mult in cerberus_multipliers.items():
        if head in pnl_state.heads:
            scale = pnl_state.heads[head].throttle_scale
            adjusted[head] = mult * scale
        else:
            adjusted[head] = mult

    return adjusted


# ---------------------------------------------------------------------------
# Hydra Integration
# ---------------------------------------------------------------------------


def apply_pnl_throttle_to_hydra_budgets(
    head_budgets: Dict[str, float],
    pnl_state: HydraPnlState | None = None,
) -> Dict[str, float]:
    """
    Apply Hydra PnL throttle scales to Hydra NAV budgets.

    Args:
        head_budgets: Original per-head NAV budgets
        pnl_state: HydraPnlState (loads from disk if None)

    Returns:
        Adjusted budgets with throttle applied
    """
    if pnl_state is None:
        pnl_state = load_hydra_pnl_state()

    adjusted = {}
    for head, budget in head_budgets.items():
        if head in pnl_state.heads:
            scale = pnl_state.heads[head].throttle_scale
            adjusted[head] = budget * scale
        else:
            adjusted[head] = budget

    return adjusted


def is_hydra_pnl_enabled(
    strategy_config: Mapping[str, Any] | None = None,
) -> bool:
    """
    Check if Hydra PnL is enabled in config.

    Args:
        strategy_config: Pre-loaded strategy config

    Returns:
        True if enabled
    """
    cfg = load_hydra_pnl_config(strategy_config=strategy_config)
    return cfg.enabled


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "STRATEGY_HEADS",
    "DEFAULT_STATE_PATH",
    "DEFAULT_EVENT_LOG_PATH",
    # Data classes
    "KillSwitchConfig",
    "ThrottlingConfig",
    "HydraPnlConfig",
    "HeadPnlStats",
    "HydraPnlState",
    # Config loading
    "load_hydra_pnl_config",
    # State I/O
    "load_hydra_pnl_state",
    "save_hydra_pnl_state",
    # Event logging
    "log_pnl_event",
    # PnL Attribution
    "attribute_fill_pnl",
    "update_unrealized_pnl",
    "update_gross_exposure",
    # Kill switch
    "check_kill_switch",
    "evaluate_all_kill_switches",
    # Throttling
    "compute_throttle_scale",
    "update_throttle_scales",
    # Integration helpers
    "get_head_throttle_scales",
    "get_head_kill_switch_status",
    "is_head_active",
    "get_head_stats_summary",
    "get_hydra_pnl_summary",
    # Pipeline
    "run_hydra_pnl_step",
    # Cerberus integration
    "apply_pnl_throttle_to_cerberus",
    # Hydra integration
    "apply_pnl_throttle_to_hydra_budgets",
    "is_hydra_pnl_enabled",
]
