"""
Volatility Target Strategy (v7.3-alpha)

A volatility-targeting strategy that scales position sizes based on ATR
to maintain consistent risk per trade across different volatility regimes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from execution.exchange_utils import get_klines
except Exception:  # pragma: no cover - fallback for tests without exchange client
    get_klines = None


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


@dataclass
class TrendConfig:
    htf_tf: str = "1h"
    fast_ema: int = 21
    slow_ema: int = 50
    min_trend_strength: float = 0.1
    use_htf_rsi_filter: bool = True
    rsi_overbought: int = 70
    rsi_oversold: int = 30


@dataclass
class CarryConfig:
    use_funding: bool = True
    min_funding_annualized: float = 0.0
    max_funding_annualized: float = 0.5
    funding_weight: float = 0.3
    use_basis: bool = False
    max_basis_pct: float = 0.1
    basis_weight: float = 0.2


@dataclass
class VolTargetConfig:
    """Configuration for the volatility target strategy."""
    enabled: bool
    base_per_trade_nav_pct: float
    min_per_trade_nav_pct: float
    max_per_trade_nav_pct: float
    target_vol: float
    min_vol: float
    max_vol: float
    min_vol_factor: float
    max_vol_factor: float
    atr_lookback: int
    use_atr_percentiles: bool
    require_trend_alignment: bool
    max_dd_regime: int
    max_risk_mode: str
    min_signal_score: float
    # TP/SL fields (v7.3-alpha1)
    sl_atr_mult: float
    tp_atr_mult: float
    min_rr: float
    side_mode: str
    enable_tp_sl: bool
    trend: TrendConfig = field(default_factory=TrendConfig)
    carry: CarryConfig = field(default_factory=CarryConfig)


RISK_MODE_ORDER = ["OK", "WARN", "DEFENSIVE", "HALTED"]


def _risk_mode_allowed(current: str, max_mode: str) -> bool:
    """Check if current risk mode is at or below the max allowed mode."""
    try:
        return RISK_MODE_ORDER.index(current.upper()) <= RISK_MODE_ORDER.index(max_mode.upper())
    except ValueError:
        # Unknown risk mode: be conservative
        return False


def compute_vol_factor(atr_value: float, price: float, cfg: VolTargetConfig) -> Optional[float]:
    """
    Compute the volatility scaling factor based on ATR.
    
    Returns factor > 1 for low vol (scale up), factor < 1 for high vol (scale down).
    Returns None if inputs are invalid.
    """
    if price <= 0:
        return None
    if atr_value <= 0:
        # no volatility estimate → skip safely
        return None

    vol = atr_value / price  # simple proxy for per-day vol (assuming ATR is daily)
    # Clamp vol into [min_vol, max_vol] to avoid degenerate factors
    vol_clamped = max(cfg.min_vol, min(vol, cfg.max_vol))

    # target_vol / current_vol → if vol is high → factor < 1, if vol low → factor > 1
    vol_factor = cfg.target_vol / vol_clamped

    # Clamp factor
    vol_factor = max(cfg.min_vol_factor, min(vol_factor, cfg.max_vol_factor))
    return vol_factor


def compute_per_trade_nav_pct(nav_pct_base: float, vol_factor: float, cfg: VolTargetConfig) -> float:
    """
    Scale base NAV percentage by vol_factor and clamp to config min/max.
    """
    # Scale base percentage by vol_factor and clamp to config min/max
    raw = nav_pct_base * vol_factor
    return max(cfg.min_per_trade_nav_pct, min(raw, cfg.max_per_trade_nav_pct))


def compute_tp_sl_prices(
    price: float,
    atr_value: float,
    side: str,
    cfg: VolTargetConfig,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute take-profit and stop-loss prices using ATR multiples.
    
    Args:
        price: Current entry price
        atr_value: ATR value in price units
        side: Trade side ("BUY" or "SELL")
        cfg: VolTargetConfig with TP/SL parameters
        
    Returns:
        Tuple of (tp_price, sl_price, reward_risk) or None if disabled or invalid.
    """
    if not cfg.enable_tp_sl:
        return None

    if price <= 0 or atr_value is None or atr_value <= 0:
        return None

    # Distance in price terms
    sl_dist = cfg.sl_atr_mult * atr_value
    tp_dist = cfg.tp_atr_mult * atr_value

    if sl_dist <= 0 or tp_dist <= 0:
        return None

    rr = tp_dist / sl_dist
    if rr < cfg.min_rr:
        # reward:risk too low → skip TP/SL to avoid bad profiles
        return None

    if side.upper() == "BUY":
        sl_price = price - sl_dist
        tp_price = price + tp_dist
    elif side.upper() == "SELL":
        sl_price = price + sl_dist
        tp_price = price - tp_dist
    else:
        return None

    # Avoid nonsense levels
    if sl_price <= 0 or tp_price <= 0:
        return None

    return tp_price, sl_price, rr


def _ema(values: Sequence[float], period: int) -> Optional[float]:
    if period <= 0 or not values or len(values) < period:
        return None
    k = 2.0 / (period + 1.0)
    ema_val = float(values[0])
    for v in values[1:]:
        ema_val = v * k + ema_val * (1.0 - k)
    return ema_val


def _compute_rsi(closes: Sequence[float], period: int = 14) -> Optional[float]:
    if not closes or len(closes) <= period:
        return None
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(delta))
    avg_gain = sum(gains[-period:]) / float(period)
    avg_loss = sum(losses[-period:]) / float(period)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_trend_bias(
    htf_closes: Sequence[float],
    htf_rsi: Optional[float],
    cfg: TrendConfig,
) -> Dict[str, Any]:
    if not htf_closes:
        return {"direction": "FLAT", "strength": 0.0, "ema_fast": None, "ema_slow": None, "rsi": htf_rsi}

    ema_fast = _ema(htf_closes, cfg.fast_ema)
    ema_slow = _ema(htf_closes, cfg.slow_ema)
    if ema_fast is None or ema_slow is None:
        return {"direction": "FLAT", "strength": 0.0, "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": htf_rsi}

    diff = ema_fast - ema_slow
    strength = 0.0
    direction = "FLAT"
    base = abs(ema_slow) if ema_slow != 0 else 1.0
    strength = min(1.0, max(0.0, abs(diff) / base))

    if diff > 0:
        direction = "LONG"
    elif diff < 0:
        direction = "SHORT"

    if cfg.use_htf_rsi_filter and htf_rsi is not None:
        if direction == "LONG" and htf_rsi >= cfg.rsi_overbought:
            strength *= 0.5
        if direction == "SHORT" and htf_rsi <= cfg.rsi_oversold:
            strength *= 0.5

    if strength < cfg.min_trend_strength:
        direction = "FLAT"

    return {"direction": direction, "strength": strength, "ema_fast": ema_fast, "ema_slow": ema_slow, "rsi": htf_rsi}


def compute_carry_bias(
    *,
    funding_annualized: Optional[float],
    basis_pct: Optional[float],
    cfg: CarryConfig,
) -> Dict[str, Any]:
    score_long = 0.0
    score_short = 0.0

    funding_cap = cfg.max_funding_annualized or 1.0
    basis_cap = cfg.max_basis_pct or 1.0

    if cfg.use_funding and funding_annualized is not None and funding_cap > 0:
        f = max(-funding_cap, min(funding_cap, funding_annualized))
        if abs(f) >= cfg.min_funding_annualized:
            if f < 0:
                score_long += cfg.funding_weight * abs(f) / funding_cap
            elif f > 0:
                score_short += cfg.funding_weight * abs(f) / funding_cap

    if cfg.use_basis and basis_pct is not None and basis_cap > 0:
        b = max(-basis_cap, min(basis_cap, basis_pct))
        if b > 0:
            score_short += cfg.basis_weight * (b / basis_cap)
        elif b < 0:
            score_long += cfg.basis_weight * (abs(b) / basis_cap)

    return {
        "score_long": min(1.0, score_long),
        "score_short": min(1.0, score_short),
        "funding_annualized": funding_annualized,
        "basis_pct": basis_pct,
    }


def decide_hybrid_side(
    trend_info: Dict[str, Any],
    carry_info: Dict[str, Any],
    cfg: VolTargetConfig,
) -> Dict[str, Any]:
    trend_dir = trend_info.get("direction", "FLAT")
    trend_strength = float(trend_info.get("strength", 0.0) or 0.0)

    carry_long = float(carry_info.get("score_long", 0.0) or 0.0)
    carry_short = float(carry_info.get("score_short", 0.0) or 0.0)

    long_score = 0.0
    short_score = 0.0

    if trend_dir == "LONG":
        long_score += trend_strength
    elif trend_dir == "SHORT":
        short_score += trend_strength

    long_score += carry_long
    short_score += carry_short

    max_score = max(long_score, short_score, 0.0)
    hybrid_score = max_score if max_score <= 1.0 else (max_score / (max_score or 1.0))

    if max_score <= 0.0:
        side = "NONE"
    elif long_score > short_score:
        side = "BUY"
    elif short_score > long_score:
        side = "SELL"
    else:
        side = "NONE"

    if cfg.side_mode == "long_only" and side == "SELL":
        side = "NONE"
    if cfg.require_trend_alignment and trend_dir == "FLAT":
        side = "NONE"

    return {
        "side": side,
        "hybrid_score": hybrid_score,
        "components": {
            "trend_dir": trend_dir,
            "trend_strength": trend_strength,
            "carry_long": carry_long,
            "carry_short": carry_short,
        },
    }


def load_htf_trend_data(symbol: str, timeframe: str, fast: int, slow: int) -> Tuple[List[float], Optional[float]]:
    """Load HTF closes and RSI for trend computation; defensive fallback if data unavailable."""
    closes: List[float] = []
    rsi_val: Optional[float] = None
    if get_klines is None:
        return closes, rsi_val
    try:
        need = max(fast, slow, 50)
        rows = get_klines(symbol, timeframe, limit=need)
        closes = [row[4] for row in rows if len(row) >= 5]
        rsi_val = _compute_rsi(closes, period=14)
    except Exception:
        return [], None
    return closes, rsi_val


def load_carry_inputs(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Fetch funding/basis inputs if available. For now, return (None, None) as a safe default.
    """
    try:
        symbol = str(symbol or "").upper()
    except Exception:
        symbol = ""
    # Placeholder: integrate funding/basis sources when available.
    return None, None


def _fallback_trend(
    trend: str,
    trend_aligned: bool,
    min_strength: float,
) -> Tuple[Optional[str], float]:
    """
    Map legacy trend params into a LONG/SHORT/FLAT directional hint.
    """
    trend_key = str(trend or "").upper()
    if not trend_aligned:
        return None, 0.0
    if trend_key == "BULL":
        return "LONG", max(0.0, min_strength)
    if trend_key == "BEAR":
        return "SHORT", max(0.0, min_strength)
    return "LONG", max(0.0, min_strength)  # default bias to long like prior behavior


def generate_vol_target_intent(
    symbol: str,
    timeframe: str,
    price: float,
    nav: float,
    atr_value: float,
    regimes_snapshot: Dict[str, Any],
    risk_snapshot: Dict[str, Any],
    trend: str,
    trend_aligned: bool,
    strategy_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Build a volatility-targeted intent for a symbol/timeframe, or return None if not allowed.
    
    This does not apply risk vetoes or adaptive sizing/weighting — those happen later
    in the pipeline via check_order().
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Timeframe string (e.g., "15m")
        price: Current price
        nav: Current NAV in USDT
        atr_value: ATR value in price units (not percentage)
        regimes_snapshot: Dict with atr_regime, dd_regime keys
        risk_snapshot: Dict with risk_mode key
        trend: Trend direction ("BULL", "BEAR", "NEUTRAL")
        trend_aligned: Whether signal is aligned with trend
        strategy_cfg: Vol target strategy configuration dict
        
    Returns:
        Intent dict if conditions allow, None otherwise
    """

    if nav <= 0 or price <= 0:
        return None

    trend_cfg_raw = strategy_cfg.get("trend", {}) if isinstance(strategy_cfg, Dict) else {}
    carry_cfg_raw = strategy_cfg.get("carry", {}) if isinstance(strategy_cfg, Dict) else {}
    trend_cfg = TrendConfig(
        htf_tf=trend_cfg_raw.get("htf_tf", "1h"),
        fast_ema=int(trend_cfg_raw.get("fast_ema", 21)),
        slow_ema=int(trend_cfg_raw.get("slow_ema", 50)),
        min_trend_strength=float(trend_cfg_raw.get("min_trend_strength", 0.1)),
        use_htf_rsi_filter=bool(trend_cfg_raw.get("use_htf_rsi_filter", True)),
        rsi_overbought=int(trend_cfg_raw.get("rsi_overbought", 70)),
        rsi_oversold=int(trend_cfg_raw.get("rsi_oversold", 30)),
    )
    carry_cfg = CarryConfig(
        use_funding=bool(carry_cfg_raw.get("use_funding", True)),
        min_funding_annualized=float(carry_cfg_raw.get("min_funding_annualized", 0.0)),
        max_funding_annualized=float(carry_cfg_raw.get("max_funding_annualized", 0.5)),
        funding_weight=float(carry_cfg_raw.get("funding_weight", 0.3)),
        use_basis=bool(carry_cfg_raw.get("use_basis", False)),
        max_basis_pct=float(carry_cfg_raw.get("max_basis_pct", 0.1)),
        basis_weight=float(carry_cfg_raw.get("basis_weight", 0.2)),
    )

    cfg = VolTargetConfig(
        enabled=strategy_cfg.get("enabled", True),
        base_per_trade_nav_pct=strategy_cfg.get("base_per_trade_nav_pct", 0.015),
        min_per_trade_nav_pct=strategy_cfg.get("min_per_trade_nav_pct", 0.005),
        max_per_trade_nav_pct=strategy_cfg.get("max_per_trade_nav_pct", 0.03),
        target_vol=strategy_cfg.get("target_vol", 0.015),
        min_vol=strategy_cfg.get("min_vol", 0.003),
        max_vol=strategy_cfg.get("max_vol", 0.08),
        min_vol_factor=strategy_cfg.get("min_vol_factor", 0.25),
        max_vol_factor=strategy_cfg.get("max_vol_factor", 2.0),
        atr_lookback=strategy_cfg.get("atr_lookback", 14),
        use_atr_percentiles=strategy_cfg.get("use_atr_percentiles", True),
        require_trend_alignment=strategy_cfg.get("require_trend_alignment", True),
        max_dd_regime=strategy_cfg.get("max_dd_regime", 2),
        max_risk_mode=strategy_cfg.get("max_risk_mode", "DEFENSIVE"),
        min_signal_score=strategy_cfg.get("min_signal_score", 0.0),
        # TP/SL fields (v7.3-alpha1)
        sl_atr_mult=strategy_cfg.get("sl_atr_mult", 2.0),
        tp_atr_mult=strategy_cfg.get("tp_atr_mult", 3.0),
        min_rr=strategy_cfg.get("min_rr", 1.2),
        side_mode=strategy_cfg.get("side_mode", "trend"),
        enable_tp_sl=strategy_cfg.get("enable_tp_sl", True),
        trend=trend_cfg,
        carry=carry_cfg,
    )

    if not cfg.enabled:
        return None

    # Risk mode / DD regime gating
    risk_mode = risk_snapshot.get("risk_mode", "OK")
    dd_regime = regimes_snapshot.get("dd_regime", 0)
    atr_regime = regimes_snapshot.get("atr_regime", 1)

    if not _risk_mode_allowed(risk_mode, cfg.max_risk_mode):
        # HALTED or too defensive
        return None

    if dd_regime > cfg.max_dd_regime:
        # Too deep in DD to allocate new risk
        return None

    # Higher timeframe trend + carry view
    htf_closes, htf_rsi = load_htf_trend_data(symbol, cfg.trend.htf_tf, cfg.trend.fast_ema, cfg.trend.slow_ema)
    trend_info = compute_trend_bias(htf_closes=htf_closes, htf_rsi=htf_rsi, cfg=cfg.trend)
    if trend_info["direction"] == "FLAT":
        # fall back to legacy trend hint if available
        legacy_dir, legacy_strength = _fallback_trend(trend, trend_aligned, cfg.trend.min_trend_strength)
        if legacy_dir:
            trend_info["direction"] = legacy_dir
            trend_info["strength"] = legacy_strength
    if trend_info["direction"] == "FLAT" and not cfg.require_trend_alignment:
        # maintain prior long-only bias when trend gating disabled
        trend_info["direction"] = "LONG"
        trend_info["strength"] = max(trend_info.get("strength", 0.0) or 0.0, cfg.trend.min_trend_strength)

    funding_annualized, basis_pct = load_carry_inputs(symbol)
    carry_info = compute_carry_bias(funding_annualized=funding_annualized, basis_pct=basis_pct, cfg=cfg.carry)

    hybrid = decide_hybrid_side(trend_info=trend_info, carry_info=carry_info, cfg=cfg)
    side = hybrid.get("side", "NONE")
    if side == "NONE":
        return None

    # ATR-based vol factor
    vol_factor = compute_vol_factor(atr_value, price, cfg)
    if vol_factor is None:
        return None

    per_trade_nav_pct = compute_per_trade_nav_pct(cfg.base_per_trade_nav_pct, vol_factor, cfg)

    gross_usd = nav * per_trade_nav_pct
    if gross_usd <= 0:
        return None

    # Compute TP/SL prices using ATR multiples
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    reward_risk: Optional[float] = None

    tp_sl = compute_tp_sl_prices(price=price, atr_value=atr_value, side=side, cfg=cfg)
    if tp_sl is not None:
        tp_price, sl_price, reward_risk = tp_sl

    # Let position_sizing + precision engine handle qty & min_notional
    intent = {
        "timestamp": _utc_now_iso(),
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": side,
        "reduceOnly": False,
        "price": price,
        "per_trade_nav_pct": per_trade_nav_pct,
        "gross_usd": gross_usd,
        # TP/SL at top-level for executor/router to consume
        "take_profit_price": tp_price,
        "stop_loss_price": sl_price,
        "metadata": {
            "strategy": "vol_target",
            "vol_target": {
                "atr_value": atr_value,
                "atr_regime": atr_regime,
                "dd_regime": dd_regime,
                "risk_mode": risk_mode,
                "target_vol": cfg.target_vol,
                "vol_factor": vol_factor,
                "base_per_trade_nav_pct": cfg.base_per_trade_nav_pct,
                "computed_per_trade_nav_pct": per_trade_nav_pct,
                "trend": trend_info,
                "carry": carry_info,
                "hybrid": hybrid,
                # TP/SL metadata (v7.3-alpha1)
                "tp_sl": {
                    "enable_tp_sl": cfg.enable_tp_sl,
                    "sl_atr_mult": cfg.sl_atr_mult,
                    "tp_atr_mult": cfg.tp_atr_mult,
                    "min_rr": cfg.min_rr,
                    "reward_risk": reward_risk,
                    "take_profit_price": tp_price,
                    "stop_loss_price": sl_price,
                },
            },
        },
    }

    return intent
