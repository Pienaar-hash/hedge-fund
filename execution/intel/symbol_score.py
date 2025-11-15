from __future__ import annotations

"""
Symbol scoring model for execution intelligence (v5.10.0).

This module combines existing metrics (Sharpe, ATR regime, router effectiveness,
fees, DD) into a single "quality score" per symbol.

The score is intended to:
- Drive size multipliers in later v5.10.x commits.
- Support dashboards and routing policies.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from execution.utils.metrics import (
    dd_today_pct,
    rolling_sharpe_7d,
    router_effectiveness_7d,
)
from execution.utils.vol import atr_pct


@dataclass
class SymbolScoreComponents:
    sharpe: float = 0.0
    sharpe_score: float = 0.0
    atr_ratio: Optional[float] = None
    atr_score: float = 0.0
    router_score: float = 0.0
    dd_pct: float = 0.0
    dd_score: float = 0.0
    total: float = 0.0


def _score_sharpe(sharpe: Optional[float]) -> float:
    """
    Map Sharpe to a bounded contribution.

    Rough skeleton:
    - Sharpe <= -1 → -1
    - Sharpe  ~ 0 → 0
    - Sharpe >=  2 → +1
    """
    if sharpe is None:
        return 0.0
    if sharpe <= -1.0:
        return -1.0
    if sharpe >= 2.0:
        return 1.0
    return sharpe / 2.0


def _score_atr_ratio(atr_now: float, atr_med: float) -> float:
    """
    Score the ATR ratio (short/long volatility).

    Idea:
    - very quiet / very extreme regimes are penalized slightly
    - normal volatility is neutral
    """
    if atr_med <= 0:
        return 0.0
    ratio = atr_now / atr_med
    if ratio < 0.5:
        return -0.2
    if ratio > 2.5:
        return -0.3
    if 0.8 <= ratio <= 1.5:
        return 0.1
    return 0.0


def _score_router(eff: Dict[str, Any]) -> float:
    """
    Score router effectiveness.

    Skeleton:
    - High maker fill ratio and low fallback ratio contribute positively.
    - Very high fallback or bad median slippage contribute negatively.
    """
    maker = eff.get("maker_fill_ratio") or 0.0
    fallback = eff.get("fallback_ratio") or 0.0
    slip_med = eff.get("slip_q50")

    score = 0.0
    score += (maker - 0.5) * 0.5  # gently reward >50% maker fills
    score -= max(0.0, fallback - 0.5)  # penalize fallback above 50%

    if slip_med is not None and slip_med > 5.0:
        score -= 0.3
    return score


def _score_dd(dd_pct: float) -> float:
    """
    Score today's DD (drawdown).

    Skeleton:
    - dd <= -3% → -0.5
    - dd ~ 0%  → 0
    - dd > 0   → small positive (if we ever encode that)
    """
    if dd_pct <= -3.0:
        return -0.5
    if dd_pct <= -1.5:
        return -0.25
    if dd_pct >= 0.5:
        return 0.1
    return 0.0


def compute_symbol_score(symbol: str) -> Dict[str, Any]:
    """
    Compute a symbol quality score.

    Returns:
        {
          "symbol": str,
          "score": float,
          "components": {
            "sharpe": float,
            "sharpe_score": float,
            "atr_ratio": float | None,
            "atr_score": float,
            "router_score": float,
            "dd_pct": float,
            "dd_score": float,
          },
        }

    v5.10.0 intentionally uses a simple linear scheme; future sprints may
    refine the weighting or add more inputs.
    """
    sharpe = rolling_sharpe_7d(symbol)
    sharpe_contrib = _score_sharpe(sharpe)

    atr_now = atr_pct(symbol, lookback_bars=50)
    atr_med = atr_pct(symbol, lookback_bars=500)
    atr_ratio = (atr_now / atr_med) if atr_med > 0 else None
    atr_contrib = _score_atr_ratio(atr_now, atr_med) if atr_ratio is not None else 0.0

    router_eff = router_effectiveness_7d(symbol) or {}
    router_contrib = _score_router(router_eff)

    dd = dd_today_pct(symbol) or 0.0
    dd_contrib = _score_dd(dd)

    total = sharpe_contrib + atr_contrib + router_contrib + dd_contrib

    comps = SymbolScoreComponents(
        sharpe=sharpe or 0.0,
        sharpe_score=sharpe_contrib,
        atr_ratio=atr_ratio,
        atr_score=atr_contrib,
        router_score=router_contrib,
        dd_pct=dd,
        dd_score=dd_contrib,
        total=total,
    )

    return {
        "symbol": symbol,
        "score": comps.total,
        "components": {
            "sharpe": comps.sharpe,
            "sharpe_score": comps.sharpe_score,
            "atr_ratio": comps.atr_ratio,
            "atr_score": comps.atr_score,
            "router_score": comps.router_score,
            "dd_pct": comps.dd_pct,
            "dd_score": comps.dd_score,
        },
    }


def score_to_size_factor(score: float) -> float:
    """
    Map a symbol score into a size multiplier.

    Skeleton:
    - score <= -1.0 → 0.5x size
    - score ~ 0    → 1.0x size
    - score >= +1.5 → 1.5x size
    - Always clamp to [0.25, 2.0]
    """
    if score <= -1.0:
        factor = 0.5
    elif score >= 1.5:
        factor = 1.5
    else:
        factor = 1.0 + 0.3 * score

    return max(0.25, min(2.0, factor))


__all__ = [
    "SymbolScoreComponents",
    "compute_symbol_score",
    "score_to_size_factor",
    "symbol_size_factor",
]


def symbol_size_factor(symbol: str) -> Dict[str, Any]:
    """
    Convenience wrapper returning score, size factor, and components.
    """
    payload = compute_symbol_score(symbol)
    score = float(payload.get("score") or 0.0)
    factor = score_to_size_factor(score)
    return {
        "symbol": symbol,
        "score": score,
        "size_factor": factor,
        "components": payload.get("components") or {},
    }
