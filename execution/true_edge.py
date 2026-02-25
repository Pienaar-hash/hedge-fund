"""
True Edge v1: ATR × Confidence Mapping (v7.9-TE1)

Replaces the proxy edge (confidence - 0.5) with an **expected dollar edge**
that scales with:

  * direction confidence (probability advantage over 50/50)
  * plausible move size over holding horizon (ATR-based)
  * trade notional
  * fees + slippage buffer

Core mapping:

  Step A — advantage:    adv = clamp(confidence - 0.5, 0, adv_cap)
  Step B — move size:    move_pct = k_atr * (ATR / price)
  Step C — expected edge: expected_edge_pct = adv * move_pct
  Step D — dollar edge:   expected_edge_usd = notional_usd * expected_edge_pct

Gate condition:
  PASS if expected_edge_usd >= required_edge_usd

When ATR is missing or stale, falls back to the legacy proxy
(confidence - 0.5) and marks edge_source = "fallback_proxy".

Logging fields (true_edge_v1.*) make every pass/veto explainable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

LOG = logging.getLogger("true_edge")

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_ADV_CAP: float = 0.25       # Clamp advantage at 25pp
DEFAULT_K_ATR: float = 0.6          # Conservative multiplier for m15
DEFAULT_MIN_ATR_PCT: float = 0.0001  # 0.01% — below this ATR is stale/invalid
DEFAULT_SPREAD_SLIPPAGE_BPS: float = 0.0  # Optional spread+slippage deduction

# Per-timeframe k_atr defaults
K_ATR_DEFAULTS: Dict[str, float] = {
    "m15": 0.6,
    "m5": 0.4,
    "h1": 0.8,
    "h4": 1.0,
    "d1": 1.2,
}


@dataclass
class TrueEdgeConfig:
    """True Edge v1 configuration — loaded from runtime.yaml."""

    adv_cap: float = DEFAULT_ADV_CAP
    k_atr: float = DEFAULT_K_ATR
    k_atr_by_timeframe: Dict[str, float] = field(default_factory=lambda: dict(K_ATR_DEFAULTS))
    min_atr_pct: float = DEFAULT_MIN_ATR_PCT
    spread_slippage_bps: float = DEFAULT_SPREAD_SLIPPAGE_BPS
    enabled: bool = True


def load_true_edge_config(runtime_cfg: Optional[Dict[str, Any]] = None) -> TrueEdgeConfig:
    """Load true edge config from runtime.yaml dict or defaults."""
    if runtime_cfg is None:
        try:
            import yaml
            from pathlib import Path

            path = Path("config/runtime.yaml")
            if path.exists():
                with open(path, "r") as fh:
                    runtime_cfg = yaml.safe_load(fh) or {}
        except Exception:
            runtime_cfg = {}

    te = runtime_cfg.get("true_edge") if runtime_cfg else None
    if not te or not isinstance(te, dict):
        return TrueEdgeConfig()

    k_atr_tf = dict(K_ATR_DEFAULTS)
    if isinstance(te.get("k_atr_by_timeframe"), dict):
        k_atr_tf.update(te["k_atr_by_timeframe"])

    return TrueEdgeConfig(
        adv_cap=float(te.get("adv_cap", DEFAULT_ADV_CAP)),
        k_atr=float(te.get("k_atr", DEFAULT_K_ATR)),
        k_atr_by_timeframe=k_atr_tf,
        min_atr_pct=float(te.get("min_atr_pct", DEFAULT_MIN_ATR_PCT)),
        spread_slippage_bps=float(te.get("spread_slippage_bps", DEFAULT_SPREAD_SLIPPAGE_BPS)),
        enabled=bool(te.get("enabled", True)),
    )


@dataclass
class TrueEdgeResult:
    """Result of compute_true_edge — all fields logged for audit."""

    expected_edge_pct: float
    expected_edge_usd: float
    atr_pct: float          # ATR / price (fractional, NOT ×100)
    k_atr: float
    adv: float              # clamped advantage
    confidence: float       # raw confidence input
    notional_usd: float
    source: str             # "atr_conf_v1" or "fallback_proxy"
    spread_slippage_usd: float = 0.0
    fallback_reason: str = ""  # why fallback: atr_missing|atr_stale|atr_unit_suspect|price_missing

    def to_dict(self) -> Dict[str, Any]:
        return {f"true_edge_v1.{k}": v for k, v in asdict(self).items()}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


def _resolve_k_atr(config: TrueEdgeConfig, timeframe: Optional[str] = None) -> float:
    """Resolve k_atr: per-timeframe override, then global default."""
    if timeframe and timeframe in config.k_atr_by_timeframe:
        return config.k_atr_by_timeframe[timeframe]
    return config.k_atr


def compute_true_edge(
    confidence: float,
    price: float,
    atr: Optional[float],
    notional_usd: float,
    *,
    timeframe: Optional[str] = None,
    spread_slippage_bps: Optional[float] = None,
    config: Optional[TrueEdgeConfig] = None,
) -> TrueEdgeResult:
    """Compute expected dollar edge using ATR × confidence mapping.

    Args:
        confidence: Model score / probability in [0, 1.5] (clamped internally).
        price: Current symbol price in USD.
        atr: ATR value at the strategy horizon (same units as price).
              None or <= 0 triggers fallback to proxy edge.
        notional_usd: Intended trade notional in USD.
        timeframe: Strategy timeframe key (e.g. "m15") for k_atr lookup.
        spread_slippage_bps: Override spread+slippage deduction in bps.
        config: TrueEdgeConfig (loaded from runtime.yaml if None).

    Returns:
        TrueEdgeResult with all intermediate values for logging.
    """
    if config is None:
        config = load_true_edge_config()

    k_atr = _resolve_k_atr(config, timeframe)
    slip_bps = spread_slippage_bps if spread_slippage_bps is not None else config.spread_slippage_bps

    # ── Step A: Advantage over 50/50 ──────────────────────────────────
    adv = _clamp(confidence - 0.5, 0.0, config.adv_cap)

    # ── Validate ATR ──────────────────────────────────────────────────
    # ATR from vol.atr_pct() is in percentage (×100), but callers should
    # pass ATR in price units (same as `price`).  We compute atr_pct here.
    atr_pct_val: float = 0.0
    use_fallback = False
    _fallback_reason = ""

    if atr is None or atr <= 0 or price <= 0:
        use_fallback = True
        _fallback_reason = "atr_missing" if (atr is None or atr <= 0) else "price_missing"
    else:
        atr_pct_val = atr / price
        # Unit guard: if ATR/price > 1.0, caller likely passed atr_pct
        # (percentage) instead of ATR in price units.  For BTC at $100k
        # an ATR of $100k+ makes no physical sense.  Force fallback and
        # log so the call site gets fixed.
        if atr_pct_val > 1.0:
            LOG.warning(
                "[true_edge] ATR_UNIT_SUSPECT: atr=%.6f price=%.2f "
                "ratio=%.4f — likely wrong units, forcing fallback",
                atr, price, atr_pct_val,
            )
            use_fallback = True
            _fallback_reason = "atr_unit_suspect"
        elif atr_pct_val < config.min_atr_pct:
            use_fallback = True
            _fallback_reason = "atr_stale"
            use_fallback = True

    if use_fallback:
        # Legacy proxy: confidence - 0.5 (same as signal_generator)
        proxy_edge_pct = _clamp(confidence - 0.5, 0.0, 1.0)
        expected_edge_usd = notional_usd * proxy_edge_pct
        slip_usd = notional_usd * (slip_bps / 10_000) if slip_bps else 0.0
        expected_edge_usd = max(0.0, expected_edge_usd - slip_usd)
        return TrueEdgeResult(
            expected_edge_pct=proxy_edge_pct,
            expected_edge_usd=round(expected_edge_usd, 6),
            atr_pct=0.0,
            k_atr=k_atr,
            adv=adv,
            confidence=confidence,
            notional_usd=notional_usd,
            source="fallback_proxy",
            spread_slippage_usd=round(slip_usd, 6),
            fallback_reason=_fallback_reason,
        )

    # ── Step B: Expected move size ────────────────────────────────────
    move_pct = k_atr * atr_pct_val

    # ── Step C: Expected edge % ───────────────────────────────────────
    expected_edge_pct = adv * move_pct

    # ── Step D: Dollar edge ───────────────────────────────────────────
    expected_edge_usd = notional_usd * expected_edge_pct

    # Deduct spread/slippage
    slip_usd = notional_usd * (slip_bps / 10_000) if slip_bps else 0.0
    expected_edge_usd = max(0.0, expected_edge_usd - slip_usd)

    return TrueEdgeResult(
        expected_edge_pct=round(expected_edge_pct, 8),
        expected_edge_usd=round(expected_edge_usd, 6),
        atr_pct=round(atr_pct_val, 8),
        k_atr=k_atr,
        adv=round(adv, 6),
        confidence=confidence,
        notional_usd=notional_usd,
        source="atr_conf_v1",
        spread_slippage_usd=round(slip_usd, 6),
    )
