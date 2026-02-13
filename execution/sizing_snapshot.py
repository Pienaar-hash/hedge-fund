"""Pre-order sizing snapshot — canonical audit record of *why* an order was sized this way.

Every snapshot captures the full sizing decomposition:

    base_nav_pct × vol_multiplier × adaptive_weight × adaptive_factor × doctrine_multiplier
                                                                        → final_nav_pct → target_notional → final_qty

Snapshots are append-only JSONL written to ``logs/execution/sizing_snapshots.jsonl``.
They are emitted **once per attempted order**, immediately before ``build_order_payload``,
so the record is _causal_ (all factors resolved) and _pre-order_ (not yet sent).

If a veto kills the order before reaching the snapshot call site, no snapshot is written —
this is intentional: the snapshot proves sizing intent, not veto paths (those live in
``risk_vetoes.jsonl`` / ``doctrine_events.jsonl``).
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

from execution.log_utils import get_logger, log_event

# ---------------------------------------------------------------------------
# Conviction shadow helpers  (E1-P3)
# ---------------------------------------------------------------------------

def _load_conviction_mode() -> str:
    """Return conviction mode from strategy_config.json ('off'|'shadow'|'live').

    Fail-open: returns 'off' on any error.
    """
    try:
        import json
        cfg_path = Path("config/strategy_config.json")
        if not cfg_path.exists():
            return "off"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        mode = (cfg.get("conviction", {}).get("mode") or "off").lower()
        return mode if mode in ("off", "shadow", "live") else "off"
    except Exception:
        return "off"


def _compute_conviction_shadow(
    intent: Mapping[str, Any],
    vol_regime: str,
) -> Optional[Dict[str, Any]]:
    """Compute conviction in shadow mode — returns fields to merge or None on failure.

    Produces ``conviction_mult_shadow``, ``conviction_band_shadow``,
    ``conviction_score_shadow`` — never touches ``final_qty``.
    """
    try:
        from execution.conviction_engine import (
            ConvictionConfig,
            ConvictionContext,
            compute_conviction,
            load_conviction_config,
        )
        import json

        # --- Load config -----------------------------------------------
        cfg_path = Path("config/strategy_config.json")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as fh:
                strat_cfg = json.load(fh)
            conv_cfg = load_conviction_config(strat_cfg)
        else:
            conv_cfg = ConvictionConfig()

        # --- Build context from intent fields --------------------------
        meta = intent.get("metadata") or {}
        if not isinstance(meta, Mapping):
            meta = {}
        adaptive = meta.get("adaptive_weight", {})
        if not isinstance(adaptive, Mapping):
            adaptive = {}

        ctx = ConvictionContext(
            hybrid_score=_safe_float(intent.get("hybrid_score"), 0.5),
            router_quality=_safe_float(intent.get("router_quality_score"), 0.5),
            vol_regime=vol_regime or "normal",
            dd_state=str(adaptive.get("dd_regime", "normal")),
            risk_mode=str(adaptive.get("risk_mode", "normal")),
        )

        result = compute_conviction(ctx, conv_cfg)
        return {
            "conviction_mult_shadow": round(result.size_multiplier, 6),
            "conviction_band_shadow": result.conviction_band,
            "conviction_score_shadow": round(result.conviction_score, 6),
        }
    except Exception:
        return None

_LOG_SIZING = get_logger("logs/execution/sizing_snapshots.jsonl")

# Required keys in every snapshot (used by tests and schema validation).
REQUIRED_KEYS: frozenset[str] = frozenset(
    [
        "ts",
        "attempt_id",
        "symbol",
        "side",
        "strategy",
        "tier",
        "nav_usd",
        "base_nav_pct",
        "vol_regime",
        "vol_multiplier",
        "adaptive_weight",
        "atr_factor",
        "dd_factor",
        "risk_factor",
        "adaptive_factor",
        "doctrine_multiplier",
        "final_nav_pct",
        "target_notional_usd",
        "final_qty",
        "caps_applied",
    ]
)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extract_strategy(intent: Mapping[str, Any]) -> str:
    """Resolve strategy name from the many possible intent fields."""
    for key in ("strategy", "strategy_name", "strategyId", "source"):
        val = intent.get(key)
        if val:
            return str(val)
    meta = intent.get("metadata")
    if isinstance(meta, Mapping):
        val = meta.get("strategy")
        if val:
            return str(val)
    return "UNKNOWN"


def _extract_adaptive(intent: Mapping[str, Any]) -> Dict[str, float]:
    """Pull adaptive decomposition from intent metadata."""
    meta = intent.get("metadata") or {}
    adaptive = meta.get("adaptive", {}) if isinstance(meta, Mapping) else {}
    weight_block = meta.get("adaptive_weight", {}) if isinstance(meta, Mapping) else {}

    atr = _safe_float(adaptive.get("atr_factor"), 1.0)
    dd = _safe_float(adaptive.get("dd_factor"), 1.0)
    risk = _safe_float(adaptive.get("risk_factor"), 1.0)
    final_factor = _safe_float(adaptive.get("final_factor"), 1.0)
    final_weight = _safe_float(weight_block.get("final_weight"), 1.0)

    return {
        "atr_factor": atr,
        "dd_factor": dd,
        "risk_factor": risk,
        "adaptive_factor": final_factor,
        "adaptive_weight": final_weight,
    }


def _extract_vol(intent: Mapping[str, Any]) -> Dict[str, Any]:
    """Pull vol regime sizing info from sizing_notes."""
    notes = intent.get("sizing_notes")
    if not isinstance(notes, Mapping):
        return {"vol_regime": "unknown", "vol_multiplier": 1.0}
    return {
        "vol_regime": str(notes.get("vol_regime", "unknown")),
        "vol_multiplier": _safe_float(notes.get("vol_sizing_mult"), 1.0),
    }


def _detect_caps(
    intent: Mapping[str, Any],
    gross_target: float,
    nav_usd: float,
) -> List[str]:
    """Identify which caps were binding (heuristic, best-effort)."""
    caps: List[str] = []
    screener_gross = _safe_float(intent.get("gross_usd"))
    if screener_gross > 0 and gross_target < screener_gross * 0.99:
        caps.append("RISK_ADJUSTED")
    if nav_usd > 0:
        nav_pct = gross_target / nav_usd
        per_sym = intent.get("symbol_caps")
        if isinstance(per_sym, Mapping):
            cap_val = _safe_float(per_sym.get("max_nav_pct"))
            if cap_val > 0 and nav_pct >= cap_val * 0.99:
                caps.append("PER_SYMBOL_CAP")
    notes = intent.get("sizing_notes")
    if isinstance(notes, Mapping):
        floors = notes.get("floors")
        if isinstance(floors, Mapping):
            for floor_key in ("symbol_min_gross", "symbol_min_notional", "exchange_min_notional"):
                floor_val = _safe_float(floors.get(floor_key))
                if floor_val > 0 and gross_target <= floor_val * 1.01:
                    caps.append("MIN_NOTIONAL")
                    break
    return caps


def emit_sizing_snapshot(
    *,
    intent: Mapping[str, Any],
    attempt_id: str,
    symbol: str,
    side: str,
    pos_side: str,
    gross_target: float,
    nav_usd: float,
    tier_name: str,
    price_hint: float,
    reduce_only: bool = False,
) -> Dict[str, Any]:
    """Build and persist the sizing snapshot.  Returns the snapshot dict.

    Parameters
    ----------
    intent : dict
        The full screener intent (carries metadata, sizing_notes, etc.).
    attempt_id : str
        Unique attempt identifier for this order.
    symbol, side, pos_side, tier_name : str
        Standard order identifiers.
    gross_target : float
        The final notional USD that will be sent to ``build_order_payload``.
    nav_usd : float
        Current NAV at time of order.
    price_hint : float
        Price used for qty estimation.
    reduce_only : bool
        Whether this is a reduce-only (exit) order.
    """
    adaptive = _extract_adaptive(intent)
    vol = _extract_vol(intent)
    base_nav_pct = _safe_float(intent.get("per_trade_nav_pct"))
    doctrine_mult = _safe_float(intent.get("doctrine_multiplier"), 1.0)

    # Reconstruct final_nav_pct from the chain
    final_nav_pct = gross_target / nav_usd if nav_usd > 0 else 0.0

    # Estimate final qty (mirrors screener estimate — actual qty comes from build_order_payload)
    final_qty = gross_target / price_hint if price_hint > 0 else 0.0

    caps = _detect_caps(intent, gross_target, nav_usd)

    snapshot: Dict[str, Any] = {
        "ts": time.time(),
        "attempt_id": attempt_id,
        "symbol": symbol,
        "side": side,
        "position_side": pos_side,
        "strategy": _extract_strategy(intent),
        "tier": tier_name,
        "reduce_only": reduce_only,
        "nav_usd": round(nav_usd, 2),
        "base_nav_pct": base_nav_pct,
        "vol_regime": vol["vol_regime"],
        "vol_multiplier": vol["vol_multiplier"],
        "adaptive_weight": adaptive["adaptive_weight"],
        "atr_factor": adaptive["atr_factor"],
        "dd_factor": adaptive["dd_factor"],
        "risk_factor": adaptive["risk_factor"],
        "adaptive_factor": adaptive["adaptive_factor"],
        "doctrine_multiplier": doctrine_mult,
        "final_nav_pct": round(final_nav_pct, 6),
        "target_notional_usd": round(gross_target, 4),
        "final_qty": final_qty,
        "price_hint": price_hint,
        "caps_applied": caps,
    }

    # ── E1-P3: Conviction Shadow Mode ──────────────────────────────────
    # When mode="shadow", compute conviction multiplier and attach to the
    # snapshot for observability.  final_qty is NEVER modified — shadow is
    # measurement only.
    conv_mode = _load_conviction_mode()
    if conv_mode == "shadow":
        shadow = _compute_conviction_shadow(intent, vol["vol_regime"])
        if shadow is not None:
            snapshot.update(shadow)

    log_event(_LOG_SIZING, "sizing_snapshot", snapshot)
    return snapshot
