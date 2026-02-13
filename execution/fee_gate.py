"""
Fee-Aware Edge Gate (v7.9-E2)

Structural gate that rejects entries where the expected edge does not
beat the estimated round-trip fee cost.

This is a doctrine-aligned safety rail: "every trade must have positive
expected value net of fees."  The existing ``expectancy_min`` defaults to
0.0 and does not encode fee awareness.

Gate logic:
    round_trip_fee = notional * taker_fee_rate * 2
    required_edge  = round_trip_fee * fee_buffer_mult
    allowed        = expected_edge_usd >= required_edge

Configuration via runtime.yaml:
    fee_gate:
      taker_fee_rate: 0.0004     # Binance futures taker fee (0.04%)
      maker_fee_rate: 0.0002     # For reference / future maker-first orders
      fee_buffer_mult: 1.5       # Require 1.5x round-trip fee as edge
      enabled: true

This gate does NOT influence exits — only entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

LOG = logging.getLogger("fee_gate")

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_TAKER_FEE_RATE: float = 0.0004    # Binance futures taker (0.04%)
DEFAULT_MAKER_FEE_RATE: float = 0.0002    # Binance futures maker (0.02%)
DEFAULT_FEE_BUFFER_MULT: float = 1.5      # Require 1.5x round-trip fee
DEFAULT_ENABLED: bool = True


@dataclass
class FeeGateConfig:
    """Fee gate configuration — loaded from runtime.yaml."""

    taker_fee_rate: float = DEFAULT_TAKER_FEE_RATE
    maker_fee_rate: float = DEFAULT_MAKER_FEE_RATE
    fee_buffer_mult: float = DEFAULT_FEE_BUFFER_MULT
    enabled: bool = DEFAULT_ENABLED


def load_fee_gate_config(runtime_cfg: Optional[Dict[str, Any]] = None) -> FeeGateConfig:
    """Load fee gate config from runtime.yaml or dict."""
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

    fg = runtime_cfg.get("fee_gate") if runtime_cfg else None
    if not fg or not isinstance(fg, dict):
        return FeeGateConfig()

    return FeeGateConfig(
        taker_fee_rate=float(fg.get("taker_fee_rate", DEFAULT_TAKER_FEE_RATE)),
        maker_fee_rate=float(fg.get("maker_fee_rate", DEFAULT_MAKER_FEE_RATE)),
        fee_buffer_mult=float(fg.get("fee_buffer_mult", DEFAULT_FEE_BUFFER_MULT)),
        enabled=bool(fg.get("enabled", DEFAULT_ENABLED)),
    )


def compute_round_trip_fee(notional_usd: float, fee_rate: float) -> float:
    """Compute estimated round-trip (entry + exit) fee in USD."""
    return notional_usd * fee_rate * 2


def compute_required_edge(round_trip_fee: float, fee_buffer_mult: float) -> float:
    """Compute minimum required edge in USD to justify the trade."""
    return round_trip_fee * fee_buffer_mult


def check_fee_edge(
    notional_usd: float,
    expected_edge_pct: float,
    config: Optional[FeeGateConfig] = None,
) -> tuple[bool, Dict[str, Any]]:
    """Check if the expected edge justifies the round-trip fee.

    Args:
        notional_usd: Proposed trade notional in USD
        expected_edge_pct: Expected edge as a fraction (e.g. 0.002 = 0.2%)
        config: Fee gate config (loaded from runtime.yaml if None)

    Returns:
        ``(allowed, details)`` where details always contains the fee
        arithmetic for logging/diagnostics.
    """
    if config is None:
        config = load_fee_gate_config()

    rt_fee = compute_round_trip_fee(notional_usd, config.taker_fee_rate)
    required_edge = compute_required_edge(rt_fee, config.fee_buffer_mult)
    expected_edge_usd = notional_usd * abs(expected_edge_pct)

    details = {
        "notional_usd": round(notional_usd, 2),
        "taker_fee_rate": config.taker_fee_rate,
        "round_trip_fee_usd": round(rt_fee, 4),
        "fee_buffer_mult": config.fee_buffer_mult,
        "required_edge_usd": round(required_edge, 4),
        "expected_edge_pct": round(expected_edge_pct, 6),
        "expected_edge_usd": round(expected_edge_usd, 4),
    }

    if not config.enabled:
        details["gate_status"] = "disabled"
        return True, details

    allowed = expected_edge_usd >= required_edge
    details["gate_status"] = "pass" if allowed else "veto"

    if not allowed:
        details["shortfall_usd"] = round(required_edge - expected_edge_usd, 4)
        LOG.info(
            "[fee_gate] VETO: edge $%.4f < required $%.4f (notional=$%.2f, rt_fee=$%.4f, buffer=%.1fx)",
            expected_edge_usd,
            required_edge,
            notional_usd,
            rt_fee,
            config.fee_buffer_mult,
        )
    else:
        LOG.debug(
            "[fee_gate] PASS: edge $%.4f >= required $%.4f",
            expected_edge_usd,
            required_edge,
        )

    return allowed, details
