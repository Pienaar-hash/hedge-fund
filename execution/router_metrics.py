"""
Router metrics helpers backed by executor JSONL logs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

LOG_DIR = Path(os.getenv("EXEC_LOG_DIR") or "logs/execution")
ROUTER_METRICS_PATH = Path(os.getenv("ROUTER_METRICS_PATH") or (LOG_DIR / "order_metrics.jsonl"))
READ_LIMIT = int(os.getenv("EXEC_LOG_MAX_ROWS", "5000") or 5000)


def _to_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        return ts
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None
    return None


def _read_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if len(rows) > limit:
        rows = rows[-limit:]
    return rows


def _recent_window(records: List[Dict[str, Any]], window_days: int) -> List[Dict[str, Any]]:
    cutoff = time.time() - max(window_days, 0) * 86400.0
    filtered: List[Dict[str, Any]] = []
    for record in records:
        ts = _to_epoch(record.get("ts"))
        if ts is None or ts >= cutoff:
            record["_ts"] = ts
            filtered.append(record)
    return filtered


def get_recent_router_events(symbol: Optional[str] = None, window_days: int = 7) -> List[Dict[str, Any]]:
    records = _read_jsonl(ROUTER_METRICS_PATH, READ_LIMIT)
    filtered = _recent_window(records, window_days)
    sym_filter = (symbol or "").upper()
    if not sym_filter:
        return filtered
    return [record for record in filtered if str(record.get("symbol") or "").upper() == sym_filter]


# ---------------------------------------------------------------------------
# Router Health Score (v7)
# ---------------------------------------------------------------------------

def compute_maker_ratio(maker_count: int, taker_count: int) -> float:
    """
    Compute maker ratio from order counts.
    
    Returns:
        Maker ratio in [0, 1], or 0.0 if no orders.
    """
    total = maker_count + taker_count
    if total <= 0:
        return 0.0
    return maker_count / total


def compute_fallback_ratio(fallback_count: int, total_orders: int) -> float:
    """
    Compute fallback ratio from order counts.
    
    Returns:
        Fallback ratio in [0, 1], or 0.0 if no orders.
    """
    if total_orders <= 0:
        return 0.0
    return fallback_count / total_orders


def compute_reject_ratio(reject_count: int, total_orders: int) -> float:
    """
    Compute reject ratio from order counts.
    
    Returns:
        Reject ratio in [0, 1], or 0.0 if no orders.
    """
    if total_orders <= 0:
        return 0.0
    return reject_count / total_orders


def compute_maker_reliability(
    maker_ratio: float,
    fallback_ratio: float,
    reject_ratio: float,
) -> float:
    """
    Compute a reliability score from maker, fallback, and reject ratios.

    Penalties:
      fallback_ratio * 0.3
      reject_ratio * 0.2

    Result is clamped between 0.0 and 1.0.
    """
    base = float(maker_ratio or 0.0)
    penalty = float(fallback_ratio or 0.0) * 0.3 + float(reject_ratio or 0.0) * 0.2
    score = base - penalty
    return max(0.0, min(1.0, score))


def compute_slippage_penalty(avg_slippage_bps: float) -> float:
    """
    Compute slippage penalty based on average slippage in basis points.
    
    Penalty scale:
    - 0-5 bps: 0.0 penalty
    - 5-15 bps: linear 0.0-0.1 penalty
    - 15+ bps: 0.1 + additional penalty
    
    Returns:
        Penalty value in [0, 0.3]
    """
    if avg_slippage_bps <= 5.0:
        return 0.0
    elif avg_slippage_bps <= 15.0:
        # Linear interpolation from 0.0 to 0.1
        return (avg_slippage_bps - 5.0) / 100.0
    else:
        # Additional penalty for high slippage
        base = 0.1
        extra = min(0.2, (avg_slippage_bps - 15.0) / 100.0)
        return base + extra


def compute_reject_penalty(reject_ratio: float) -> float:
    """
    Compute reject penalty based on reject ratio.
    
    Returns:
        Penalty value in [0, 0.2]
    """
    if reject_ratio <= 0.0:
        return 0.0
    return min(0.2, reject_ratio * 0.5)


def compute_router_health_score(
    maker_ratio: float,
    fallback_ratio: float,
    avg_slippage_bps: float = 0.0,
    reject_ratio: float = 0.0,
) -> float:
    """
    Compute overall router health score.
    
    Formula:
        base = maker_ratio
        penalty = fallback_ratio * 0.3 + slippage_penalty + reject_penalty
        health = clamp(base - penalty, 0.0, 1.0)
    
    Args:
        maker_ratio: Ratio of maker orders [0, 1]
        fallback_ratio: Ratio of fallback orders [0, 1]
        avg_slippage_bps: Average slippage in basis points
        reject_ratio: Ratio of rejected orders [0, 1]
    
    Returns:
        Health score in [0, 1]
    """
    base = maker_ratio
    
    # Calculate penalties
    fallback_penalty = fallback_ratio * 0.3
    slippage_penalty = compute_slippage_penalty(avg_slippage_bps)
    reject_penalty = compute_reject_penalty(reject_ratio)
    
    total_penalty = fallback_penalty + slippage_penalty + reject_penalty
    
    health = base - total_penalty
    return max(0.0, min(1.0, health))


def build_router_health_snapshot(
    router_events: Optional[List[Dict[str, Any]]] = None,
    window_days: int = 7,
) -> Dict[str, Any]:
    """
    Build router health snapshot from events.
    
    Returns dict with:
    - router_health_score: Overall health [0, 1]
    - maker_ratio: Maker order ratio [0, 1]
    - fallback_ratio: Fallback order ratio [0, 1]
    - reject_ratio: Reject order ratio [0, 1]
    - avg_slippage_bps: Average slippage in bps
    - ts: Timestamp
    """
    if router_events is None:
        router_events = get_recent_router_events(window_days=window_days)
    
    if not router_events:
        return {
            "router_health_score": 0.0,
            "maker_ratio": 0.0,
            "fallback_ratio": 0.0,
            "reject_ratio": 0.0,
            "avg_slippage_bps": 0.0,
            "order_count": 0,
            "ts": time.time(),
        }
    
    # Count order types
    maker_count = 0
    taker_count = 0
    fallback_count = 0
    reject_count = 0
    slippage_values: List[float] = []
    
    for event in router_events:
        # Count maker vs taker
        order_type = str(event.get("order_type") or event.get("type") or "").upper()
        is_maker = event.get("is_maker") or order_type in ("MAKER", "LIMIT", "POST_ONLY")
        is_taker = event.get("is_taker") or order_type in ("TAKER", "MARKET")
        
        if is_maker:
            maker_count += 1
        elif is_taker:
            taker_count += 1
        else:
            # Default: count as taker
            taker_count += 1
        
        # Count fallbacks
        if event.get("fallback") or event.get("is_fallback"):
            fallback_count += 1
        
        # Count rejects
        if event.get("rejected") or event.get("is_rejected") or event.get("status") == "REJECTED":
            reject_count += 1
        
        # Collect slippage
        slip = event.get("slippage_bps") or event.get("slip_bps")
        if slip is not None:
            try:
                slippage_values.append(float(slip))
            except (TypeError, ValueError):
                pass
    
    total_orders = maker_count + taker_count
    
    # Compute ratios
    maker_ratio = compute_maker_ratio(maker_count, taker_count)
    fallback_ratio = compute_fallback_ratio(fallback_count, total_orders)
    reject_ratio = compute_reject_ratio(reject_count, total_orders)
    
    # Compute average slippage
    avg_slippage_bps = sum(slippage_values) / len(slippage_values) if slippage_values else 0.0
    
    # Compute health score
    health_score = compute_router_health_score(
        maker_ratio=maker_ratio,
        fallback_ratio=fallback_ratio,
        avg_slippage_bps=avg_slippage_bps,
        reject_ratio=reject_ratio,
    )
    
    maker_reliability = compute_maker_reliability(maker_ratio, fallback_ratio, reject_ratio)

    return {
        "router_health_score": health_score,
        "maker_ratio": maker_ratio,
        "fallback_ratio": fallback_ratio,
        "reject_ratio": reject_ratio,
        "maker_reliability": maker_reliability,
        "avg_slippage_bps": avg_slippage_bps,
        "order_count": total_orders,
        "maker_count": maker_count,
        "taker_count": taker_count,
        "fallback_count": fallback_count,
        "reject_count": reject_count,
        "ts": time.time(),
    }


# ---------------------------------------------------------------------------
# TWAP Metrics (v7.4 C1)
# ---------------------------------------------------------------------------

TWAP_METRICS_PATH = Path(os.getenv("TWAP_METRICS_PATH") or (LOG_DIR / "twap_events.jsonl"))


def get_recent_twap_events(symbol: Optional[str] = None, window_days: int = 7) -> List[Dict[str, Any]]:
    """
    Get recent TWAP execution events.
    
    Args:
        symbol: Optional symbol filter
        window_days: Number of days to look back
    
    Returns:
        List of TWAP event dicts
    """
    records = _read_jsonl(TWAP_METRICS_PATH, READ_LIMIT)
    filtered = _recent_window(records, window_days)
    
    # Filter to only TWAP events
    twap_events = [r for r in filtered if r.get("execution_style") == "twap" or r.get("event", "").startswith("twap_")]
    
    sym_filter = (symbol or "").upper()
    if not sym_filter:
        return twap_events
    return [record for record in twap_events if str(record.get("symbol") or "").upper() == sym_filter]


def build_twap_metrics_snapshot(
    twap_events: Optional[List[Dict[str, Any]]] = None,
    router_events: Optional[List[Dict[str, Any]]] = None,
    window_days: int = 7,
) -> Dict[str, Any]:
    """
    Build TWAP vs single execution metrics snapshot.
    
    Separates metrics for TWAP and single-shot executions to allow
    comparison of execution quality.
    
    Args:
        twap_events: Optional pre-loaded TWAP events
        router_events: Optional pre-loaded router events
        window_days: Number of days to look back
    
    Returns:
        Dict with TWAP and single execution metrics
    """
    if twap_events is None:
        twap_events = get_recent_twap_events(window_days=window_days)
    
    if router_events is None:
        router_events = get_recent_router_events(window_days=window_days)
    
    # Separate TWAP vs single executions from router events
    # Note: twap_router_events reserved for future TWAP-specific analysis
    _twap_router_events = [e for e in router_events if e.get("execution_style") == "twap"]  # noqa: F841
    single_router_events = [e for e in router_events if e.get("execution_style") != "twap"]
    
    # Count TWAP parent orders (unique by looking at twap_complete events)
    twap_complete_events = [e for e in twap_events if e.get("event") == "twap_complete"]
    twap_trades_count = len(twap_complete_events)
    
    # Count single trades
    single_trades_count = len(single_router_events)
    
    # Compute TWAP slippage
    twap_slippage_values: List[float] = []
    for event in twap_complete_events:
        slip = event.get("avg_slippage_bps")
        if slip is not None:
            try:
                twap_slippage_values.append(float(slip))
            except (TypeError, ValueError):
                pass
    
    twap_avg_slippage_bps = (
        sum(twap_slippage_values) / len(twap_slippage_values)
        if twap_slippage_values else 0.0
    )
    
    # Compute single slippage
    single_slippage_values: List[float] = []
    for event in single_router_events:
        slip = event.get("slippage_bps")
        if slip is not None:
            try:
                single_slippage_values.append(float(slip))
            except (TypeError, ValueError):
                pass
    
    single_avg_slippage_bps = (
        sum(single_slippage_values) / len(single_slippage_values)
        if single_slippage_values else 0.0
    )
    
    # Compute TWAP maker ratio
    twap_maker_count = sum(1 for e in twap_complete_events if e.get("maker_count", 0) > e.get("taker_count", 0))
    twap_maker_ratio = twap_maker_count / twap_trades_count if twap_trades_count > 0 else 0.0
    
    # Compute single maker ratio
    single_maker_count = sum(1 for e in single_router_events if e.get("is_maker_final"))
    single_maker_ratio = single_maker_count / single_trades_count if single_trades_count > 0 else 0.0
    
    return {
        "twap": {
            "trades_count": twap_trades_count,
            "avg_slippage_bps": twap_avg_slippage_bps,
            "maker_ratio": twap_maker_ratio,
            "total_slices": sum(e.get("slices_executed", 0) for e in twap_complete_events),
        },
        "single": {
            "trades_count": single_trades_count,
            "avg_slippage_bps": single_avg_slippage_bps,
            "maker_ratio": single_maker_ratio,
        },
        "comparison": {
            "slippage_diff_bps": twap_avg_slippage_bps - single_avg_slippage_bps,
            "maker_ratio_diff": twap_maker_ratio - single_maker_ratio,
        },
        "ts": time.time(),
    }


# ---------------------------------------------------------------------------
# v7.5_B1: Slippage Metrics Snapshot
# ---------------------------------------------------------------------------

def build_slippage_metrics_snapshot() -> Dict[str, Any]:
    """
    Build a snapshot of per-symbol slippage metrics for state publishing.
    
    Returns:
        Dict with slippage metrics from slippage_model store
    """
    try:
        from execution.slippage_model import build_slippage_snapshot
        return build_slippage_snapshot()
    except ImportError:
        return {"updated_ts": time.time(), "per_symbol": {}}
    except Exception:
        return {"updated_ts": time.time(), "per_symbol": {}}


def build_liquidity_buckets_snapshot() -> Dict[str, Any]:
    """
    Build a snapshot of liquidity bucket assignments for state publishing.
    
    Returns:
        Dict with symbol -> bucket mapping
    """
    try:
        from execution.liquidity_model import build_liquidity_snapshot, get_liquidity_model
        
        model = get_liquidity_model()
        snapshot = build_liquidity_snapshot()
        
        return {
            "updated_ts": time.time(),
            "symbols": snapshot,
            "buckets": {
                name: {
                    "max_spread_bps": bucket.max_spread_bps,
                    "default_maker_bias": bucket.default_maker_bias,
                    "symbol_count": len([s for s, b in model.symbol_to_bucket.items() if b.name == name]),
                }
                for name, bucket in model.buckets.items()
            },
        }
    except ImportError:
        return {"updated_ts": time.time(), "symbols": {}, "buckets": {}}
    except Exception:
        return {"updated_ts": time.time(), "symbols": {}, "buckets": {}}


# ---------------------------------------------------------------------------
# v7.5_B2: Router Quality Score
# ---------------------------------------------------------------------------

@dataclass
class RouterQualityConfig:
    """Configuration for router quality score computation."""
    enabled: bool = True
    base_score: float = 0.8
    min_score: float = 0.2
    max_score: float = 1.0
    stats_window_seconds: float = 900.0
    stats_window_min_events: int = 5
    latency_fast_ms: float = 150.0
    latency_normal_ms: float = 400.0
    slippage_drift_green_bps: float = 2.0
    slippage_drift_yellow_bps: float = 6.0
    bucket_penalty_a_high: float = 0.0
    bucket_penalty_b_medium: float = -0.05
    bucket_penalty_c_low: float = -0.15
    twap_skip_penalty: float = 0.10
    low_quality_threshold: float = 0.5
    high_quality_threshold: float = 0.9
    low_quality_hybrid_multiplier: float = 0.5
    high_quality_hybrid_multiplier: float = 1.05
    min_for_emission: float = 0.35


def load_router_quality_config(
    strategy_config: Mapping[str, Any] | None = None,
) -> RouterQualityConfig:
    """
    Load router quality configuration from strategy config.
    
    Args:
        strategy_config: Pre-loaded config or None to load from file
        
    Returns:
        RouterQualityConfig with settings
    """
    if strategy_config is None:
        try:
            with open("config/strategy_config.json") as f:
                strategy_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return RouterQualityConfig()
    
    rq_cfg = strategy_config.get("router_quality", {})
    if not rq_cfg:
        return RouterQualityConfig()

    drift_thresholds = rq_cfg.get("slippage_drift_bps_thresholds", {})
    bucket_penalties = rq_cfg.get("bucket_penalties", {})
    stats_window = rq_cfg.get("stats_window", {})
    latency_thresholds = rq_cfg.get("latency_ms_thresholds", {})

    return RouterQualityConfig(
        enabled=bool(rq_cfg.get("enabled", True)),
        base_score=float(rq_cfg.get("base_score", 0.8)),
        min_score=float(rq_cfg.get("min_score", 0.2)),
        max_score=float(rq_cfg.get("max_score", 1.0)),
        stats_window_seconds=float(stats_window.get("seconds", 900.0)),
        stats_window_min_events=int(stats_window.get("min_events", 5) or 0),
        latency_fast_ms=float(latency_thresholds.get("fast", 150.0)),
        latency_normal_ms=float(latency_thresholds.get("normal", 400.0)),
        slippage_drift_green_bps=float(drift_thresholds.get("green", 2.0)),
        slippage_drift_yellow_bps=float(drift_thresholds.get("yellow", 6.0)),
        bucket_penalty_a_high=float(bucket_penalties.get("A_HIGH", 0.0)),
        bucket_penalty_b_medium=float(bucket_penalties.get("B_MEDIUM", -0.05)),
        bucket_penalty_c_low=float(bucket_penalties.get("C_LOW", -0.15)),
        twap_skip_penalty=float(rq_cfg.get("twap_skip_penalty", 0.10)),
        low_quality_threshold=float(rq_cfg.get("low_quality_threshold", 0.5)),
        high_quality_threshold=float(rq_cfg.get("high_quality_threshold", 0.9)),
        low_quality_hybrid_multiplier=float(rq_cfg.get("low_quality_hybrid_multiplier", 0.5)),
        high_quality_hybrid_multiplier=float(rq_cfg.get("high_quality_hybrid_multiplier", 1.05)),
        min_for_emission=float(rq_cfg.get("min_for_emission", 0.35)),
    )


@dataclass
class RouterQualitySnapshot:
    """Snapshot of router quality metrics for a symbol."""
    symbol: str
    score: float
    bucket: str
    ewma_expected_bps: float
    ewma_realized_bps: float
    slippage_drift_bps: float
    twap_skip_ratio: float
    trade_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "score": round(self.score, 4),
            "bucket": self.bucket,
            "ewma_expected_bps": round(self.ewma_expected_bps, 4),
            "ewma_realized_bps": round(self.ewma_realized_bps, 4),
            "slippage_drift_bps": round(self.slippage_drift_bps, 4),
            "twap_skip_ratio": round(self.twap_skip_ratio, 4),
            "trade_count": self.trade_count,
        }


def compute_router_quality_score(
    *,
    bucket: str,
    ewma_expected_bps: float,
    ewma_realized_bps: float,
    twap_skip_ratio: float,
    cfg: RouterQualityConfig | None = None,
) -> float:
    """
    Compute a router quality score ∈ [min_score, max_score].
    
    Args:
        bucket: Liquidity bucket name (A_HIGH, B_MEDIUM, C_LOW, GENERIC)
        ewma_expected_bps: EWMA expected slippage in basis points
        ewma_realized_bps: EWMA realized slippage in basis points
        twap_skip_ratio: Ratio of skipped TWAP slices (0-1)
        cfg: Router quality configuration
        
    Returns:
        Router quality score in [min_score, max_score]
    """
    if cfg is None:
        cfg = load_router_quality_config()
    
    if not cfg.enabled:
        return cfg.base_score
    
    # 1. Start with base score
    score = cfg.base_score
    
    # 2. Apply bucket penalty
    bucket_upper = bucket.upper() if bucket else "GENERIC"
    if bucket_upper == "A_HIGH":
        score += cfg.bucket_penalty_a_high
    elif bucket_upper == "B_MEDIUM":
        score += cfg.bucket_penalty_b_medium
    elif bucket_upper == "C_LOW":
        score += cfg.bucket_penalty_c_low
    # GENERIC bucket gets no penalty
    
    # 3. Compute slippage drift (realized - expected)
    drift = ewma_realized_bps - ewma_expected_bps
    
    # 4. Slippage drift adjustment
    if drift <= cfg.slippage_drift_green_bps:
        # Good: small or negative drift
        score -= 0.02  # Small penalty even for good drift
    elif drift <= cfg.slippage_drift_yellow_bps:
        # Moderate: medium drift
        score -= 0.08
    else:
        # Bad: large drift
        score -= 0.18
    
    # 5. TWAP skip penalty (linear with skip ratio)
    score -= cfg.twap_skip_penalty * max(0.0, min(1.0, twap_skip_ratio))
    
    # 6. Clamp to valid range
    return max(cfg.min_score, min(cfg.max_score, score))


def build_router_quality_snapshot(
    symbol: str,
    ewma_expected_bps: float,
    ewma_realized_bps: float,
    bucket: str,
    twap_skip_ratio: float,
    trade_count: int,
    cfg: RouterQualityConfig | None = None,
) -> RouterQualitySnapshot:
    """
    Build a router quality snapshot for a symbol.
    
    Args:
        symbol: Trading pair symbol
        ewma_expected_bps: EWMA expected slippage
        ewma_realized_bps: EWMA realized slippage
        bucket: Liquidity bucket name
        twap_skip_ratio: Ratio of skipped TWAP slices
        trade_count: Number of trades
        cfg: Router quality configuration
        
    Returns:
        RouterQualitySnapshot with computed score
    """
    if cfg is None:
        cfg = load_router_quality_config()
    
    score = compute_router_quality_score(
        bucket=bucket,
        ewma_expected_bps=ewma_expected_bps,
        ewma_realized_bps=ewma_realized_bps,
        twap_skip_ratio=twap_skip_ratio,
        cfg=cfg,
    )
    
    return RouterQualitySnapshot(
        symbol=symbol.upper(),
        score=score,
        bucket=bucket,
        ewma_expected_bps=ewma_expected_bps,
        ewma_realized_bps=ewma_realized_bps,
        slippage_drift_bps=ewma_realized_bps - ewma_expected_bps,
        twap_skip_ratio=twap_skip_ratio,
        trade_count=trade_count,
    )


def build_all_router_quality_snapshots(
    cfg: RouterQualityConfig | None = None,
) -> Dict[str, RouterQualitySnapshot]:
    """
    Build router quality snapshots for all symbols with slippage data.
    
    Combines data from:
    - slippage_model (EWMA expected/realized)
    - liquidity_model (buckets)
    - TWAP metrics (skip ratios)
    
    Args:
        cfg: Router quality configuration
        
    Returns:
        Dict mapping symbol -> RouterQualitySnapshot
    """
    if cfg is None:
        cfg = load_router_quality_config()
    
    snapshots: Dict[str, RouterQualitySnapshot] = {}
    
    # Get slippage stats
    try:
        from execution.slippage_model import get_all_slippage_stats
        slippage_stats = get_all_slippage_stats()
    except ImportError:
        slippage_stats = {}
    except Exception:
        slippage_stats = {}
    
    # Get liquidity buckets
    try:
        from execution.liquidity_model import get_liquidity_model
        liquidity_model = get_liquidity_model()
    except ImportError:
        liquidity_model = None
    except Exception:
        liquidity_model = None
    
    # Get TWAP metrics for skip ratios
    twap_skip_by_symbol: Dict[str, float] = {}
    try:
        twap_events = get_recent_twap_events(window_days=7)
        for event in twap_events:
            if event.get("event") == "twap_complete":
                sym = str(event.get("symbol", "")).upper()
                slices_total = event.get("slices_total", 0)
                slices_executed = event.get("slices_executed", 0)
                if slices_total > 0:
                    skip_ratio = 1.0 - (slices_executed / slices_total)
                    # Weighted average with prior
                    if sym in twap_skip_by_symbol:
                        twap_skip_by_symbol[sym] = 0.5 * twap_skip_by_symbol[sym] + 0.5 * skip_ratio
                    else:
                        twap_skip_by_symbol[sym] = skip_ratio
    except Exception:
        pass
    
    # Build snapshots for all symbols with slippage data
    for symbol, stats in slippage_stats.items():
        symbol_upper = symbol.upper()
        
        # Get bucket
        if liquidity_model:
            bucket = liquidity_model.get_bucket_name(symbol_upper)
        else:
            bucket = "GENERIC"
        
        # Get TWAP skip ratio (default to 0 if no data)
        twap_skip_ratio = twap_skip_by_symbol.get(symbol_upper, 0.0)
        
        snapshot = build_router_quality_snapshot(
            symbol=symbol_upper,
            ewma_expected_bps=stats.ewma_expected_bps,
            ewma_realized_bps=stats.ewma_realized_bps,
            bucket=bucket,
            twap_skip_ratio=twap_skip_ratio,
            trade_count=stats.trade_count,
            cfg=cfg,
        )
        snapshots[symbol_upper] = snapshot
    
    return snapshots


def build_router_quality_state_snapshot(
    cfg: RouterQualityConfig | None = None,
) -> Dict[str, Any]:
    """
    Build complete router quality state for publishing.
    
    Returns:
        Dict with updated_ts and per-symbol router_quality data
    """
    if cfg is None:
        cfg = load_router_quality_config()
    
    snapshots = build_all_router_quality_snapshots(cfg)
    
    # Calculate aggregate metrics
    if snapshots:
        scores = [s.score for s in snapshots.values()]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score_val = max(scores)
        low_quality_count = sum(1 for s in snapshots.values() if s.score < cfg.low_quality_threshold)
        high_quality_count = sum(1 for s in snapshots.values() if s.score >= cfg.high_quality_threshold)
    else:
        avg_score = cfg.base_score
        min_score = cfg.base_score
        max_score_val = cfg.base_score
        low_quality_count = 0
        high_quality_count = 0
    
    return {
        "updated_ts": time.time(),
        "enabled": cfg.enabled,
        "summary": {
            "symbol_count": len(snapshots),
            "avg_score": round(avg_score, 4),
            "min_score": round(min_score, 4),
            "max_score": round(max_score_val, 4),
            "low_quality_count": low_quality_count,
            "high_quality_count": high_quality_count,
        },
        "symbols": {
            symbol: snapshot.to_dict()
            for symbol, snapshot in snapshots.items()
        },
    }


def get_router_quality_score(symbol: str, cfg: RouterQualityConfig | None = None) -> float:
    """
    Get the router quality score for a single symbol.
    
    Args:
        symbol: Trading pair symbol
        cfg: Router quality configuration
        
    Returns:
        Router quality score, or base_score if no data available
    """
    if cfg is None:
        cfg = load_router_quality_config()
    
    if not cfg.enabled:
        return cfg.base_score
    
    snapshots = build_all_router_quality_snapshots(cfg)
    snapshot = snapshots.get(symbol.upper())
    
    if snapshot:
        return snapshot.score
    
    return cfg.base_score


__all__ = [
    "get_recent_router_events",
    "compute_maker_ratio",
    "compute_fallback_ratio",
    "compute_reject_ratio",
    "compute_slippage_penalty",
    "compute_reject_penalty",
    "compute_router_health_score",
    "compute_maker_reliability",
    "build_router_health_snapshot",
    "build_twap_metrics_snapshot",
    "get_recent_twap_events",
    "build_slippage_metrics_snapshot",
    "build_liquidity_buckets_snapshot",
    # v7.5_B2: Router Quality Score
    "RouterQualityConfig",
    "RouterQualitySnapshot",
    "load_router_quality_config",
    "compute_router_quality_score",
    "build_router_quality_snapshot",
    "build_all_router_quality_snapshots",
    "build_router_quality_state_snapshot",
    "get_router_quality_score",
]
