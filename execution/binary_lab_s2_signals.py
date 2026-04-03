"""
Binary Lab S2 — probability signal extraction from CLOB health data.

Reads the CLOB market health log (60-second summaries) and the Sentinel-X
state surface to produce probability-based entry signals.

Signal source: ``entry_gate.signal_source == "probability_model"``

Direction rule:
    ``p_model_yes > mid + threshold`` → YES
    ``p_model_yes < mid - threshold`` → NO
    otherwise → SKIP

Key design invariants:
    - baseline and model are *always* separate (**Amendment 1**)
    - entry_cost uses executable ask, not mid (**Amendment 2**)
    - quote staleness is a hard gate (**Amendment 3**)
    - NO-side quotes are explicit (**Amendment 4**)
    - friction-erased-edge is a first-class skip reason (**Amendment 6**)
    - quote_reconstruction_mode logged on every signal (**final review**)
    - reconstructed probabilities are clamped and validated (**final review**)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLOB_HEALTH_PATH = Path("logs/prediction/clob_market_health.jsonl")
SENTINEL_X_STATE_PATH = Path("logs/state/sentinel_x.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUOTE_RECONSTRUCTION_MODE = "mid_plus_mean_spread"

# Tolerance for reconstructed quote invariant checks (accounts for float rounding)
_QUOTE_INVARIANT_TOL = 0.02


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BinaryLabS2Signal:
    """Immutable probability signal snapshot from CLOB + model."""

    # CLOB YES-side quotes (reconstructed from health log)
    p_yes_bid: float
    p_yes_ask: float
    p_yes_mid: float
    # CLOB NO-side quotes (derived) — Amendment 4
    p_no_bid: float
    p_no_ask: float
    # Spread (probability units)
    spread: float
    depth_score: float
    # Quote freshness — Amendment 3
    quote_age_s: float
    quote_reconstruction_mode: str
    # Baseline + model — Amendment 1 (permanently separate)
    p_baseline_yes: float
    p_model_yes: float
    edge_yes: float           # p_model_yes - p_yes_mid
    baseline_edge: float      # p_baseline_yes - p_yes_mid
    # Executable pricing — Amendment 2
    entry_cost: float         # ask-side price for the chosen side
    executable_edge: float    # abs(p_model_yes - entry_cost)
    # Trade decision
    trade_side: str           # "YES" | "NO" | "SKIP"
    skip_reason: Optional[str]
    expected_value_usd: float
    # Calibration state — Amendment 7
    calibration_active: bool
    calibration_confident: bool
    # Price region classification (PM sleeve v1)
    price_region: str
    # Replay
    features: Dict[str, float]
    model_version: str
    ts: str


@dataclass(frozen=True)
class S2EligibilityResult:
    """Gate outcome for a single S2 round."""
    eligible: bool
    deny_reason: Optional[str]
    signal: Optional[BinaryLabS2Signal]


# ---------------------------------------------------------------------------
# Price region classification (canonical — PM sleeve v1)
# ---------------------------------------------------------------------------
def _price_region(p_market: float) -> str:
    """Classify market price into a region for PnL attribution."""
    if p_market < 0.15:
        return "extreme_low"
    if p_market < 0.30:
        return "low"
    if p_market < 0.45:
        return "mid_low"
    if p_market < 0.55:
        return "center"
    if p_market < 0.70:
        return "mid_high"
    if p_market < 0.85:
        return "high"
    return "extreme_high"


# Regions that pass the price gate (PM sleeve v1)
_PM_SLEEVE_ALLOWED_REGIONS = frozenset({"extreme_low", "low", "mid_low"})


# ---------------------------------------------------------------------------
# Health log reader
# ---------------------------------------------------------------------------
def _read_last_health_entry(path: Path = CLOB_HEALTH_PATH) -> Optional[Dict[str, Any]]:
    """Read the last line of the CLOB health JSONL."""
    if not path.exists():
        return None
    try:
        last_line = ""
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if not last_line:
            return None
        return json.loads(last_line)
    except Exception as exc:
        logger.warning("s2_signals: health log read failed: %s", exc)
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("s2_signals: failed to read %s: %s", path, exc)
        return None


def _parse_ts(ts_str: str) -> Optional[float]:
    """Parse ISO 8601 timestamp to unix seconds."""
    try:
        dt = datetime.fromisoformat(ts_str.rstrip("Z").replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Quote reconstruction + invariant validation
# ---------------------------------------------------------------------------
def _reconstruct_quotes(
    mid: float, spread: float,
) -> Optional[Dict[str, float]]:
    """
    Reconstruct bid/ask from mid + mean spread.

    Returns None if invariants fail (Amendment: final review — clamp/validate).

    Invariants:
        0.0 <= p_yes_bid <= p_yes_mid <= p_yes_ask <= 1.0
        0.0 <= p_no_bid <= p_no_ask <= 1.0
        abs((p_yes_ask + p_no_bid) - 1.0) < tolerance
        abs((p_yes_bid + p_no_ask) - 1.0) < tolerance
    """
    half_spread = spread / 2.0
    p_yes_bid = mid - half_spread
    p_yes_ask = mid + half_spread

    # Clamp to [0, 1]
    p_yes_bid = max(0.0, min(1.0, p_yes_bid))
    p_yes_ask = max(0.0, min(1.0, p_yes_ask))

    # Enforce ordering after clamp
    if p_yes_bid > p_yes_ask:
        return None
    if not (p_yes_bid <= mid <= p_yes_ask + _QUOTE_INVARIANT_TOL):
        return None

    # NO-side (derived) — Amendment 4
    p_no_bid = 1.0 - p_yes_ask
    p_no_ask = 1.0 - p_yes_bid

    # Clamp NO-side
    p_no_bid = max(0.0, min(1.0, p_no_bid))
    p_no_ask = max(0.0, min(1.0, p_no_ask))

    # Cross-check invariants
    if abs((p_yes_ask + p_no_bid) - 1.0) > _QUOTE_INVARIANT_TOL:
        return None
    if abs((p_yes_bid + p_no_ask) - 1.0) > _QUOTE_INVARIANT_TOL:
        return None
    if p_no_bid > p_no_ask:
        return None

    return {
        "p_yes_bid": round(p_yes_bid, 6),
        "p_yes_ask": round(p_yes_ask, 6),
        "p_no_bid": round(p_no_bid, 6),
        "p_no_ask": round(p_no_ask, 6),
    }


# ---------------------------------------------------------------------------
# Edge bucket mapping — Amendment 8 (pure numeric labels)
# ---------------------------------------------------------------------------
def edge_to_bucket(abs_edge: float) -> str:
    """Map absolute edge to a pure numeric bucket label."""
    if abs_edge < 0.02:
        return "edge_0_2pp"
    if abs_edge < 0.05:
        return "edge_2_5pp"
    if abs_edge < 0.08:
        return "edge_5_8pp"
    return "edge_8pp_plus"


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
def extract_s2_signal(
    model: Any,  # BinaryProbabilityModel
    *,
    per_round_usd: float = 30.0,
    clob_health_path: Path = CLOB_HEALTH_PATH,
    sentinel_path: Path = SENTINEL_X_STATE_PATH,
    min_edge_threshold: float = 0.03,
) -> Optional[BinaryLabS2Signal]:
    """
    Build a :class:`BinaryLabS2Signal` from CLOB health + Sentinel-X features.

    Returns ``None`` when upstream data is missing (no health log, no mid, etc.).
    Returns a signal with ``trade_side=SKIP`` when data is present but no trade
    is indicated — including friction-erased-edge detection.
    """
    # 1. Read CLOB health
    health = _read_last_health_entry(clob_health_path)
    if health is None:
        logger.debug("s2_signals: health log unavailable")
        return None

    # 2. Quote age — Amendment 3
    health_ts = health.get("ts")
    if not health_ts:
        return None
    parsed_ts = _parse_ts(str(health_ts))
    if parsed_ts is None:
        return None
    now = time.time()
    quote_age_s = now - parsed_ts

    # 3. Extract mid + spread
    mid_block = health.get("mid")
    spread_block = health.get("spread")
    if mid_block is None or spread_block is None:
        logger.debug("s2_signals: health missing mid or spread")
        return None

    p_yes_mid = float(mid_block.get("last", 0))
    spread_mean = float(spread_block.get("mean", 0))

    if p_yes_mid <= 0 or p_yes_mid >= 1:
        logger.debug("s2_signals: mid out of range: %s", p_yes_mid)
        return None

    # 4. Prefer live CLOB bid/ask from health snapshot; fall back to reconstruction
    clob_quotes = health.get("clob_quotes")
    quote_source = QUOTE_RECONSTRUCTION_MODE
    clob_bid: Optional[float] = None
    clob_ask: Optional[float] = None
    if clob_quotes and isinstance(clob_quotes, dict):
        # Pick the first asset with valid bid/ask (YES-side token)
        for _aid, q in clob_quotes.items():
            bid = q.get("best_bid")
            ask = q.get("best_ask")
            if bid is not None and ask is not None and 0 < bid < ask <= 1:
                clob_bid = float(bid)
                clob_ask = float(ask)
                quote_source = "clob_live"
                break

    if clob_bid is not None and clob_ask is not None:
        p_yes_bid = clob_bid
        p_yes_ask = clob_ask
        p_no_bid = max(0.0, round(1.0 - p_yes_ask, 6))
        p_no_ask = min(1.0, round(1.0 - p_yes_bid, 6))
        actual_spread = round(p_yes_ask - p_yes_bid, 6)
    else:
        quotes = _reconstruct_quotes(p_yes_mid, spread_mean)
        if quotes is None:
            # Build a minimal signal for the rejection log
            features: Dict[str, float] = {"p_yes_mid": p_yes_mid, "spread": spread_mean}
            return BinaryLabS2Signal(
                p_yes_bid=0.0, p_yes_ask=0.0, p_yes_mid=p_yes_mid,
                p_no_bid=0.0, p_no_ask=0.0,
                spread=spread_mean, depth_score=0.0,
                quote_age_s=round(quote_age_s, 2),
                quote_reconstruction_mode=QUOTE_RECONSTRUCTION_MODE,
                p_baseline_yes=p_yes_mid, p_model_yes=p_yes_mid,
                edge_yes=0.0, baseline_edge=0.0,
                entry_cost=0.0, executable_edge=0.0,
                trade_side="SKIP",
                skip_reason="SKIP_INVALID_QUOTE_RECONSTRUCTION",
                expected_value_usd=0.0,
                calibration_active=model.calibration_active,
                calibration_confident=model.calibration_confident,
                price_region=_price_region(p_yes_mid),
                features=features,
                model_version=model.model_version,
                ts=datetime.now(timezone.utc).isoformat(),
            )
        p_yes_bid = quotes["p_yes_bid"]
        p_yes_ask = quotes["p_yes_ask"]
        p_no_bid = quotes["p_no_bid"]
        p_no_ask = quotes["p_no_ask"]
        actual_spread = spread_mean

    # 5. Depth score from health log
    event_freq = float(health.get("event_frequency_hz", 0))
    trade_count = int(health.get("trade_count", 0))
    depth_score = round(min(1.0, (event_freq * 10 + trade_count) / 100), 4)

    # 6. Read Binance features from sentinel state surface
    sentinel = _read_json(sentinel_path)
    sentinel_features: Dict[str, float] = {}
    if sentinel is not None:
        raw_features = sentinel.get("features") or {}
        for key in ("trend_slope", "vol_regime_z", "volume_z", "return_1m", "return_3m", "return_5m"):
            val = raw_features.get(key)
            if val is not None:
                sentinel_features[key] = float(val)

    # 7. Build feature dict for model
    features = {
        "p_yes_mid": p_yes_mid,
        "p_yes_bid": p_yes_bid,
        "p_yes_ask": p_yes_ask,
        "spread": actual_spread,
        "depth_score": depth_score,
        "quote_age_s": round(quote_age_s, 2),
        **sentinel_features,
    }

    # 8. Separate baseline + model predictions — Amendment 1
    p_baseline_yes = model.predict_baseline(features)
    p_model_yes = model.predict(features)

    # 9. Edges
    edge_yes = p_model_yes - p_yes_mid
    baseline_edge = p_baseline_yes - p_yes_mid

    # 10. Trade side + executable pricing — Amendment 2
    trade_side = "SKIP"
    skip_reason: Optional[str] = None
    entry_cost = 0.0

    if edge_yes >= min_edge_threshold:
        # Model says YES is underpriced → buy YES at ask
        trade_side = "YES"
        entry_cost = p_yes_ask
    elif edge_yes <= -min_edge_threshold:
        # Model says YES is overpriced → buy NO at NO ask
        trade_side = "NO"
        entry_cost = p_no_ask
    else:
        skip_reason = "edge_below_threshold"

    # 11. Executable edge — does edge survive friction? — Amendment 6
    executable_edge = 0.0
    if trade_side != "SKIP":
        if trade_side == "YES":
            executable_edge = p_model_yes - entry_cost
        else:  # NO
            executable_edge = entry_cost - p_model_yes  # NO profits when model < mid

        # Friction kills edge: mid-edge passes threshold but ask-edge doesn't
        if executable_edge < min_edge_threshold:
            skip_reason = "SKIP_FRICTION_ERASED_EDGE"
            trade_side = "SKIP"

    # 12. EV from entry_cost (not mid) — Amendment 2
    expected_value_usd = 0.0
    if trade_side != "SKIP" and entry_cost > 0:
        expected_value_usd = round(executable_edge * per_round_usd, 4)

    ts = datetime.now(timezone.utc).isoformat()

    return BinaryLabS2Signal(
        p_yes_bid=p_yes_bid,
        p_yes_ask=p_yes_ask,
        p_yes_mid=round(p_yes_mid, 6),
        p_no_bid=p_no_bid,
        p_no_ask=p_no_ask,
        spread=round(actual_spread, 6),
        depth_score=depth_score,
        quote_age_s=round(quote_age_s, 2),
        quote_reconstruction_mode=quote_source,
        p_baseline_yes=round(p_baseline_yes, 6),
        p_model_yes=round(p_model_yes, 6),
        edge_yes=round(edge_yes, 6),
        baseline_edge=round(baseline_edge, 6),
        entry_cost=round(entry_cost, 6),
        executable_edge=round(executable_edge, 6),
        trade_side=trade_side,
        skip_reason=skip_reason,
        expected_value_usd=expected_value_usd,
        calibration_active=model.calibration_active,
        calibration_confident=model.calibration_confident,
        price_region=_price_region(p_yes_mid),
        features=features,
        model_version=model.model_version,
        ts=ts,
    )


# ---------------------------------------------------------------------------
# Eligibility gate
# ---------------------------------------------------------------------------
def check_s2_eligibility(
    signal: BinaryLabS2Signal,
    limits: Mapping[str, Any],
    *,
    current_nav_usd: float,
    open_positions: int,
    freeze_intact: bool,
    time_remaining_s: float,
) -> S2EligibilityResult:
    """
    Full eligibility gate for S2.

    Gates (all must pass):
        1. Quote reconstruction valid (not SKIP_INVALID_QUOTE_RECONSTRUCTION)
        2. spread < max_spread_threshold
        3. executable_edge > min_edge_threshold  — Amendment 2
        4. quote_age_s <= max_quote_age_s          — Amendment 3
        5. time_remaining_s > min_time_remaining_s
        6. freeze_intact == True
        7. open_positions < max_concurrent
        8. current_nav_usd > kill_nav_usd
        9. trade_side is not SKIP
    """
    entry_gate = limits.get("entry_gate") or {}
    position_rules = limits.get("position_rules") or {}
    kill_cfg = limits.get("kill_conditions") or {}

    # Gate 1: quote reconstruction valid
    if signal.skip_reason == "SKIP_INVALID_QUOTE_RECONSTRUCTION":
        return S2EligibilityResult(False, "invalid_quote_reconstruction", signal)

    # Gate 2: spread
    max_spread = float(entry_gate.get("max_spread_threshold", 0.04))
    if signal.spread >= max_spread:
        return S2EligibilityResult(
            False, f"spread_too_wide:{signal.spread:.4f}>={max_spread}", signal,
        )

    # Gate 3: executable edge — Amendment 2
    min_edge = float(entry_gate.get("min_edge_threshold", 0.03))
    if signal.executable_edge < min_edge:
        return S2EligibilityResult(
            False, f"executable_edge_below_min:{signal.executable_edge:.4f}<{min_edge}", signal,
        )

    # Gate 4: quote age — Amendment 3
    max_quote_age = float(entry_gate.get("max_quote_age_s", 75))
    if signal.quote_age_s > max_quote_age:
        return S2EligibilityResult(
            False, f"quote_stale:{signal.quote_age_s:.1f}s>{max_quote_age}s", signal,
        )

    # Gate 5: time remaining
    min_time = float(entry_gate.get("min_time_remaining_s", 120))
    if time_remaining_s < min_time:
        return S2EligibilityResult(
            False, f"time_remaining_low:{time_remaining_s:.0f}<{min_time}", signal,
        )

    # Gate 6: freeze
    if not freeze_intact:
        return S2EligibilityResult(False, "freeze_broken", signal)

    # Gate 7: concurrent positions
    max_concurrent = int(position_rules.get("max_concurrent", 3))
    if open_positions >= max_concurrent:
        return S2EligibilityResult(
            False, f"concurrent_cap:{open_positions}>={max_concurrent}", signal,
        )

    # Gate 8: kill line
    kill_nav = float(kill_cfg.get("kill_nav_usd", 0))
    if kill_nav > 0 and current_nav_usd <= kill_nav:
        return S2EligibilityResult(False, "kill_line_breached", signal)

    # Gate 9: trade side
    if signal.trade_side == "SKIP":
        return S2EligibilityResult(
            False, signal.skip_reason or "no_trade_signal", signal,
        )

    # Gate 10-11: Ablation gates (magnitude + side filter)
    ablation = limits.get("ablation_gate") or {}
    if ablation.get("enabled", False):
        # Gate 10: |edge_yes| magnitude threshold (raw mid-edge, not executable)
        abl_min_edge = float(ablation.get("min_edge_abs", 0.0))
        if abl_min_edge > 0 and abs(signal.edge_yes) < abl_min_edge:
            return S2EligibilityResult(
                False,
                f"ablation_edge_below_min:|edge|={abs(signal.edge_yes):.4f}<{abl_min_edge}",
                signal,
            )

        # Gate 11: Side filter
        side_filter = ablation.get("side_filter", "ALL")
        if side_filter == "YES_ONLY" and signal.trade_side != "YES":
            return S2EligibilityResult(
                False, f"ablation_side_blocked:{signal.trade_side}", signal,
            )
        elif side_filter == "NO_ONLY" and signal.trade_side != "NO":
            return S2EligibilityResult(
                False, f"ablation_side_blocked:{signal.trade_side}", signal,
            )

    return S2EligibilityResult(True, None, signal)


# ---------------------------------------------------------------------------
# PM Sleeve v1 — region-first signal extraction
# ---------------------------------------------------------------------------
def extract_pm_sleeve_signal(
    model: Any,  # BinaryProbabilityModel
    *,
    per_round_usd: float = 30.0,
    clob_health_path: Path = CLOB_HEALTH_PATH,
    sentinel_path: Path = SENTINEL_X_STATE_PATH,
    max_entry_cost: float = 0.45,
    confidence_filter_enabled: bool = True,
    min_edge_abs: float = 0.05,
) -> Optional[BinaryLabS2Signal]:
    """
    Region-first signal extraction for PM Sleeve v1.

    Gate ordering (control inversion from ``extract_s2_signal``):

        1. price_region — entry_cost < max_entry_cost → PASS
        2. side lock — YES only
        3. confidence filter — |edge_yes| >= min_edge_abs (optional)
        4. friction — spread, quote age checks (deferred to eligibility)

    Returns ``None`` when upstream data is missing.
    Returns a signal with ``trade_side=SKIP`` when any gate blocks.
    """
    # 1. Read CLOB health
    health = _read_last_health_entry(clob_health_path)
    if health is None:
        logger.debug("pm_sleeve: health log unavailable")
        return None

    # 2. Quote age
    health_ts = health.get("ts")
    if not health_ts:
        return None
    parsed_ts = _parse_ts(str(health_ts))
    if parsed_ts is None:
        return None
    now = time.time()
    quote_age_s = now - parsed_ts

    # 3. Extract mid + spread
    mid_block = health.get("mid")
    spread_block = health.get("spread")
    if mid_block is None or spread_block is None:
        logger.debug("pm_sleeve: health missing mid or spread")
        return None

    p_yes_mid = float(mid_block.get("last", 0))
    spread_mean = float(spread_block.get("mean", 0))

    if p_yes_mid <= 0 or p_yes_mid >= 1:
        logger.debug("pm_sleeve: mid out of range: %s", p_yes_mid)
        return None

    # 4. Prefer live CLOB bid/ask; fall back to reconstruction
    clob_quotes = health.get("clob_quotes")
    quote_source = QUOTE_RECONSTRUCTION_MODE
    clob_bid: Optional[float] = None
    clob_ask: Optional[float] = None
    if clob_quotes and isinstance(clob_quotes, dict):
        for _aid, q in clob_quotes.items():
            bid = q.get("best_bid")
            ask = q.get("best_ask")
            if bid is not None and ask is not None and 0 < bid < ask <= 1:
                clob_bid = float(bid)
                clob_ask = float(ask)
                quote_source = "clob_live"
                break

    if clob_bid is not None and clob_ask is not None:
        p_yes_bid = clob_bid
        p_yes_ask = clob_ask
        p_no_bid = max(0.0, round(1.0 - p_yes_ask, 6))
        p_no_ask = min(1.0, round(1.0 - p_yes_bid, 6))
        actual_spread = round(p_yes_ask - p_yes_bid, 6)
    else:
        quotes = _reconstruct_quotes(p_yes_mid, spread_mean)
        if quotes is None:
            features: Dict[str, float] = {"p_yes_mid": p_yes_mid, "spread": spread_mean}
            return BinaryLabS2Signal(
                p_yes_bid=0.0, p_yes_ask=0.0, p_yes_mid=p_yes_mid,
                p_no_bid=0.0, p_no_ask=0.0,
                spread=spread_mean, depth_score=0.0,
                quote_age_s=round(quote_age_s, 2),
                quote_reconstruction_mode=QUOTE_RECONSTRUCTION_MODE,
                p_baseline_yes=p_yes_mid, p_model_yes=p_yes_mid,
                edge_yes=0.0, baseline_edge=0.0,
                entry_cost=0.0, executable_edge=0.0,
                trade_side="SKIP",
                skip_reason="SKIP_INVALID_QUOTE_RECONSTRUCTION",
                expected_value_usd=0.0,
                calibration_active=model.calibration_active,
                calibration_confident=model.calibration_confident,
                price_region=_price_region(p_yes_mid),
                features=features,
                model_version=model.model_version,
                ts=datetime.now(timezone.utc).isoformat(),
            )
        p_yes_bid = quotes["p_yes_bid"]
        p_yes_ask = quotes["p_yes_ask"]
        p_no_bid = quotes["p_no_bid"]
        p_no_ask = quotes["p_no_ask"]
        actual_spread = spread_mean

    # 5. Depth score
    event_freq = float(health.get("event_frequency_hz", 0))
    trade_count = int(health.get("trade_count", 0))
    depth_score = round(min(1.0, (event_freq * 10 + trade_count) / 100), 4)

    # 6. Sentinel features
    sentinel = _read_json(sentinel_path)
    sentinel_features: Dict[str, float] = {}
    if sentinel is not None:
        raw_features = sentinel.get("features") or {}
        for key in ("trend_slope", "vol_regime_z", "volume_z", "return_1m", "return_3m", "return_5m"):
            val = raw_features.get(key)
            if val is not None:
                sentinel_features[key] = float(val)

    # 7. Feature dict
    features = {
        "p_yes_mid": p_yes_mid,
        "p_yes_bid": p_yes_bid,
        "p_yes_ask": p_yes_ask,
        "spread": actual_spread,
        "depth_score": depth_score,
        "quote_age_s": round(quote_age_s, 2),
        **sentinel_features,
    }

    # 8. Model predictions (used for confidence magnitude only)
    p_baseline_yes = model.predict_baseline(features)
    p_model_yes = model.predict(features)
    edge_yes = p_model_yes - p_yes_mid
    baseline_edge = p_baseline_yes - p_yes_mid

    # ---------------------------------------------------------------
    # PM Sleeve v1 Gate Ordering (control inversion)
    # ---------------------------------------------------------------

    # Entry cost is always YES ask (side-locked)
    entry_cost = p_yes_ask
    region = _price_region(entry_cost)

    trade_side = "YES"
    skip_reason: Optional[str] = None

    # GATE 1: Price region (PRIMARY — this is the alpha)
    if entry_cost >= max_entry_cost:
        trade_side = "SKIP"
        skip_reason = "SKIP_REGION_BLOCKED"

    # GATE 2: Side lock (YES only — hard)
    # Side is always YES in PM sleeve v1; this gate blocks NO signals.
    # If the model's edge is negative (model says YES overpriced),
    # the old S2 logic would go NO. PM sleeve blocks that.
    if trade_side != "SKIP" and edge_yes < 0:
        # Model disagrees with YES direction — this is fine for PM sleeve
        # since we don't use direction. But if confidence filter is on,
        # we use |edge| magnitude below. No block here.
        pass

    # GATE 3: Confidence filter (optional — uses magnitude, not direction)
    if trade_side != "SKIP" and confidence_filter_enabled:
        if abs(edge_yes) < min_edge_abs:
            trade_side = "SKIP"
            skip_reason = "SKIP_CONFIDENCE_BELOW_MIN"

    # Executable edge (for logging / friction gate in eligibility)
    executable_edge = 0.0
    if trade_side != "SKIP" and entry_cost > 0:
        executable_edge = abs(edge_yes)

    # EV
    expected_value_usd = 0.0
    if trade_side != "SKIP" and entry_cost > 0:
        # Structural EV: payoff_ratio weighted by base rate (~50%)
        payoff_ratio = (1.0 - entry_cost) / entry_cost
        expected_value_usd = round((payoff_ratio - 1.0) * 0.5 * per_round_usd, 4)

    ts = datetime.now(timezone.utc).isoformat()

    return BinaryLabS2Signal(
        p_yes_bid=p_yes_bid,
        p_yes_ask=p_yes_ask,
        p_yes_mid=round(p_yes_mid, 6),
        p_no_bid=p_no_bid,
        p_no_ask=p_no_ask,
        spread=round(actual_spread, 6),
        depth_score=depth_score,
        quote_age_s=round(quote_age_s, 2),
        quote_reconstruction_mode=quote_source,
        p_baseline_yes=round(p_baseline_yes, 6),
        p_model_yes=round(p_model_yes, 6),
        edge_yes=round(edge_yes, 6),
        baseline_edge=round(baseline_edge, 6),
        entry_cost=round(entry_cost, 6),
        executable_edge=round(executable_edge, 6),
        trade_side=trade_side,
        skip_reason=skip_reason,
        expected_value_usd=expected_value_usd,
        calibration_active=model.calibration_active,
        calibration_confident=model.calibration_confident,
        price_region=region,
        features=features,
        model_version=model.model_version,
        ts=ts,
    )


# ---------------------------------------------------------------------------
# PM Sleeve v1 — eligibility gate
# ---------------------------------------------------------------------------
def check_pm_sleeve_eligibility(
    signal: BinaryLabS2Signal,
    limits: Mapping[str, Any],
    *,
    current_nav_usd: float,
    open_positions: int,
    freeze_intact: bool,
    time_remaining_s: float,
) -> S2EligibilityResult:
    """
    Eligibility gate for PM Sleeve v1.

    Uses ``pm_sleeve_v1`` config block. Falls back to ``entry_gate`` for
    friction thresholds that are unchanged from S2.

    Gates:
        1. Quote reconstruction valid
        2. trade_side is not SKIP (region/side/confidence already checked)
        3. spread < max_spread_threshold
        4. quote_age_s <= max_quote_age_s
        5. time_remaining_s > min_time_remaining_s
        6. freeze_intact == True
        7. open_positions < max_concurrent
        8. current_nav_usd > kill_nav_usd
    """
    entry_gate = limits.get("entry_gate") or {}
    position_rules = limits.get("position_rules") or {}
    kill_cfg = limits.get("kill_conditions") or {}

    # Gate 1: quote reconstruction valid
    if signal.skip_reason == "SKIP_INVALID_QUOTE_RECONSTRUCTION":
        return S2EligibilityResult(False, "invalid_quote_reconstruction", signal)

    # Gate 2: trade side (region/side/confidence gates already applied in signal)
    if signal.trade_side == "SKIP":
        return S2EligibilityResult(
            False, signal.skip_reason or "no_trade_signal", signal,
        )

    # Gate 3: spread
    max_spread = float(entry_gate.get("max_spread_threshold", 0.04))
    if signal.spread >= max_spread:
        return S2EligibilityResult(
            False, f"spread_too_wide:{signal.spread:.4f}>={max_spread}", signal,
        )

    # Gate 4: quote age
    max_quote_age = float(entry_gate.get("max_quote_age_s", 75))
    if signal.quote_age_s > max_quote_age:
        return S2EligibilityResult(
            False, f"quote_stale:{signal.quote_age_s:.1f}s>{max_quote_age}s", signal,
        )

    # Gate 5: time remaining
    min_time = float(entry_gate.get("min_time_remaining_s", 120))
    if time_remaining_s < min_time:
        return S2EligibilityResult(
            False, f"time_remaining_low:{time_remaining_s:.0f}<{min_time}", signal,
        )

    # Gate 6: freeze
    if not freeze_intact:
        return S2EligibilityResult(False, "freeze_broken", signal)

    # Gate 7: concurrent positions
    max_concurrent = int(position_rules.get("max_concurrent", 3))
    if open_positions >= max_concurrent:
        return S2EligibilityResult(
            False, f"concurrent_cap:{open_positions}>={max_concurrent}", signal,
        )

    # Gate 8: kill line
    kill_nav = float(kill_cfg.get("kill_nav_usd", 0))
    if kill_nav > 0 and current_nav_usd <= kill_nav:
        return S2EligibilityResult(False, "kill_line_breached", signal)

    return S2EligibilityResult(True, None, signal)
