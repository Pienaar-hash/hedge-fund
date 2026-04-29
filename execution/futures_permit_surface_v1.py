"""
Futures Permit Surface v1 — binary-entry research surface (shadow-only).

Ontology: permission, not ranking.
- PERMIT_CANDIDATE: state is admissible for entry research
- ABSTAIN: insufficient evidence (default)
- DENY_STRUCTURAL: state is structurally inadmissible

This module is **structurally incapable** of gating execution:
- authority is hard-coded to "none" (not configurable)
- mode is hard-coded to "shadow_only"
- NO imports from order_router, doctrine_kernel, sizing, order_dispatch
- Output is append-only JSONL shadow log

4-gate pipeline:
  Gate 1 — Structural admissibility (5 setup classes)
  Gate 2 — Direction validity (regime → direction)
  Gate 3 — Fee bridge (ATR edge must clear round-trip fees)
  Gate 4 — Shadow emit (verdict routing)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("fps_v1")

# ---------------------------------------------------------------------------
# Shadow log path
# ---------------------------------------------------------------------------
_SHADOW_LOG = Path("logs/execution/futures_permit_surface_shadow.jsonl")

# ---------------------------------------------------------------------------
# Verdicts (exhaustive enum — no score, rank, band, percentile, weight)
# ---------------------------------------------------------------------------
PERMIT_CANDIDATE = "PERMIT_CANDIDATE"
ABSTAIN = "ABSTAIN"
DENY_STRUCTURAL = "DENY_STRUCTURAL"
_VALID_VERDICTS = frozenset({PERMIT_CANDIDATE, ABSTAIN, DENY_STRUCTURAL})

# ---------------------------------------------------------------------------
# Setup classes — regime-event (5) + hypothesis-driven (3)
# ---------------------------------------------------------------------------
REGIME_TRANSITION_BREAK = "REGIME_TRANSITION_BREAK"
POST_CRISIS_RESTABILIZATION = "POST_CRISIS_RESTABILIZATION"
DISLOCATION_REVERSION = "DISLOCATION_REVERSION"
BREAKOUT_RETEST_CONFIRM = "BREAKOUT_RETEST_CONFIRM"
LIQUIDITY_VACUUM_RECLAIM = "LIQUIDITY_VACUUM_RECLAIM"

# Hypothesis-driven classes (precise causal predicates)
VOL_EXPANSION_BREAKOUT = "VOL_EXPANSION_BREAKOUT"
EXHAUSTION_REVERSAL = "EXHAUSTION_REVERSAL"
TREND_PULLBACK = "TREND_PULLBACK"

_VALID_SETUP_CLASSES = frozenset({
    REGIME_TRANSITION_BREAK,
    POST_CRISIS_RESTABILIZATION,
    DISLOCATION_REVERSION,
    BREAKOUT_RETEST_CONFIRM,
    LIQUIDITY_VACUUM_RECLAIM,
    VOL_EXPANSION_BREAKOUT,
    EXHAUSTION_REVERSAL,
    TREND_PULLBACK,
})

# ---------------------------------------------------------------------------
# Direction constants
# ---------------------------------------------------------------------------
LONG = "LONG"
SHORT = "SHORT"
_VALID_DIRECTIONS = frozenset({LONG, SHORT})


# ---------------------------------------------------------------------------
# Config — frozen, shadow-only, authority="none" (hard-coded)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FPSv1Config:
    """FPS v1 configuration. Authority and mode are structurally locked."""

    enabled: bool = True
    mode: str = "shadow_only"
    authority: str = "none"

    # Setup class toggles
    setup_classes: frozenset = frozenset(_VALID_SETUP_CLASSES)

    # Fee bridge params (from runtime.yaml fee_gate)
    taker_fee_rate: float = 0.0004
    maker_fee_rate: float = 0.0002
    fee_buffer_mult: float = 1.5

    # Influence flags — all permanently false
    can_influence_sizing: bool = False
    can_influence_entry: bool = False
    can_influence_exit: bool = False
    can_influence_routing: bool = False

    def __post_init__(self) -> None:
        # Structural locks — cannot be overridden by config
        if self.authority != "none":
            object.__setattr__(self, "authority", "none")
        if self.mode != "shadow_only":
            object.__setattr__(self, "mode", "shadow_only")
        if self.can_influence_sizing or self.can_influence_entry or \
           self.can_influence_exit or self.can_influence_routing:
            object.__setattr__(self, "can_influence_sizing", False)
            object.__setattr__(self, "can_influence_entry", False)
            object.__setattr__(self, "can_influence_exit", False)
            object.__setattr__(self, "can_influence_routing", False)


def load_fps_config(raw: Optional[Dict[str, Any]] = None) -> FPSv1Config:
    """Load FPS v1 config from dict. Forces authority='none' regardless of input."""
    if raw is None:
        return FPSv1Config()
    setup_raw = raw.get("setup_classes", list(_VALID_SETUP_CLASSES))
    valid_classes = frozenset(s for s in setup_raw if s in _VALID_SETUP_CLASSES)
    return FPSv1Config(
        enabled=bool(raw.get("enabled", True)),
        mode="shadow_only",         # forced
        authority="none",           # forced
        setup_classes=valid_classes if valid_classes else frozenset(_VALID_SETUP_CLASSES),
        taker_fee_rate=float(raw.get("taker_fee_rate", 0.0004)),
        maker_fee_rate=float(raw.get("maker_fee_rate", 0.0002)),
        fee_buffer_mult=float(raw.get("fee_buffer_mult", 1.5)),
        can_influence_sizing=False,   # forced
        can_influence_entry=False,    # forced
        can_influence_exit=False,     # forced
        can_influence_routing=False,  # forced
    )


# ---------------------------------------------------------------------------
# Evaluation context — frozen market snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FPSEvalContext:
    """Immutable snapshot of market state for FPS evaluation."""

    symbol: str
    timestamp: float  # unix seconds

    # Regime state
    regime_current: str
    regime_previous: str
    regime_age_bars: int
    regime_confidence: float
    crisis_flag: bool

    # Market microstructure
    atr_pct: float             # ATR as fraction of price
    volume_z: float            # z-score of recent volume vs baseline
    spread_bps: float          # bid-ask spread in basis points
    price: float

    # Direction intent (from Hydra signal, not FPS decision)
    proposed_direction: str    # LONG or SHORT

    # Microstructure inputs for hypothesis-driven setup classes (all optional)
    atr_percentile: float = 50.0       # ATR percentile over rolling window (0-100)
    range_ratio: float = 1.0           # current_range / rolling_avg_range
    local_breakout_dir: Optional[str] = None  # "HIGH" or "LOW" — break direction
    zscore: float = 0.0                # z-score of price vs rolling mean
    rsi: float = 50.0                  # RSI value (0-100)
    continuation_failed: bool = False  # no higher high / lower low over N bars
    wick_rejection: bool = False       # wick rejection or momentum slowdown
    ema_slope: float = 0.0             # EMA slope (positive = uptrend)
    ema_aligned: bool = False          # price on correct side of major EMA
    pullback_atr_ratio: float = 0.0    # retracement depth in ATR units
    momentum_reacceleration: bool = False  # momentum resumption confirmed


# ---------------------------------------------------------------------------
# Result — frozen, no numeric authority field
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FPSResult:
    """Evaluation result. Contains NO score, rank, band, percentile, or weight."""

    verdict: str                          # PERMIT_CANDIDATE | ABSTAIN | DENY_STRUCTURAL
    setup_class: Optional[str] = None     # which setup matched (None if ABSTAIN)
    direction: Optional[str] = None       # validated direction (None if not PERMIT)
    gate_trace: tuple = ()                # ordered gate pass/fail trace
    deny_reason: Optional[str] = None     # reason if DENY_STRUCTURAL
    snapshot_hash: str = ""               # SHA-256 of input context for determinism

    def __post_init__(self) -> None:
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError(f"Invalid verdict: {self.verdict!r}. Must be one of {_VALID_VERDICTS}")
        if self.direction is not None and self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction: {self.direction!r}. Must be one of {_VALID_DIRECTIONS}")


# ---------------------------------------------------------------------------
# Gate 1 — Structural admissibility (setup class detection)
# ---------------------------------------------------------------------------
def _gate1_structural_admissibility(
    ctx: FPSEvalContext,
    cfg: FPSv1Config,
) -> Optional[str]:
    """Return matching setup class or None (ABSTAIN)."""

    # REGIME_TRANSITION_BREAK
    if REGIME_TRANSITION_BREAK in cfg.setup_classes:
        if (ctx.regime_previous != ctx.regime_current
                and ctx.regime_age_bars <= 3
                and ctx.regime_confidence >= 0.55
                and ctx.regime_current in ("TREND_UP", "TREND_DOWN", "BREAKOUT")):
            return REGIME_TRANSITION_BREAK

    # POST_CRISIS_RESTABILIZATION
    if POST_CRISIS_RESTABILIZATION in cfg.setup_classes:
        if (ctx.regime_previous == "CRISIS"
                and ctx.regime_current not in ("CRISIS", "CHOPPY")
                and not ctx.crisis_flag
                and ctx.regime_confidence >= 0.50):
            return POST_CRISIS_RESTABILIZATION

    # DISLOCATION_REVERSION
    if DISLOCATION_REVERSION in cfg.setup_classes:
        if (ctx.regime_current == "MEAN_REVERT"
                and ctx.volume_z >= 1.5
                and ctx.regime_confidence >= 0.50):
            return DISLOCATION_REVERSION

    # BREAKOUT_RETEST_CONFIRM
    if BREAKOUT_RETEST_CONFIRM in cfg.setup_classes:
        if (ctx.regime_current == "BREAKOUT"
                and ctx.regime_age_bars >= 2
                and ctx.regime_confidence >= 0.60):
            return BREAKOUT_RETEST_CONFIRM

    # LIQUIDITY_VACUUM_RECLAIM
    if LIQUIDITY_VACUUM_RECLAIM in cfg.setup_classes:
        if (ctx.regime_previous in ("CHOPPY", "CRISIS")
                and ctx.regime_current not in ("CHOPPY", "CRISIS")
                and ctx.spread_bps < 2.0
                and ctx.volume_z >= 1.0):
            return LIQUIDITY_VACUUM_RECLAIM

    # --- Hypothesis-driven classes ---

    # VOL_EXPANSION_BREAKOUT: compression → sudden expansion → directional break
    if VOL_EXPANSION_BREAKOUT in cfg.setup_classes:
        if (ctx.atr_percentile < 20.0
                and ctx.range_ratio > 1.8
                and ctx.local_breakout_dir in ("HIGH", "LOW")):
            return VOL_EXPANSION_BREAKOUT

    # EXHAUSTION_REVERSAL: statistical extreme → failure → reversal
    if EXHAUSTION_REVERSAL in cfg.setup_classes:
        if (abs(ctx.zscore) >= 2.0
                and (ctx.rsi >= 75.0 or ctx.rsi <= 25.0)
                and ctx.continuation_failed
                and ctx.wick_rejection):
            return EXHAUSTION_REVERSAL

    # TREND_PULLBACK: strong trend → controlled pullback → continuation
    # Slope-regime coherence enforced here (not deferred to gate 2)
    if TREND_PULLBACK in cfg.setup_classes:
        if (ctx.ema_aligned
                and 0.5 <= ctx.pullback_atr_ratio <= 1.5
                and ctx.momentum_reacceleration
                and ctx.regime_current in ("TREND_UP", "TREND_DOWN")
                and ((ctx.ema_slope > 0 and ctx.regime_current == "TREND_UP")
                     or (ctx.ema_slope < 0 and ctx.regime_current == "TREND_DOWN"))):
            return TREND_PULLBACK

    return None  # no setup matched → ABSTAIN


# ---------------------------------------------------------------------------
# Gate 2 — Direction validity (regime → permitted direction)
# ---------------------------------------------------------------------------
_REGIME_DIRECTION_MAP: Dict[str, Optional[str]] = {
    "TREND_UP": LONG,
    "TREND_DOWN": SHORT,
    "BREAKOUT": None,      # ambiguous — both permitted
    "MEAN_REVERT": None,   # ambiguous — both permitted
    "CHOPPY": None,        # should not reach here (filtered by gate 1)
    "CRISIS": None,        # should not reach here (filtered by gate 1)
}


def _gate2_direction_validity(
    ctx: FPSEvalContext,
    setup_class: str,
) -> Optional[str]:
    """Validate proposed direction against regime/setup. Return direction or None."""

    # --- Hypothesis-driven classes: direction from setup inputs, not regime ---

    if setup_class == VOL_EXPANSION_BREAKOUT:
        if ctx.local_breakout_dir == "HIGH" and ctx.proposed_direction == LONG:
            return LONG
        if ctx.local_breakout_dir == "LOW" and ctx.proposed_direction == SHORT:
            return SHORT
        return None

    if setup_class == EXHAUSTION_REVERSAL:
        if ctx.zscore >= 2.0 and ctx.proposed_direction == SHORT:
            return SHORT
        if ctx.zscore <= -2.0 and ctx.proposed_direction == LONG:
            return LONG
        return None

    if setup_class == TREND_PULLBACK:
        if ctx.ema_slope > 0 and ctx.proposed_direction == LONG:
            return LONG
        if ctx.ema_slope < 0 and ctx.proposed_direction == SHORT:
            return SHORT
        return None

    # --- Regime-event classes: direction from regime ---

    required_dir = _REGIME_DIRECTION_MAP.get(ctx.regime_current)

    if required_dir is not None:
        if ctx.proposed_direction == required_dir:
            return required_dir
        return None

    if ctx.proposed_direction in _VALID_DIRECTIONS:
        return ctx.proposed_direction
    return None


# ---------------------------------------------------------------------------
# Gate 3 — Fee bridge (ATR edge must clear round-trip fees)
# ---------------------------------------------------------------------------
def _gate3_fee_bridge(
    ctx: FPSEvalContext,
    cfg: FPSv1Config,
) -> bool:
    """Return True if expected edge clears fee hurdle."""
    round_trip_fee = 2.0 * cfg.taker_fee_rate * cfg.fee_buffer_mult
    expected_edge = ctx.atr_pct * 0.5  # conservative: half ATR as expected capture
    return expected_edge > round_trip_fee


# ---------------------------------------------------------------------------
# Gate 4 — Shadow emit (verdict routing)
# ---------------------------------------------------------------------------
def _gate4_emit(
    setup_class: Optional[str],
    direction: Optional[str],
    fee_clears: bool,
    gate_trace: List[str],
) -> FPSResult:
    """Route to final verdict based on gate results."""
    # No setup matched → ABSTAIN
    if setup_class is None:
        return FPSResult(
            verdict=ABSTAIN,
            gate_trace=tuple(gate_trace),
        )

    # Direction invalid → ABSTAIN
    if direction is None:
        gate_trace.append("g2:direction_mismatch")
        return FPSResult(
            verdict=ABSTAIN,
            setup_class=setup_class,
            gate_trace=tuple(gate_trace),
        )

    # Fee bridge failed → DENY_STRUCTURAL
    if not fee_clears:
        gate_trace.append("g3:fee_deny")
        return FPSResult(
            verdict=DENY_STRUCTURAL,
            setup_class=setup_class,
            direction=direction,
            gate_trace=tuple(gate_trace),
            deny_reason="fee_bridge_insufficient",
        )

    # All gates passed → PERMIT_CANDIDATE
    gate_trace.append("g3:fee_pass")
    return FPSResult(
        verdict=PERMIT_CANDIDATE,
        setup_class=setup_class,
        direction=direction,
        gate_trace=tuple(gate_trace),
    )


# ---------------------------------------------------------------------------
# Context hasher (determinism proof)
# ---------------------------------------------------------------------------
def _hash_context(ctx: FPSEvalContext) -> str:
    """SHA-256 of frozen context for determinism verification."""
    raw = json.dumps(asdict(ctx), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Main evaluate — pure function (no side effects)
# ---------------------------------------------------------------------------
def evaluate(ctx: FPSEvalContext, cfg: Optional[FPSv1Config] = None) -> FPSResult:
    """
    Evaluate market snapshot through 4-gate pipeline.

    Pure function: same (ctx, cfg) → identical FPSResult every time.
    Does NOT log, does NOT mutate state, does NOT call exchange.
    """
    if cfg is None:
        cfg = FPSv1Config()

    if not cfg.enabled:
        return FPSResult(verdict=ABSTAIN, gate_trace=("disabled",))

    gate_trace: List[str] = []
    ctx_hash = _hash_context(ctx)

    # Gate 1 — structural admissibility
    setup_class = _gate1_structural_admissibility(ctx, cfg)
    if setup_class is not None:
        gate_trace.append(f"g1:{setup_class}")
    else:
        gate_trace.append("g1:no_match")

    # Gate 2 — direction validity (only if gate 1 passed)
    direction: Optional[str] = None
    if setup_class is not None:
        direction = _gate2_direction_validity(ctx, setup_class)
        if direction is not None:
            gate_trace.append(f"g2:{direction}")
        else:
            gate_trace.append("g2:direction_mismatch")

    # Gate 3 — fee bridge (only if gate 2 passed)
    fee_clears = False
    if direction is not None:
        fee_clears = _gate3_fee_bridge(ctx, cfg)

    # Gate 4 — emit verdict
    result = _gate4_emit(setup_class, direction, fee_clears, gate_trace)

    # Attach snapshot hash
    object.__setattr__(result, "snapshot_hash", ctx_hash)

    return result


# ---------------------------------------------------------------------------
# Shadow JSONL append (fail-open)
# ---------------------------------------------------------------------------
def append_shadow_record(
    result: FPSResult,
    ctx: FPSEvalContext,
    cfg: FPSv1Config,
    log_path: Optional[Path] = None,
) -> None:
    """Append shadow evaluation record to JSONL. Fail-open: never raises."""
    try:
        path = log_path or _SHADOW_LOG
        record = {
            "ts": time.time(),
            "schema": "fps_shadow_v1",
            "symbol": ctx.symbol,
            "verdict": result.verdict,
            "setup_class": result.setup_class,
            "direction": result.direction,
            "gate_trace": list(result.gate_trace),
            "deny_reason": result.deny_reason,
            "snapshot_hash": result.snapshot_hash,
            "regime_current": ctx.regime_current,
            "regime_previous": ctx.regime_previous,
            "regime_age_bars": ctx.regime_age_bars,
            "regime_confidence": ctx.regime_confidence,
            "atr_pct": ctx.atr_pct,
            "volume_z": ctx.volume_z,
            "spread_bps": ctx.spread_bps,
            "price": ctx.price,
            "proposed_direction": ctx.proposed_direction,
            "atr_percentile": ctx.atr_percentile,
            "range_ratio": ctx.range_ratio,
            "local_breakout_dir": ctx.local_breakout_dir,
            "zscore": ctx.zscore,
            "rsi": ctx.rsi,
            "ema_slope": ctx.ema_slope,
            "pullback_atr_ratio": ctx.pullback_atr_ratio,
            "continuation_failed": ctx.continuation_failed,
            "wick_rejection": ctx.wick_rejection,
            "ema_aligned": ctx.ema_aligned,
            "momentum_reacceleration": ctx.momentum_reacceleration,
            "config_mode": cfg.mode,
            "config_authority": cfg.authority,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        LOG.debug("[fps_v1] shadow log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Integration helpers (shadow telemetry bridges)
# ---------------------------------------------------------------------------
def build_selector_mirror(result: FPSResult, ctx: FPSEvalContext) -> Dict[str, Any]:
    """Build fps_v1_* namespaced keys for selector shadow events."""
    return {
        "fps_v1_verdict": result.verdict,
        "fps_v1_setup_class": result.setup_class,
        "fps_v1_direction": result.direction,
        "fps_v1_gate_trace": list(result.gate_trace),
        "fps_v1_deny_reason": result.deny_reason,
        "fps_v1_snapshot_hash": result.snapshot_hash,
        "fps_v1_regime": ctx.regime_current,
        "fps_v1_regime_prev": ctx.regime_previous,
    }


def build_dle_shadow_context(
    result: FPSResult,
    ctx: FPSEvalContext,
) -> Dict[str, Any]:
    """Build context dict for DLE shadow_build_chain integration."""
    return {
        "fps_v1_verdict": result.verdict,
        "fps_v1_setup_class": result.setup_class,
        "fps_v1_direction": result.direction,
        "fps_v1_gate_trace": list(result.gate_trace),
        "fps_v1_snapshot_hash": result.snapshot_hash,
        "fps_v1_symbol": ctx.symbol,
        "fps_v1_regime": ctx.regime_current,
        "fps_v1_atr_pct": ctx.atr_pct,
        "fps_v1_fee_cleared": result.deny_reason != "fee_bridge_insufficient",
    }


# ---------------------------------------------------------------------------
# Setup sparsity invariant — prevent categorical entropy
# ---------------------------------------------------------------------------
_PERMIT_RATE_FLOOR = 0.01   # <1% → dead surface
_PERMIT_RATE_CEIL = 0.15    # >15% → drift / too loose
_MAX_SETUP_CLASSES = 8      # hard cap on taxonomy size (5 regime + 3 hypothesis)


@dataclass(frozen=True)
class SparsityReport:
    """Per-symbol/regime permit rate + overlap diagnostics."""
    total_evals: int
    permit_count: int
    permit_rate: float
    by_symbol: Dict[str, float]          # symbol → permit rate
    by_regime: Dict[str, float]          # regime → permit rate
    by_setup_class: Dict[str, int]       # setup_class → count
    overlap_violations: List[Dict[str, Any]]  # co-firing events
    alerts: List[str]


def compute_sparsity(
    log_path: Optional[Path] = None,
    *,
    permit_rate_floor: float = _PERMIT_RATE_FLOOR,
    permit_rate_ceil: float = _PERMIT_RATE_CEIL,
) -> SparsityReport:
    """
    Analyse shadow log for setup-class entropy degradation.

    Pure read-only — loads JSONL, computes rates, returns report.
    """
    path = log_path or _SHADOW_LOG
    records: List[Dict[str, Any]] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    total = len(records)
    if total == 0:
        return SparsityReport(
            total_evals=0, permit_count=0, permit_rate=0.0,
            by_symbol={}, by_regime={}, by_setup_class={},
            overlap_violations=[], alerts=["no_data"],
        )

    permits = [r for r in records if r.get("verdict") == PERMIT_CANDIDATE]
    permit_count = len(permits)
    permit_rate = permit_count / total

    # Per-symbol permit rate
    sym_total: Dict[str, int] = {}
    sym_permit: Dict[str, int] = {}
    for r in records:
        s = r.get("symbol", "?")
        sym_total[s] = sym_total.get(s, 0) + 1
        if r.get("verdict") == PERMIT_CANDIDATE:
            sym_permit[s] = sym_permit.get(s, 0) + 1
    by_symbol = {s: sym_permit.get(s, 0) / sym_total[s] for s in sym_total}

    # Per-regime permit rate
    reg_total: Dict[str, int] = {}
    reg_permit: Dict[str, int] = {}
    for r in records:
        rg = r.get("regime_current", "?")
        reg_total[rg] = reg_total.get(rg, 0) + 1
        if r.get("verdict") == PERMIT_CANDIDATE:
            reg_permit[rg] = reg_permit.get(rg, 0) + 1
    by_regime = {rg: reg_permit.get(rg, 0) / reg_total[rg] for rg in reg_total}

    # Setup class distribution
    by_setup_class: Dict[str, int] = {}
    for r in permits:
        sc = r.get("setup_class", "?")
        by_setup_class[sc] = by_setup_class.get(sc, 0) + 1

    # Overlap detection: same symbol + same timestamp window (within 1s)
    # having different setup classes → mutual exclusivity violation
    overlap_violations: List[Dict[str, Any]] = []
    permits_sorted = sorted(permits, key=lambda r: (r.get("symbol", ""), r.get("ts", 0)))
    for i in range(1, len(permits_sorted)):
        prev, curr = permits_sorted[i - 1], permits_sorted[i]
        if (prev.get("symbol") == curr.get("symbol")
                and abs(prev.get("ts", 0) - curr.get("ts", 0)) < 1.0
                and prev.get("setup_class") != curr.get("setup_class")):
            overlap_violations.append({
                "symbol": curr.get("symbol"),
                "ts": curr.get("ts"),
                "classes": [prev.get("setup_class"), curr.get("setup_class")],
            })

    # Alerts
    alerts: List[str] = []
    if permit_rate > permit_rate_ceil:
        alerts.append(f"DRIFT: global permit rate {permit_rate:.1%} > {permit_rate_ceil:.0%} ceiling")
    if 0 < permit_rate < permit_rate_floor:
        alerts.append(f"DEAD: global permit rate {permit_rate:.1%} < {permit_rate_floor:.0%} floor")
    for s, rate in by_symbol.items():
        if rate > permit_rate_ceil:
            alerts.append(f"DRIFT: {s} permit rate {rate:.1%} > ceiling")
    for rg, rate in by_regime.items():
        if rate > permit_rate_ceil:
            alerts.append(f"DRIFT: regime {rg} permit rate {rate:.1%} > ceiling")
    if overlap_violations:
        alerts.append(f"OVERLAP: {len(overlap_violations)} mutual-exclusivity violations")
    if len(by_setup_class) > _MAX_SETUP_CLASSES:
        alerts.append(f"ENTROPY: {len(by_setup_class)} setup classes > {_MAX_SETUP_CLASSES} cap")

    return SparsityReport(
        total_evals=total,
        permit_count=permit_count,
        permit_rate=permit_rate,
        by_symbol=by_symbol,
        by_regime=by_regime,
        by_setup_class=by_setup_class,
        overlap_violations=overlap_violations,
        alerts=alerts,
    )


# ---------------------------------------------------------------------------
# Temporal clustering — detect permit bursts (non-independence)
# ---------------------------------------------------------------------------
_BURST_WINDOW_1H = 3600.0    # 1-hour window (seconds)
_BURST_WINDOW_4H = 14400.0   # 4-hour window (seconds)
_BURST_RATE_CEIL = 0.30       # >30% permit rate within window → burst
_MAX_CONSECUTIVE = 5          # max consecutive permits before alert


@dataclass(frozen=True)
class BurstReport:
    """Temporal clustering diagnostics for permit independence."""
    total_permits: int
    windows_checked_1h: int
    windows_checked_4h: int
    max_rate_1h: float                           # peak 1h permit rate
    max_rate_4h: float                           # peak 4h permit rate
    burst_windows_1h: List[Dict[str, Any]]       # windows exceeding ceiling
    burst_windows_4h: List[Dict[str, Any]]       # windows exceeding ceiling
    max_consecutive: int                          # longest consecutive permit streak
    consecutive_streaks: List[Dict[str, Any]]     # streaks exceeding cap
    alerts: List[str]


def compute_burst_risk(
    log_path: Optional[Path] = None,
    *,
    burst_rate_ceil: float = _BURST_RATE_CEIL,
    max_consecutive: int = _MAX_CONSECUTIVE,
) -> BurstReport:
    """
    Detect temporal clustering in permit emissions.

    Checks rolling-window permit density and consecutive permit streaks.
    Pure read-only — loads JSONL, computes, returns report.
    """
    path = log_path or _SHADOW_LOG
    records: List[Dict[str, Any]] = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    if not records:
        return BurstReport(
            total_permits=0,
            windows_checked_1h=0, windows_checked_4h=0,
            max_rate_1h=0.0, max_rate_4h=0.0,
            burst_windows_1h=[], burst_windows_4h=[],
            max_consecutive=0, consecutive_streaks=[],
            alerts=["no_data"],
        )

    # Sort by timestamp
    records.sort(key=lambda r: r.get("ts", 0))
    timestamps = [r.get("ts", 0) for r in records]
    is_permit = [r.get("verdict") == PERMIT_CANDIDATE for r in records]
    total_permits = sum(is_permit)

    # --- Rolling window analysis ---
    def _check_windows(window_s: float) -> tuple:
        """Slide window across records, compute per-window permit rate."""
        if len(records) < 2:
            return 0.0, [], 0
        burst_windows: List[Dict[str, Any]] = []
        max_rate = 0.0
        windows_checked = 0

        # Use non-overlapping windows for clean counting
        t_start = timestamps[0]
        t_end = timestamps[-1]
        t = t_start
        while t < t_end:
            w_end = t + window_s
            # Records in this window
            w_permits = 0
            w_total = 0
            for j, ts_j in enumerate(timestamps):
                if t <= ts_j < w_end:
                    w_total += 1
                    if is_permit[j]:
                        w_permits += 1
            if w_total > 0:
                windows_checked += 1
                rate = w_permits / w_total
                if rate > max_rate:
                    max_rate = rate
                if rate > burst_rate_ceil and w_permits >= 3:
                    burst_windows.append({
                        "window_start": t,
                        "window_end": w_end,
                        "total": w_total,
                        "permits": w_permits,
                        "rate": round(rate, 4),
                    })
            t += window_s

        return max_rate, burst_windows, windows_checked

    max_rate_1h, burst_1h, wc_1h = _check_windows(_BURST_WINDOW_1H)
    max_rate_4h, burst_4h, wc_4h = _check_windows(_BURST_WINDOW_4H)

    # --- Consecutive permit streak detection ---
    streaks: List[Dict[str, Any]] = []
    current_streak = 0
    streak_start_ts = 0.0
    longest_streak = 0

    for i, perm in enumerate(is_permit):
        if perm:
            if current_streak == 0:
                streak_start_ts = timestamps[i]
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
        else:
            if current_streak > max_consecutive:
                streaks.append({
                    "start_ts": streak_start_ts,
                    "end_ts": timestamps[i - 1],
                    "length": current_streak,
                    "symbol": records[i - 1].get("symbol", "?"),
                })
            current_streak = 0

    # Handle trailing streak
    if current_streak > max_consecutive:
        streaks.append({
            "start_ts": streak_start_ts,
            "end_ts": timestamps[-1],
            "length": current_streak,
            "symbol": records[-1].get("symbol", "?"),
        })

    # --- Alerts ---
    alerts: List[str] = []
    if burst_1h:
        alerts.append(
            f"BURST_1H: {len(burst_1h)} windows with permit rate "
            f"> {burst_rate_ceil:.0%} (peak {max_rate_1h:.1%})"
        )
    if burst_4h:
        alerts.append(
            f"BURST_4H: {len(burst_4h)} windows with permit rate "
            f"> {burst_rate_ceil:.0%} (peak {max_rate_4h:.1%})"
        )
    if streaks:
        alerts.append(
            f"CONSECUTIVE: {len(streaks)} streaks of "
            f"> {max_consecutive} consecutive permits (longest {longest_streak})"
        )

    return BurstReport(
        total_permits=total_permits,
        windows_checked_1h=wc_1h,
        windows_checked_4h=wc_4h,
        max_rate_1h=max_rate_1h,
        max_rate_4h=max_rate_4h,
        burst_windows_1h=burst_1h,
        burst_windows_4h=burst_4h,
        max_consecutive=longest_streak,
        consecutive_streaks=streaks,
        alerts=alerts,
    )


# ---------------------------------------------------------------------------
# Executor integration — single-call shadow evaluation
# ---------------------------------------------------------------------------
_FPS_CFG_CACHE: Optional[FPSv1Config] = None


def evaluate_shadow_for_intent(
    intent: Dict[str, Any],
    sentinel_state: Dict[str, Any],
) -> None:
    """
    Executor-callable shadow evaluation.  Builds FPSEvalContext from an
    executor intent dict + sentinel_x state, evaluates, and appends to
    shadow JSONL.  Completely fail-open: catches ALL exceptions so it can
    never block the trading loop.

    This function is the ONLY entry point the executor should use.
    """
    try:
        global _FPS_CFG_CACHE  # noqa: PLW0603
        if _FPS_CFG_CACHE is None:
            import json as _json
            _strat = {}
            try:
                _p = Path("config/strategy_config.json")
                if _p.exists():
                    _strat = _json.loads(_p.read_text(encoding="utf-8"))
            except Exception:
                pass
            _fps_block = (_strat.get("futures_permit_surface_v1") or {})
            _FPS_CFG_CACHE = load_fps_config(_fps_block)

        if not _FPS_CFG_CACHE.enabled:
            return

        # --- Extract fields from intent + sentinel state ---
        symbol = str(intent.get("symbol", ""))
        if not symbol:
            return

        price = float(intent.get("price", 0.0))
        if price <= 0:
            return

        signal = str(intent.get("signal", intent.get("side", ""))).upper()
        if signal in ("BUY", "LONG"):
            proposed_direction = "LONG"
        elif signal in ("SELL", "SHORT"):
            proposed_direction = "SHORT"
        else:
            proposed_direction = "LONG"  # default; gate 2 will filter

        # Regime from sentinel state
        regime_current = str(sentinel_state.get("primary_regime", "UNKNOWN"))
        # previous_regime: sentinel uses history_meta.last_primary or
        # meta.regime_regret.prev_regime — fall back to top-level key if present.
        _hist = sentinel_state.get("history_meta") or {}
        _regret = (sentinel_state.get("meta") or {}).get("regime_regret") or {}
        regime_previous = str(
            sentinel_state.get("previous_regime")
            or _regret.get("prev_regime")
            or _hist.get("last_primary")
            or ""
        )
        # regime_age_bars: consecutive_count from history_meta, or
        # cycles_in_prev from regime_regret as fallback.
        _raw_age = (
            sentinel_state.get("regime_age_bars")
            or _hist.get("consecutive_count")
            or _regret.get("cycles_in_prev")
            or 0
        )
        regime_age_bars = int(_raw_age)
        regime_probs = sentinel_state.get("regime_probs") or {}
        regime_confidence = float(
            max(regime_probs.values()) if regime_probs else 0.0
        )
        crisis_flag = bool(sentinel_state.get("crisis_flag", False))

        # Microstructure — use intent metadata or sensible defaults
        meta = intent.get("metadata") or {}
        atr_pct = float(meta.get("atr_pct", intent.get("atr_pct", 0.02)))
        volume_z = float(meta.get("volume_z", 0.0))
        spread_bps = float(meta.get("spread_bps", 1.0))

        # Optional hypothesis fields (defaults are safe non-triggering values)
        zscore = float(meta.get("zscore", intent.get("zscore", 0.0)))
        rsi = float(meta.get("rsi", 50.0))
        ema_slope = float(meta.get("ema_slope", 0.0))

        # Hypothesis-class fields — omitted keys stay at FPSEvalContext defaults
        _hyp_kwargs: Dict[str, Any] = {}
        if "atr_percentile" in meta:
            _hyp_kwargs["atr_percentile"] = float(meta["atr_percentile"])
        if "range_ratio" in meta:
            _hyp_kwargs["range_ratio"] = float(meta["range_ratio"])
        if "local_breakout_dir" in meta:
            _hyp_kwargs["local_breakout_dir"] = meta["local_breakout_dir"]
        if "continuation_failed" in meta:
            _hyp_kwargs["continuation_failed"] = bool(meta["continuation_failed"])
        if "wick_rejection" in meta:
            _hyp_kwargs["wick_rejection"] = bool(meta["wick_rejection"])
        if "ema_aligned" in meta:
            _hyp_kwargs["ema_aligned"] = bool(meta["ema_aligned"])
        if "pullback_atr_ratio" in meta:
            _hyp_kwargs["pullback_atr_ratio"] = float(meta["pullback_atr_ratio"])
        if "momentum_reacceleration" in meta:
            _hyp_kwargs["momentum_reacceleration"] = bool(meta["momentum_reacceleration"])

        ctx = FPSEvalContext(
            symbol=symbol,
            timestamp=time.time(),
            regime_current=regime_current,
            regime_previous=regime_previous,
            regime_age_bars=regime_age_bars,
            regime_confidence=regime_confidence,
            crisis_flag=crisis_flag,
            atr_pct=atr_pct,
            volume_z=volume_z,
            spread_bps=spread_bps,
            price=price,
            proposed_direction=proposed_direction,
            zscore=zscore,
            rsi=rsi,
            ema_slope=ema_slope,
            **_hyp_kwargs,
        )

        result = evaluate(ctx, _FPS_CFG_CACHE)
        append_shadow_record(result, ctx, _FPS_CFG_CACHE)

    except Exception as exc:
        LOG.debug("[fps_v1] shadow evaluation failed (non-blocking): %s", exc)
