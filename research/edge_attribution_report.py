"""
Episode Edge Attribution Report (v7.9-TE1)

Proves whether predicted edge corresponds to realized outcomes,
and provides data to calibrate k_atr and confidence mapping.

Usage:
    python -m research.edge_attribution_report
    # or
    python research/edge_attribution_report.py [--episodes path] [--output-dir path]

Outputs:
    reports/edge_attribution/latest.json   — structured report
    reports/edge_attribution/latest.md     — human-readable summary
    reports/edge_attribution/latest.csv    — flat CSV for dashboard ingest

Data sources:
    - Episode ledger (logs/state/episode_ledger.json)
    - Fee gate events (logs/execution/fee_gate_events.jsonl) — for true_edge_v1 fields

The report is READ-ONLY observability.  It does NOT influence execution.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("edge_attribution")

# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_EPISODE_LEDGER_PATH = Path("logs/state/episode_ledger.json")
DEFAULT_FEE_GATE_LOG_PATH = Path("logs/execution/fee_gate_events.jsonl")
DEFAULT_OUTPUT_DIR = Path("reports/edge_attribution")

NUM_EDGE_DECILES = 10
CONFIDENCE_BUCKET_WIDTH = 0.05  # 5pp buckets: 0.50-0.55, 0.55-0.60, ...


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Unified record joining episode data with predicted edge fields."""

    episode_id: str
    symbol: str
    strategy: str
    side: str
    regime_at_entry: str
    regime_at_exit: str

    # Timing
    entry_ts: str
    exit_ts: str
    holding_time_s: float

    # Execution
    notional_usd: float
    fees_usd: float
    slippage_est_usd: float

    # Predicted edge (from true_edge_v1 or fallback)
    pred_expected_edge_pct: float
    pred_expected_edge_usd: float
    confidence: float
    atr_pct: float
    k_atr: float
    adv: float
    edge_source: str  # "atr_conf_v1" or "fallback_proxy"

    # Realized outcome
    realized_pnl_usd: float  # net PnL (after fees)
    realized_move_pct: float  # |exit_price - entry_price| / entry_price

    # Raw prices for move calculation
    avg_entry_price: float = 0.0
    avg_exit_price: float = 0.0

    def is_win(self) -> bool:
        return self.realized_pnl_usd > 0.0


@dataclass
class BucketStats:
    """Aggregated statistics for a bucket of trades."""

    bucket_label: str
    count: int = 0
    mean_pred_edge_pct: float = 0.0
    median_pred_edge_pct: float = 0.0
    mean_realized_pnl: float = 0.0
    median_realized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    win_rate: float = 0.0
    mean_fees: float = 0.0
    profit_factor: float = 0.0  # gross_wins / abs(gross_losses)
    mean_holding_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KAtrSuggestion:
    """k_atr tuning suggestion based on realized moves vs ATR."""

    regime: str
    sample_count: int
    current_k_atr: float
    median_move_multiple: float  # median(|move_pct| / atr_pct)
    mean_move_multiple: float
    suggested_k_atr: float  # median_move_multiple

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributionReport:
    """Full edge attribution report."""

    generated_at: str
    total_trades: int
    trades_with_atr_edge: int
    trades_with_fallback: int

    # A) Calibration curve by predicted edge decile
    edge_decile_buckets: List[BucketStats] = field(default_factory=list)

    # B) Reliability by confidence bucket
    confidence_buckets: List[BucketStats] = field(default_factory=list)

    # C) k_atr tuning suggestions
    k_atr_suggestions: List[KAtrSuggestion] = field(default_factory=list)

    # D) Strategy breakdown
    strategy_buckets: Dict[str, List[BucketStats]] = field(default_factory=dict)

    # D) Regime breakdown
    regime_buckets: Dict[str, List[BucketStats]] = field(default_factory=dict)

    # Summary
    overall_win_rate: float = 0.0
    overall_mean_pnl: float = 0.0
    overall_profit_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "generated_at": self.generated_at,
            "total_trades": self.total_trades,
            "trades_with_atr_edge": self.trades_with_atr_edge,
            "trades_with_fallback": self.trades_with_fallback,
            "overall_win_rate": round(self.overall_win_rate, 4),
            "overall_mean_pnl": round(self.overall_mean_pnl, 4),
            "overall_profit_factor": round(self.overall_profit_factor, 4),
            "edge_decile_buckets": [b.to_dict() for b in self.edge_decile_buckets],
            "confidence_buckets": [b.to_dict() for b in self.confidence_buckets],
            "k_atr_suggestions": [s.to_dict() for s in self.k_atr_suggestions],
            "strategy_buckets": {
                k: [b.to_dict() for b in v]
                for k, v in self.strategy_buckets.items()
            },
            "regime_buckets": {
                k: [b.to_dict() for b in v]
                for k, v in self.regime_buckets.items()
            },
        }
        return d


# ── Loading ────────────────────────────────────────────────────────────────

def _load_episodes(path: Path) -> List[Dict[str, Any]]:
    """Load episodes from the episode ledger JSON file."""
    if not path.exists():
        LOG.warning("Episode ledger not found: %s", path)
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("episodes", [])
    except Exception as e:
        LOG.error("Failed to load episodes: %s", e)
        return []


def _load_fee_gate_events(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load fee gate events, indexed for joining with episodes.
    
    Returns two indices:
      - by_intent_id: intent_id → event (exact join, preferred)
      - by_symbol_side: (symbol, side) → latest event (fallback)
    
    Packed into a single dict with keys:
      {"by_intent": {...}, "by_symbol_side": {...}}
    """
    by_intent: Dict[str, Dict[str, Any]] = {}
    by_symbol_side: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return {"by_intent": by_intent, "by_symbol_side": by_symbol_side}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                # Primary index: intent_id (exact join)
                iid = ev.get("intent_id", "")
                if iid:
                    by_intent[iid] = ev
                # Secondary index: symbol+side (fallback)
                symbol = ev.get("symbol", "")
                side = ev.get("side", "")
                if symbol and side:
                    by_symbol_side[f"{symbol}_{side}"] = ev
    except Exception as e:
        LOG.error("Failed to load fee gate events: %s", e)
    return {"by_intent": by_intent, "by_symbol_side": by_symbol_side}


def _find_edge_event(
    fee_gate_events: Dict[str, Dict[str, Any]],
    symbol: str,
    side: str,
    intent_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Find the best matching fee gate event for a given episode.
    
    Join strategy (priority order):
      1. Exact match on intent_id (v7.9-TE1.1)
      2. Fallback: latest event matching symbol+side
    """
    by_intent = fee_gate_events.get("by_intent", {})
    by_symbol_side = fee_gate_events.get("by_symbol_side", {})

    # Priority 1: exact intent_id join
    if intent_id and intent_id in by_intent:
        return by_intent[intent_id]

    # Priority 2: symbol+side fallback
    key = f"{symbol}_{side}"
    return by_symbol_side.get(key)


# ── Trade record construction ─────────────────────────────────────────────

def build_trade_records(
    episodes: List[Dict[str, Any]],
    fee_gate_events: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[TradeRecord]:
    """Convert raw episodes + fee gate events into unified TradeRecords."""
    records: List[TradeRecord] = []

    for ep in episodes:
        try:
            entry_price = float(ep.get("avg_entry_price", 0) or 0)
            exit_price = float(ep.get("avg_exit_price", 0) or 0)
            side = ep.get("side", "LONG")

            # Realized move pct (directional)
            if entry_price > 0:
                if side == "LONG":
                    realized_move_pct = (exit_price - entry_price) / entry_price
                else:
                    realized_move_pct = (entry_price - exit_price) / entry_price
            else:
                realized_move_pct = 0.0

            # Duration
            holding_s = 0.0
            try:
                duration_h = float(ep.get("duration_hours", 0) or 0)
                holding_s = duration_h * 3600.0
            except (TypeError, ValueError):
                holding_s = 0.0

            # Try to get true_edge_v1 fields from fee gate event
            # Join key priority: intent_id (exact) → symbol+side (fallback)
            edge_ev: Optional[Dict[str, Any]] = None
            if fee_gate_events:
                edge_ev = _find_edge_event(
                    fee_gate_events,
                    ep.get("symbol", ""),
                    side,
                    intent_id=ep.get("intent_id", ""),
                )

            pred_edge_pct = 0.0
            pred_edge_usd = 0.0
            atr_pct = 0.0
            k_atr_val = 0.6
            adv = 0.0
            edge_source = "none"
            confidence = float(ep.get("confidence", 0) or 0)

            if edge_ev:
                pred_edge_pct = float(edge_ev.get("true_edge_v1.expected_edge_pct", 0) or 0)
                pred_edge_usd = float(edge_ev.get("true_edge_v1.expected_edge_usd", 0) or 0)
                atr_pct = float(edge_ev.get("true_edge_v1.atr_pct", 0) or 0)
                k_atr_val = float(edge_ev.get("true_edge_v1.k_atr", 0.6) or 0.6)
                adv = float(edge_ev.get("true_edge_v1.adv", 0) or 0)
                edge_source = str(edge_ev.get("edge_source", "none") or "none")
                if confidence == 0.0:
                    confidence = float(edge_ev.get("true_edge_v1.confidence", 0) or 0)
            elif confidence > 0:
                # Derive from episode confidence (legacy proxy)
                adv = max(0.0, min(0.25, confidence - 0.5))
                pred_edge_pct = confidence - 0.5  # legacy proxy
                notional = float(ep.get("entry_notional", 0) or 0)
                pred_edge_usd = notional * abs(pred_edge_pct)
                edge_source = "episode_confidence_proxy"

            records.append(TradeRecord(
                episode_id=ep.get("episode_id", ""),
                symbol=ep.get("symbol", ""),
                strategy=ep.get("strategy", "unknown"),
                side=side,
                regime_at_entry=ep.get("regime_at_entry", ""),
                regime_at_exit=ep.get("regime_at_exit", ""),
                entry_ts=ep.get("entry_ts", ""),
                exit_ts=ep.get("exit_ts", ""),
                holding_time_s=holding_s,
                notional_usd=float(ep.get("entry_notional", 0) or 0),
                fees_usd=float(ep.get("fees", 0) or 0),
                slippage_est_usd=0.0,  # TODO: join with slippage tracker
                pred_expected_edge_pct=pred_edge_pct,
                pred_expected_edge_usd=pred_edge_usd,
                confidence=confidence,
                atr_pct=atr_pct,
                k_atr=k_atr_val,
                adv=adv,
                edge_source=edge_source,
                realized_pnl_usd=float(ep.get("net_pnl", 0) or 0),
                realized_move_pct=realized_move_pct,
                avg_entry_price=entry_price,
                avg_exit_price=exit_price,
            ))
        except Exception as e:
            LOG.warning("Skipping malformed episode: %s", e)
            continue

    return records


# ── Bucketing & statistics ────────────────────────────────────────────────

def _compute_bucket_stats(label: str, trades: List[TradeRecord]) -> BucketStats:
    """Compute aggregated stats for a bucket of trades."""
    if not trades:
        return BucketStats(bucket_label=label)

    pnls = [t.realized_pnl_usd for t in trades]
    pred_edges = [t.pred_expected_edge_pct for t in trades]
    fees = [t.fees_usd for t in trades]
    holdings = [t.holding_time_s for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    pf = gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)

    return BucketStats(
        bucket_label=label,
        count=len(trades),
        mean_pred_edge_pct=round(statistics.mean(pred_edges), 8) if pred_edges else 0.0,
        median_pred_edge_pct=round(statistics.median(pred_edges), 8) if pred_edges else 0.0,
        mean_realized_pnl=round(statistics.mean(pnls), 4) if pnls else 0.0,
        median_realized_pnl=round(statistics.median(pnls), 4) if pnls else 0.0,
        total_realized_pnl=round(sum(pnls), 4),
        win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
        mean_fees=round(statistics.mean(fees), 4) if fees else 0.0,
        profit_factor=round(pf, 4) if math.isfinite(pf) else 999.0,
        mean_holding_s=round(statistics.mean(holdings), 1) if holdings else 0.0,
    )


def _bucket_by_edge_decile(trades: List[TradeRecord]) -> List[BucketStats]:
    """Bucket trades by predicted edge into deciles."""
    if not trades:
        return []

    sorted_trades = sorted(trades, key=lambda t: t.pred_expected_edge_pct)
    n = len(sorted_trades)
    bucket_size = max(1, n // NUM_EDGE_DECILES)
    buckets: List[BucketStats] = []

    for i in range(NUM_EDGE_DECILES):
        start = i * bucket_size
        end = start + bucket_size if i < NUM_EDGE_DECILES - 1 else n
        if start >= n:
            break
        chunk = sorted_trades[start:end]
        lo = chunk[0].pred_expected_edge_pct
        hi = chunk[-1].pred_expected_edge_pct
        label = f"D{i+1} [{lo:.6f}, {hi:.6f}]"
        buckets.append(_compute_bucket_stats(label, chunk))

    return buckets


def _bucket_by_confidence(trades: List[TradeRecord]) -> List[BucketStats]:
    """Bucket trades by confidence in 5pp increments."""
    if not trades:
        return []

    by_bucket: Dict[str, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        # Floor to nearest bucket
        lo = math.floor(t.confidence / CONFIDENCE_BUCKET_WIDTH) * CONFIDENCE_BUCKET_WIDTH
        hi = lo + CONFIDENCE_BUCKET_WIDTH
        label = f"[{lo:.2f}, {hi:.2f})"
        by_bucket[label].append(t)

    buckets: List[BucketStats] = []
    for label in sorted(by_bucket.keys()):
        buckets.append(_compute_bucket_stats(label, by_bucket[label]))
    return buckets


def _compute_k_atr_suggestions(
    trades: List[TradeRecord],
    current_k_atr: float = 0.6,
) -> List[KAtrSuggestion]:
    """Compute k_atr tuning suggestions by regime.
    
    For each regime, compute median(|realized_move_pct| / atr_pct)
    to estimate the actual move multiple.
    """
    # Group by regime
    by_regime: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    all_multiples: List[float] = []

    for t in trades:
        if t.atr_pct > 0 and t.realized_move_pct != 0:
            multiple = abs(t.realized_move_pct) / t.atr_pct
            regime = t.regime_at_entry or "UNKNOWN"
            by_regime[regime].append((multiple, t.atr_pct))
            all_multiples.append(multiple)

    suggestions: List[KAtrSuggestion] = []

    # Overall suggestion
    if all_multiples:
        suggestions.append(KAtrSuggestion(
            regime="ALL",
            sample_count=len(all_multiples),
            current_k_atr=current_k_atr,
            median_move_multiple=round(statistics.median(all_multiples), 4),
            mean_move_multiple=round(statistics.mean(all_multiples), 4),
            suggested_k_atr=round(statistics.median(all_multiples), 4),
        ))

    # Per-regime
    for regime, entries in sorted(by_regime.items()):
        multiples = [m for m, _ in entries]
        if len(multiples) >= 3:  # need at least 3 samples
            suggestions.append(KAtrSuggestion(
                regime=regime,
                sample_count=len(multiples),
                current_k_atr=current_k_atr,
                median_move_multiple=round(statistics.median(multiples), 4),
                mean_move_multiple=round(statistics.mean(multiples), 4),
                suggested_k_atr=round(statistics.median(multiples), 4),
            ))

    return suggestions


def _bucket_by_dimension(
    trades: List[TradeRecord],
    key_fn: Any,
) -> Dict[str, List[BucketStats]]:
    """Bucket trades by an arbitrary key and compute stats per group."""
    by_key: Dict[str, List[TradeRecord]] = defaultdict(list)
    for t in trades:
        k = key_fn(t)
        by_key[k].append(t)

    result: Dict[str, List[BucketStats]] = {}
    for k in sorted(by_key.keys()):
        group_trades = by_key[k]
        result[k] = _bucket_by_edge_decile(group_trades)
    return result


# ── Report generation ─────────────────────────────────────────────────────

def generate_report(
    trades: List[TradeRecord],
    current_k_atr: float = 0.6,
) -> AttributionReport:
    """Generate the full edge attribution report from trade records."""
    now_str = datetime.now(timezone.utc).isoformat()

    if not trades:
        return AttributionReport(
            generated_at=now_str,
            total_trades=0,
            trades_with_atr_edge=0,
            trades_with_fallback=0,
        )

    atr_trades = [t for t in trades if t.edge_source == "atr_conf_v1"]
    fallback_trades = [t for t in trades if t.edge_source != "atr_conf_v1"]

    # Overall stats
    pnls = [t.realized_pnl_usd for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    overall_pf = gross_wins / gross_losses if gross_losses > 0 else (
        float("inf") if gross_wins > 0 else 0.0
    )
    if not math.isfinite(overall_pf):
        overall_pf = 999.0

    return AttributionReport(
        generated_at=now_str,
        total_trades=len(trades),
        trades_with_atr_edge=len(atr_trades),
        trades_with_fallback=len(fallback_trades),
        edge_decile_buckets=_bucket_by_edge_decile(trades),
        confidence_buckets=_bucket_by_confidence(trades),
        k_atr_suggestions=_compute_k_atr_suggestions(trades, current_k_atr),
        strategy_buckets=_bucket_by_dimension(trades, lambda t: t.strategy),
        regime_buckets=_bucket_by_dimension(trades, lambda t: t.regime_at_entry or "UNKNOWN"),
        overall_win_rate=round(len(wins) / len(trades), 4) if trades else 0.0,
        overall_mean_pnl=round(statistics.mean(pnls), 4) if pnls else 0.0,
        overall_profit_factor=round(overall_pf, 4),
    )


# ── Markdown rendering ────────────────────────────────────────────────────

def _render_bucket_table(buckets: List[BucketStats]) -> str:
    """Render a list of BucketStats as a Markdown table."""
    if not buckets:
        return "*No data*\n"

    lines = [
        "| Bucket | Count | Win Rate | Mean PnL | Median PnL | Total PnL | Profit Factor | Mean Fees | Mean Hold (s) |",
        "|--------|-------|----------|----------|------------|-----------|---------------|-----------|---------------|",
    ]
    for b in buckets:
        lines.append(
            f"| {b.bucket_label} | {b.count} | {b.win_rate:.2%} | "
            f"${b.mean_realized_pnl:.2f} | ${b.median_realized_pnl:.2f} | "
            f"${b.total_realized_pnl:.2f} | {b.profit_factor:.2f} | "
            f"${b.mean_fees:.4f} | {b.mean_holding_s:.0f} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown(report: AttributionReport) -> str:
    """Render the attribution report as Markdown."""
    lines: List[str] = []
    lines.append(f"# Edge Attribution Report")
    lines.append(f"Generated: {report.generated_at}\n")

    lines.append(f"## Summary")
    lines.append(f"- Total trades: {report.total_trades}")
    lines.append(f"- Trades with ATR edge: {report.trades_with_atr_edge}")
    lines.append(f"- Trades with fallback proxy: {report.trades_with_fallback}")
    lines.append(f"- Overall win rate: {report.overall_win_rate:.2%}")
    lines.append(f"- Overall mean PnL: ${report.overall_mean_pnl:.2f}")
    lines.append(f"- Overall profit factor: {report.overall_profit_factor:.2f}\n")

    lines.append(f"## A) Calibration by Predicted Edge Decile")
    lines.append(_render_bucket_table(report.edge_decile_buckets))

    lines.append(f"## B) Reliability by Confidence Bucket")
    lines.append(_render_bucket_table(report.confidence_buckets))

    lines.append(f"## C) k_atr Tuning Suggestions")
    if report.k_atr_suggestions:
        lines.append(
            "| Regime | Samples | Current k | Median Multiple | Mean Multiple | Suggested k |"
        )
        lines.append(
            "|--------|---------|-----------|-----------------|---------------|-------------|"
        )
        for s in report.k_atr_suggestions:
            lines.append(
                f"| {s.regime} | {s.sample_count} | {s.current_k_atr:.2f} | "
                f"{s.median_move_multiple:.4f} | {s.mean_move_multiple:.4f} | "
                f"{s.suggested_k_atr:.4f} |"
            )
        lines.append("")
    else:
        lines.append("*No ATR data available for tuning suggestions.*\n")

    lines.append(f"## D) Strategy Breakdown")
    for strat, buckets in sorted(report.strategy_buckets.items()):
        lines.append(f"### {strat}")
        lines.append(_render_bucket_table(buckets))

    lines.append(f"## E) Regime Breakdown")
    for regime, buckets in sorted(report.regime_buckets.items()):
        lines.append(f"### {regime}")
        lines.append(_render_bucket_table(buckets))

    return "\n".join(lines)


# ── CSV rendering ─────────────────────────────────────────────────────────

def render_csv(trades: List[TradeRecord]) -> str:
    """Render trade records as a flat CSV for dashboard ingest."""
    headers = [
        "episode_id", "symbol", "strategy", "side",
        "regime_at_entry", "regime_at_exit",
        "entry_ts", "exit_ts", "holding_time_s",
        "notional_usd", "fees_usd",
        "pred_expected_edge_pct", "pred_expected_edge_usd",
        "confidence", "atr_pct", "k_atr", "adv", "edge_source",
        "realized_pnl_usd", "realized_move_pct",
        "avg_entry_price", "avg_exit_price",
    ]
    lines = [",".join(headers)]
    for t in trades:
        row = [
            t.episode_id, t.symbol, t.strategy, t.side,
            t.regime_at_entry, t.regime_at_exit,
            t.entry_ts, t.exit_ts, f"{t.holding_time_s:.1f}",
            f"{t.notional_usd:.2f}", f"{t.fees_usd:.4f}",
            f"{t.pred_expected_edge_pct:.8f}", f"{t.pred_expected_edge_usd:.4f}",
            f"{t.confidence:.4f}", f"{t.atr_pct:.8f}",
            f"{t.k_atr:.4f}", f"{t.adv:.6f}", t.edge_source,
            f"{t.realized_pnl_usd:.4f}", f"{t.realized_move_pct:.8f}",
            f"{t.avg_entry_price:.8f}", f"{t.avg_exit_price:.8f}",
        ]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


# ── Main ──────────────────────────────────────────────────────────────────

def run(
    episode_path: Optional[Path] = None,
    fee_gate_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    current_k_atr: float = 0.6,
) -> AttributionReport:
    """Run the full attribution report pipeline.

    Args:
        episode_path: Path to episode ledger JSON.
        fee_gate_path: Path to fee gate events JSONL.
        output_dir: Directory for output files.
        current_k_atr: Current k_atr value for tuning comparison.

    Returns:
        The generated AttributionReport.
    """
    ep_path = episode_path or DEFAULT_EPISODE_LEDGER_PATH
    fg_path = fee_gate_path or DEFAULT_FEE_GATE_LOG_PATH
    out_dir = output_dir or DEFAULT_OUTPUT_DIR

    LOG.info("Loading episodes from %s", ep_path)
    episodes = _load_episodes(ep_path)
    LOG.info("Loaded %d episodes", len(episodes))

    LOG.info("Loading fee gate events from %s", fg_path)
    fee_gate_events = _load_fee_gate_events(fg_path)
    LOG.info("Loaded %d fee gate events", len(fee_gate_events))

    trades = build_trade_records(episodes, fee_gate_events)
    LOG.info("Built %d trade records", len(trades))

    report = generate_report(trades, current_k_atr)

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "latest.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    LOG.info("Wrote JSON report: %s", json_path)

    md_path = out_dir / "latest.md"
    with open(md_path, "w") as f:
        f.write(render_markdown(report))
    LOG.info("Wrote Markdown report: %s", md_path)

    csv_path = out_dir / "latest.csv"
    with open(csv_path, "w") as f:
        f.write(render_csv(trades))
    LOG.info("Wrote CSV: %s", csv_path)

    return report


def main() -> None:
    """CLI entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    parser = argparse.ArgumentParser(description="Edge Attribution Report")
    parser.add_argument(
        "--episodes",
        type=Path,
        default=DEFAULT_EPISODE_LEDGER_PATH,
        help="Path to episode ledger JSON",
    )
    parser.add_argument(
        "--fee-gate-log",
        type=Path,
        default=DEFAULT_FEE_GATE_LOG_PATH,
        help="Path to fee gate events JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--k-atr",
        type=float,
        default=0.6,
        help="Current k_atr for tuning comparison",
    )
    args = parser.parse_args()

    report = run(
        episode_path=args.episodes,
        fee_gate_path=args.fee_gate_log,
        output_dir=args.output_dir,
        current_k_atr=args.k_atr,
    )
    print(f"\nDone — {report.total_trades} trades analyzed.")
    print(f"  ATR edge: {report.trades_with_atr_edge}")
    print(f"  Fallback: {report.trades_with_fallback}")
    print(f"  Win rate: {report.overall_win_rate:.2%}")
    print(f"  Mean PnL: ${report.overall_mean_pnl:.2f}")
    print(f"  Profit factor: {report.overall_profit_factor:.2f}")


if __name__ == "__main__":
    main()
