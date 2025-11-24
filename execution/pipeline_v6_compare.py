"""Comparison engine for v5 live vs v6 shadow pipeline."""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from execution.log_utils import get_logger
from execution.pipeline_v6_shadow import PIPELINE_SHADOW_LOG
from execution.state_publish import write_pipeline_v6_compare_summary

COMPARE_LOG_PATH = Path("logs/pipeline_v6_compare.jsonl")
COMPARE_STATE_PATH = Path("logs/state/pipeline_v6_compare_summary.json")
MIN_SAMPLE_SIZE_FOR_STEADY_STATE = 200
LOG = logging.getLogger("pipeline_compare")


def _stat_block(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    values_sorted = sorted(values)
    mean = sum(values_sorted) / len(values_sorted)
    p50 = statistics.median(values_sorted)
    idx_95 = int(len(values_sorted) * 0.95) - 1
    idx_95 = max(0, min(len(values_sorted) - 1, idx_95))
    return {"mean": mean, "p50": p50, "p95": values_sorted[idx_95]}


def _load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    except Exception as exc:
        LOG.debug("[pipeline_compare] jsonl_load_failed path=%s err=%s", path, exc)
        return []
    if isinstance(limit, int) and limit > 0:
        return rows[-limit:]
    return rows


def _align_events(shadow: List[Mapping[str, Any]], orders: List[Mapping[str, Any]]) -> List[Tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]]:
    aligned: List[Tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]] = []
    live_by_symbol: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for entry in orders:
        symbol = str(entry.get("symbol") or "").upper()
        if symbol:
            live_by_symbol[symbol].append(entry)
    for shadow_entry in shadow:
        sym = str(shadow_entry.get("symbol") or "").upper()
        matches = live_by_symbol.get(sym, [])
        candidate: Optional[Mapping[str, Any]] = matches[-1] if matches else None
        aligned.append((shadow_entry, candidate))
    return aligned


def _extract_shadow_sized(shadow_entry: Mapping[str, Any]) -> Optional[float]:
    sizing = shadow_entry.get("sizing") if isinstance(shadow_entry, Mapping) else {}
    if isinstance(sizing, Mapping):
        try:
            sized = float(sizing.get("sized_gross_usd") or 0.0)
            if sized > 0.0:
                return sized
        except Exception:
            pass
    size_decision = shadow_entry.get("size_decision") if isinstance(shadow_entry, Mapping) else {}
    if isinstance(size_decision, Mapping):
        try:
            sized = float(size_decision.get("gross_usd") or 0.0)
            if sized > 0.0:
                return sized
        except Exception:
            pass
    intent = shadow_entry.get("intent") if isinstance(shadow_entry, Mapping) else {}
    if isinstance(intent, Mapping):
        try:
            notional = float(intent.get("quote_notional") or intent.get("notional") or 0.0)
            if notional > 0.0:
                return notional
        except Exception:
            pass
    return None


def _extract_live_gross(entry: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not isinstance(entry, Mapping):
        return None
    candidates = [
        entry.get("notional"),
        entry.get("gross"),
        entry.get("quote_notional"),
        entry.get("notional_usd"),
        entry.get("requested_notional"),
        entry.get("desired_gross_usd"),
        entry.get("requested_gross"),
    ]
    for candidate in candidates:
        try:
            val = float(candidate)
            if val > 0.0:
                return val
        except Exception:
            continue
    normalized = entry.get("normalized") if isinstance(entry, Mapping) else {}
    if isinstance(normalized, Mapping):
        try:
            val = float(normalized.get("notional") or normalized.get("gross") or 0.0)
            if val > 0.0:
                return val
        except Exception:
            pass
    return None


def _summarize_diffs(
    diffs: List[Mapping[str, Any]],
    sizing_deltas: List[float],
    upsize_count: int,
    sizing_samples: int,
) -> Dict[str, Any]:
    veto_mismatch = sum(1 for diff in diffs if diff.get("veto_mismatch"))
    size_deltas = [abs(diff.get("size_pct_diff", 0.0)) for diff in diffs if diff.get("size_pct_diff") is not None]
    slip_deltas = [diff.get("slippage_diff_bps", 0.0) for diff in diffs if diff.get("slippage_diff_bps") is not None]
    summary = {
        "generated_ts": time.time(),
        "sample_size": len(diffs),
        "veto_mismatch_pct": (veto_mismatch / max(1, len(diffs))) * 100.0,
        "size_diff_stats": _stat_block(size_deltas),
        "slippage_diff_bps": _stat_block(slip_deltas),
        "sizing_diff_stats": {
            "p50": _stat_block(sizing_deltas)["p50"],
            "p95": _stat_block(sizing_deltas)["p95"],
            "upsize_count": upsize_count,
            "sample_size": sizing_samples,
        },
    }
    if summary["sample_size"] < MIN_SAMPLE_SIZE_FOR_STEADY_STATE:
        summary["is_warmup"] = True
        summary["warmup_reason"] = "sample_size_below_min"
    else:
        summary["is_warmup"] = False
        summary["warmup_reason"] = None
    summary["min_sample_size"] = MIN_SAMPLE_SIZE_FOR_STEADY_STATE
    return summary


def compare_pipeline_v6(
    *,
    shadow_limit: Optional[int] = 500,
    orders_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> Dict[str, Any]:
    shadow_entries = _load_jsonl(PIPELINE_SHADOW_LOG, limit=shadow_limit)
    orders = _load_jsonl(Path(orders_path or "logs/execution/orders_executed.jsonl"))
    metrics = _load_jsonl(Path(metrics_path or "logs/execution/order_metrics.jsonl"))
    aligned = _align_events(shadow_entries, orders)
    diffs: List[Dict[str, Any]] = []
    sizing_deltas: List[float] = []
    upsize_count = 0
    sizing_samples = 0
    for shadow_entry, live_entry in aligned:
        shadow_risk = shadow_entry.get("risk_decision") or {}
        live_risk = live_entry or {}
        shadow_sized = _extract_shadow_sized(shadow_entry)
        live_gross = _extract_live_gross(live_entry)
        sizing_diff = None
        if shadow_sized and live_gross:
            sizing_diff = abs(live_gross - shadow_sized) / max(shadow_sized, 1e-9)
            sizing_deltas.append(sizing_diff)
            sizing_samples += 1
            if live_gross > shadow_sized:
                upsize_count += 1
        diff = {
            "symbol": shadow_entry.get("symbol"),
            "ts": shadow_entry.get("timestamp"),
            "shadow": shadow_entry,
            "live": live_entry,
            "veto_mismatch": bool(shadow_risk.get("allowed") != (live_entry is not None)),
            "size_pct_diff": None,
            "slippage_diff_bps": None,
            "sizing_diff_pct": sizing_diff,
        }
        diffs.append(diff)
    summary = _summarize_diffs(diffs, sizing_deltas, upsize_count, sizing_samples)
    try:
        logger = get_logger(str(COMPARE_LOG_PATH))
        wrote_diff = False
        for diff in diffs:
            logger.write(diff)
            wrote_diff = True
        heartbeat = {
            "ts": summary.get("generated_ts"),
            "diff_count": len(diffs),
            "heartbeat": not wrote_diff,
            "summary": summary,
        }
        logger.write(heartbeat)
    except Exception:
        pass
    write_pipeline_v6_compare_summary(summary)
    return summary


__all__ = ["compare_pipeline_v6"]
