"""Comparison engine for v5 live vs v6 shadow pipeline.

Contract (v6.1 compare):
* Expected shadow input: logs/pipeline_v6_shadow.jsonl entries with fields
  {symbol: str (uppercase), risk_decision.allowed: bool, timestamp: float seconds,
   optional size_decision/router_decision}.
* Expected live inputs: logs/execution/orders_executed.jsonl (order ack/fill records)
  and optional logs/execution/order_metrics.jsonl. Records must include symbol and
  timestamp convertible to float seconds.
* Outputs: diffs → logs/pipeline_v6_compare.jsonl, summary → logs/state/
  pipeline_v6_compare_summary.json (sample_size, warmup flag, mismatch metrics).
"""

from __future__ import annotations

import json
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
    except Exception:
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


def _coerce_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _summarize_diffs(diffs: List[Mapping[str, Any]]) -> Dict[str, Any]:
    veto_mismatch = sum(1 for diff in diffs if diff.get("veto_mismatch"))
    missing_live = sum(1 for diff in diffs if diff.get("live") is None)
    size_deltas = [abs(diff.get("size_pct_diff", 0.0)) for diff in diffs if diff.get("size_pct_diff") is not None]
    slip_deltas = [diff.get("slippage_diff_bps", 0.0) for diff in diffs if diff.get("slippage_diff_bps") is not None]
    def _stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
        values_sorted = sorted(values)
        mean = sum(values_sorted) / len(values_sorted)
        p50 = statistics.median(values_sorted)
        idx_95 = int(len(values_sorted) * 0.95) - 1
        idx_95 = max(0, min(len(values_sorted) - 1, idx_95))
        return {"mean": mean, "p50": p50, "p95": values_sorted[idx_95]}
    summary = {
        "generated_ts": time.time(),
        "sample_size": len(diffs),
        "veto_mismatch_pct": (veto_mismatch / max(1, len(diffs))) * 100.0,
        "missing_live_count": missing_live,
        "missing_live_pct": (missing_live / max(1, len(diffs))) * 100.0,
        "size_diff_stats": _stats(size_deltas),
        "slippage_diff_bps": _stats(slip_deltas),
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
    logger = get_logger(str(COMPARE_LOG_PATH))
    logger.write(
        {
            "ts": time.time(),
            "heartbeat": True,
            "shadow_sample": len(shadow_entries),
            "orders_sample": len(orders),
            "metrics_sample": len(metrics),
            "note": "pipeline_v6_compare_start",
        }
    )
    aligned = _align_events(shadow_entries, orders)
    diffs: List[Dict[str, Any]] = []
    for shadow_entry, live_entry in aligned:
        shadow_risk = shadow_entry.get("risk_decision") or {}
        live_risk = live_entry or {}
        diff = {
            "symbol": shadow_entry.get("symbol"),
            "ts": _coerce_ts(shadow_entry.get("timestamp")),
            "shadow": shadow_entry,
            "live": live_entry,
            "veto_mismatch": bool(shadow_risk.get("allowed") != (live_entry is not None)),
            "size_pct_diff": None,
            "slippage_diff_bps": None,
        }
        if diff["ts"] is None:
            diff["parse_error"] = "missing_timestamp"
        diffs.append(diff)
    summary = _summarize_diffs(diffs)
    try:
        wrote_diff = False
        for diff in diffs:
            logger.write(diff)
            wrote_diff = True
        heartbeat = {
            "ts": summary.get("generated_ts"),
            "diff_count": len(diffs),
            "heartbeat": not wrote_diff,
            "summary": summary,
            "shadow_sample": len(shadow_entries),
            "orders_sample": len(orders),
            "metrics_sample": len(metrics),
        }
        logger.write(heartbeat)
    except Exception:
        pass
    write_pipeline_v6_compare_summary(summary)
    return summary


__all__ = ["compare_pipeline_v6"]
