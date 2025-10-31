from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st

from execution.log_utils import percentile

SIGNAL_METRICS_PATH = Path("logs/execution/signal_metrics.jsonl")
ORDER_METRICS_PATH = Path("logs/execution/order_metrics.jsonl")
ATTEMPTS_PATH = Path("logs/execution/orders_attempted.jsonl")
DEFAULT_WINDOW = 200


def _tail_jsonl(path: Path, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    rows: deque[Dict[str, Any]] = deque(maxlen=limit)
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
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    return list(rows)


def _to_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            seconds = float(value)
        except Exception:
            return None
        if seconds > 1e12:
            seconds = seconds / 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            cleaned = txt.replace("Z", "+00:00") if txt.endswith("Z") else txt
            parsed = datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _orders_dataframe(order_rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in order_rows:
        ts = _to_ts(row.get("ts"))
        prices = row.get("prices") or {}
        qty = row.get("qty") or {}
        timing = row.get("timing_ms") or {}
        result = row.get("result") or {}
        records.append(
            {
                "time": ts,
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "status": result.get("status"),
                "slippage_bps": _to_float(row.get("slippage_bps")),
                "decision_ms": _to_float(timing.get("decision")),
                "ack_ms": _to_float(timing.get("ack")),
                "fill_ms": _to_float(timing.get("fill")),
                "avg_fill": _to_float(prices.get("avg_fill")),
                "mark_px": _to_float(prices.get("mark")),
                "submitted_px": _to_float(prices.get("submitted")),
                "contracts": _to_float(qty.get("contracts")),
                "notional_usd": _to_float(qty.get("notional_usd")),
                "fees_usd": _to_float(row.get("fees_usd")),
                "attempt_id": row.get("attempt_id"),
                "intent_id": row.get("intent_id"),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df.sort_values("time", ascending=False, inplace=True, ignore_index=True)
    return df


def _slippage_series(orders_df: pd.DataFrame) -> pd.DataFrame:
    if orders_df.empty or "time" not in orders_df.columns:
        return pd.DataFrame(columns=["minute", "slippage_p50"])
    df = orders_df[["time", "slippage_bps"]].dropna()
    if df.empty:
        return pd.DataFrame(columns=["minute", "slippage_p50"])
    df = df.copy()
    df["minute"] = df["time"].dt.floor("1min")
    grouped = df.groupby("minute", as_index=False)["slippage_bps"].median()
    grouped.rename(columns={"slippage_bps": "slippage_p50"}, inplace=True)
    return grouped.sort_values("minute", ascending=True, ignore_index=True)


@st.cache_data(ttl=10, show_spinner=False)
def load_router_health(window: int = DEFAULT_WINDOW) -> Dict[str, Any]:
    signal_rows = _tail_jsonl(SIGNAL_METRICS_PATH, window)
    order_rows = _tail_jsonl(ORDER_METRICS_PATH, window)
    attempt_rows = _tail_jsonl(ATTEMPTS_PATH, window)

    attempted = len(signal_rows)
    emitted = sum(1 for row in signal_rows if isinstance(row.get("doctor"), dict) and row["doctor"].get("ok"))
    vetoed = max(0, attempted - emitted)
    veto_rate = (vetoed / attempted * 100.0) if attempted else 0.0
    emit_rate = (emitted / attempted * 100.0) if attempted else 0.0

    slip_values = [
        val for val in (_to_float(row.get("slippage_bps")) for row in order_rows) if val is not None
    ]
    decision_values = [
        val
        for val in (
            _to_float((row.get("timing_ms") or {}).get("decision"))
            for row in order_rows
        )
        if val is not None
    ]
    attempt_decision_values = [
        val
        for val in (_to_float(row.get("decision_latency_ms")) for row in attempt_rows)
        if val is not None
    ]
    if not decision_values and attempt_decision_values:
        decision_values = attempt_decision_values

    summary = {
        "attempted": attempted,
        "emitted": emitted,
        "vetoed": vetoed,
        "veto_rate": veto_rate,
        "emit_rate": emit_rate,
        "slippage_p50": percentile(slip_values, 50.0) if slip_values else 0.0,
        "slippage_p95": percentile(slip_values, 95.0) if slip_values else 0.0,
        "decision_p50_ms": percentile(decision_values, 50.0) if decision_values else 0.0,
        "decision_p95_ms": percentile(decision_values, 95.0) if decision_values else 0.0,
    }

    orders_df = _orders_dataframe(order_rows)
    slip_series = _slippage_series(orders_df)

    return {
        "summary": summary,
        "orders_df": orders_df,
        "slip_series": slip_series,
        "window": window,
        "signal_rows": signal_rows,
        "order_rows": order_rows,
        "attempt_rows": attempt_rows,
    }


def render_router_health_tab(window: int = DEFAULT_WINDOW) -> None:
    data = load_router_health(window=window)
    summary = data["summary"]
    orders_df: pd.DataFrame = data["orders_df"]
    slip_series: pd.DataFrame = data["slip_series"]

    st.caption(f"Window size: last {data['window']} entries · auto-refresh every 10s")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Attempted", summary["attempted"])
    metric_cols[1].metric("Emitted", summary["emitted"], f"{summary['emit_rate']:.1f}%")
    metric_cols[2].metric("Veto %", f"{summary['veto_rate']:.1f}%")
    metric_cols[3].metric("Slippage p50", f"{summary['slippage_p50']:.2f} bps")
    metric_cols[4].metric("Decision p50", f"{summary['decision_p50_ms']:.0f} ms")

    detail_cols = st.columns(3)
    detail_cols[0].metric("Slippage p95", f"{summary['slippage_p95']:.2f} bps")
    detail_cols[1].metric("Decision p95", f"{summary['decision_p95_ms']:.0f} ms")
    detail_cols[2].metric(
        "Veto Count",
        summary["vetoed"],
        None,
    )

    st.markdown("### Slippage Trend (p50)")
    if slip_series.empty:
        st.info("No order metrics recorded yet.")
    else:
        chart_df = slip_series.set_index("minute")
        st.line_chart(chart_df["slippage_p50"], use_container_width=True)

    st.markdown("### Recent Orders")
    if orders_df.empty:
        st.info("No router orders logged in the selected window.")
    else:
        table_df = orders_df.copy()
        if "time" in table_df.columns:
            table_df["time"] = pd.to_datetime(table_df["time"], utc=True, errors="coerce")
            table_df["time"] = table_df["time"].dt.tz_convert(timezone.utc).dt.strftime("%Y-%m-%d %H:%M:%S")
        display_cols = [
            "time",
            "symbol",
            "side",
            "status",
            "slippage_bps",
            "decision_ms",
            "ack_ms",
            "avg_fill",
            "contracts",
            "fees_usd",
        ]
        existing_cols = [col for col in display_cols if col in table_df.columns]
        table_display = table_df[existing_cols].head(100)
        st.dataframe(table_display, use_container_width=True, height=360)
