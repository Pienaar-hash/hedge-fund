#!/usr/bin/env python3
"""Smoke test for execution logging utilities."""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from execution.log_utils import get_logger, log_event  # noqa: E402


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh if _.strip())


def _write_sample_logs(base: Path) -> None:
    attempts = get_logger(str(base / "orders_attempted.jsonl"))
    orders = get_logger(str(base / "orders_executed.jsonl"))
    vetoes = get_logger(str(base / "risk_vetoes.jsonl"))
    positions = get_logger(str(base / "position_state.jsonl"))
    heartbeats = get_logger(str(base / "sync_heartbeats.jsonl"))

    ts = time.time()
    common = {
        "strategy": "smoke_test",
        "symbol": "BTCUSDT",
        "signal_ts": ts,
    }

    for idx in range(3):
        log_event(
            attempts,
            "order_attempt",
            {
                **common,
                "attempt_id": f"attempt_{idx}",
                "qty": 0.1 + idx * 0.01,
                "local_ts": ts + idx,
            },
        )

    log_event(
        vetoes,
        "risk_veto",
        {
            **common,
            "veto_reason": "smoke_block",
            "veto_detail": {"reason": "smoke_block", "thresholds": {"test": 1}},
            "local_ts": ts + 0.5,
        },
    )

    for idx in range(2):
        log_event(
            orders,
            "order_executed",
            {
                **common,
                "order_idx": idx,
                "status": "FILLED",
                "price": 30000 + idx * 10,
                "qty": 0.1,
                "client_order_id": f"smoke_{idx}",
                "local_ts": ts + 1 + idx,
            },
        )

    log_event(
        positions,
        "position_snapshot",
        {
            **common,
            "pos_qty": 0.2,
            "entry_px": 30050,
            "unrealized_pnl": 5.0,
            "leverage": 2,
            "mode": "LONG",
            "ts": ts + 1.5,
        },
    )

    log_event(
        heartbeats,
        "heartbeat",
        {
            "service": "smoke_daemon",
            "ts": ts + 2,
            "ok": True,
            "lag_secs": 1.23,
        },
    )


def main() -> int:
    unique_dir = Path("logs/execution") / f"smoke_{uuid.uuid4().hex[:8]}"
    unique_dir.mkdir(parents=True, exist_ok=True)

    _write_sample_logs(unique_dir)

    summary = {}
    for name in (
        "orders_attempted.jsonl",
        "risk_vetoes.jsonl",
        "orders_executed.jsonl",
        "position_state.jsonl",
        "sync_heartbeats.jsonl",
    ):
        summary[name] = _count_lines(unique_dir / name)

    print(json.dumps({"log_dir": str(unique_dir), "counts": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
