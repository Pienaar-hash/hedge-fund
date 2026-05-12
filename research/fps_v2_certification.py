from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.backtest_engine_v8 import REPLAY_STRATEGY_ID, run_replay


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compute_max_drawdown_pct(equity_rows: list[dict[str, str]]) -> float:
    peak = 0.0
    max_dd = 0.0
    for row in equity_rows:
        nav = _safe_float(row.get("nav"), 0.0)
        if nav > peak:
            peak = nav
        if peak > 0.0:
            dd = (peak - nav) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100.0


def _collect_trade_metrics(trades_rows: list[dict[str, str]]) -> dict[str, float | int]:
    gross_pnl = 0.0
    fees = 0.0
    net_pnl = 0.0
    wins = 0
    gross_profit = 0.0

    for row in trades_rows:
        gross = _safe_float(row.get("gross_pnl"), 0.0)
        fee = _safe_float(row.get("fees"), 0.0)
        net = _safe_float(row.get("net_pnl"), 0.0)
        gross_pnl += gross
        fees += fee
        net_pnl += net
        if net > 0.0:
            wins += 1
        if gross > 0.0:
            gross_profit += gross

    n = len(trades_rows)
    win_rate = (wins / n * 100.0) if n > 0 else 0.0
    return {
        "sample_size": n,
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl": net_pnl,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
    }


def _build_verdict(*, sample_size: int, net_pnl: float, fees: float, gross_profit: float, max_dd_pct: float, output_hash_stable: bool) -> str:
    if sample_size < 30:
        return "INSUFFICIENT_SAMPLE"

    fees_rule = gross_profit > 0.0 and fees < (0.4 * gross_profit)
    checks = [
        net_pnl > 0.0,
        fees_rule,
        max_dd_pct <= 10.0,
        output_hash_stable,
    ]
    return "PASS" if all(checks) else "FAIL"


def certify_replay(
    *,
    data_dir: str,
    symbols: list[str],
    timeframe: str,
    config_path: str,
    starting_nav: float,
    fee_bps: float,
    slippage_bps: float,
    run_id: str,
    output_base_dir: str = "data/replay_certifications",
) -> dict[str, Any]:
    cert_dir = Path(output_base_dir) / run_id
    cert_dir.mkdir(parents=True, exist_ok=False)

    replay_base = cert_dir / "replay_runs"

    run_a = run_replay(
        data_dir=data_dir,
        symbols=symbols,
        timeframe=timeframe,
        config_path=config_path,
        starting_nav=starting_nav,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        output_base_dir=str(replay_base),
        run_id=f"{run_id}_a",
    )
    run_b = run_replay(
        data_dir=data_dir,
        symbols=symbols,
        timeframe=timeframe,
        config_path=config_path,
        starting_nav=starting_nav,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        output_base_dir=str(replay_base),
        run_id=f"{run_id}_b",
    )

    run_a_dir = Path(str(run_a["run_dir"]))
    trades_rows = _read_csv_rows(run_a_dir / "trades.csv")
    equity_rows = _read_csv_rows(run_a_dir / "equity_curve.csv")
    veto_rows = _read_csv_rows(run_a_dir / "veto_trace.csv")

    trade_metrics = _collect_trade_metrics(trades_rows)
    max_dd_pct = _compute_max_drawdown_pct(equity_rows)

    out_hash_a = str(run_a["manifest"].get("output_hash") or "")
    out_hash_b = str(run_b["manifest"].get("output_hash") or "")
    output_hash_stable = bool(out_hash_a and out_hash_a == out_hash_b)

    verdict = _build_verdict(
        sample_size=int(trade_metrics["sample_size"]),
        net_pnl=float(trade_metrics["net_pnl"]),
        fees=float(trade_metrics["fees"]),
        gross_profit=float(trade_metrics["gross_profit"]),
        max_dd_pct=max_dd_pct,
        output_hash_stable=output_hash_stable,
    )

    report = {
        "setup_class": REPLAY_STRATEGY_ID,
        "symbols": list(symbols),
        "sample_size": int(trade_metrics["sample_size"]),
        "gross_pnl": round(float(trade_metrics["gross_pnl"]), 8),
        "fees": round(float(trade_metrics["fees"]), 8),
        "net_pnl": round(float(trade_metrics["net_pnl"]), 8),
        "win_rate": round(float(trade_metrics["win_rate"]), 8),
        "max_drawdown": round(max_dd_pct, 8),
        "veto_count": len(veto_rows),
        "output_hash": out_hash_a,
        "verdict": verdict,
        "output_hash_stable": output_hash_stable,
        "conviction_authority": "frozen",
        "doctrine_mutated": False,
        "live_exchange_calls": False,
        "started_at": _now_iso(),
        "completed_at": _now_iso(),
    }

    report_path = cert_dir / "certification_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "certification_dir": str(cert_dir),
        "report": report,
        "report_path": str(report_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FPS v2 replay certification")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--config", required=True, dest="config_path")
    parser.add_argument("--starting-nav", type=float, required=True)
    parser.add_argument("--fee-bps", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-base-dir", default="data/replay_certifications")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = certify_replay(
        data_dir=args.data_dir,
        symbols=args.symbols,
        timeframe=args.timeframe,
        config_path=args.config_path,
        starting_nav=args.starting_nav,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        run_id=args.run_id,
        output_base_dir=args.output_base_dir,
    )
    print(json.dumps(result["report"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
