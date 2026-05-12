from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from execution.min_notional_planner import MinNotionalAction, plan_min_notional_action

REPLAY_STRATEGY_ID = "TREND_PULLBACK_V2_REPLAY_CANDIDATE"
LIVE_MODULE_BLOCKLIST = (
    "execution.exchange_utils",
    "execution.order_router",
    "execution.order_dispatch",
    "execution.executor_live",
)


@dataclass(frozen=True)
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    symbol: str
    qty: float
    entry_ts: int
    entry_px: float
    entry_fee: float
    entry_notional: float


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_ts(value: str) -> int:
    v = str(value).strip()
    if not v:
        raise ValueError("empty timestamp")
    if v.isdigit():
        return int(v)
    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
    return int(dt.timestamp())


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in config: {path}")
    return data


def _resolve_data_file(data_dir: Path, symbol: str, timeframe: str) -> Path:
    candidates = [
        data_dir / f"{symbol}_{timeframe}.csv",
        data_dir / f"{symbol}_{timeframe}.parquet",
        data_dir / f"{symbol}.csv",
        data_dir / f"{symbol}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing local data for {symbol} {timeframe}. "
        f"Checked: {[str(p) for p in candidates]}"
    )


def _load_bars(path: Path) -> list[Bar]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            out: list[Bar] = []
            for row in reader:
                out.append(
                    Bar(
                        ts=_parse_ts(str(row.get("timestamp") or row.get("ts") or "")),
                        open=_safe_float(row.get("open")),
                        high=_safe_float(row.get("high")),
                        low=_safe_float(row.get("low")),
                        close=_safe_float(row.get("close")),
                        volume=_safe_float(row.get("volume")),
                    )
                )
        out.sort(key=lambda b: b.ts)
        return out

    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            raise RuntimeError("Parquet input requires pandas/pyarrow") from exc
        frame = pd.read_parquet(path)
        out = [
            Bar(
                ts=_parse_ts(str(row.get("timestamp") or row.get("ts") or "")),
                open=_safe_float(row.get("open")),
                high=_safe_float(row.get("high")),
                low=_safe_float(row.get("low")),
                close=_safe_float(row.get("close")),
                volume=_safe_float(row.get("volume")),
            )
            for row in frame.to_dict(orient="records")
        ]
        out.sort(key=lambda b: b.ts)
        return out

    raise ValueError(f"Unsupported file extension: {path}")


def _strategy_signal(prev_bar: Bar, bar: Bar) -> str:
    # Deterministic placeholder strategy for replay certification only.
    # Entry: up-close with an intrabar pullback under prior close.
    if bar.close > prev_bar.close and bar.low <= (prev_bar.close * 0.999):
        return "ENTER_LONG"
    if bar.close < prev_bar.close:
        return "EXIT_LONG"
    return "HOLD"


def _resolve_per_trade_nav_pct(config: dict[str, Any]) -> float:
    replay_cfg = config.get("replay")
    if isinstance(replay_cfg, dict):
        v = _safe_float(replay_cfg.get("per_trade_nav_pct"), 0.0)
        if v > 0.0:
            return v

    strategies = config.get("strategies")
    if isinstance(strategies, list):
        for item in strategies:
            if not isinstance(item, dict):
                continue
            params = item.get("params")
            if not isinstance(params, dict):
                continue
            v = _safe_float(params.get("per_trade_nav_pct"), 0.0)
            if v > 0.0:
                return v

    return 0.02


def _resolve_min_notional(config: dict[str, Any]) -> float:
    replay_cfg = config.get("replay")
    if isinstance(replay_cfg, dict):
        v = _safe_float(replay_cfg.get("min_notional_usdt"), 0.0)
        if v > 0.0:
            return v

    risk_cfg = config.get("risk")
    if isinstance(risk_cfg, dict):
        v = _safe_float(risk_cfg.get("min_notional_usdt"), 0.0)
        if v > 0.0:
            return v

    return 25.0


def _resolve_max_nav_pct(config: dict[str, Any], per_trade_nav_pct: float) -> float:
    replay_cfg = config.get("replay")
    if isinstance(replay_cfg, dict):
        v = _safe_float(replay_cfg.get("max_trade_nav_pct"), 0.0)
        if v > 0.0:
            return v

    return max(per_trade_nav_pct, 0.0)


def _resolve_git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_replay(
    *,
    data_dir: str,
    symbols: list[str],
    timeframe: str,
    config_path: str,
    starting_nav: float,
    fee_bps: float,
    slippage_bps: float,
    output_base_dir: str = "data/replay_runs",
    run_id: str | None = None,
) -> dict[str, Any]:
    if not symbols:
        raise ValueError("symbols must be non-empty")

    started_at = _now_iso()
    run_id_final = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output_base_dir) / run_id_final
    out_dir.mkdir(parents=True, exist_ok=False)

    data_root = Path(data_dir)
    cfg_path = Path(config_path)
    config = _load_json(cfg_path)

    per_trade_nav_pct = _resolve_per_trade_nav_pct(config)
    min_notional = _resolve_min_notional(config)
    max_trade_nav_pct = _resolve_max_nav_pct(config, per_trade_nav_pct)

    file_map: dict[str, Path] = {}
    bars_by_symbol: dict[str, list[Bar]] = {}
    input_hash_builder = hashlib.sha256()
    for symbol in symbols:
        fp = _resolve_data_file(data_root, symbol, timeframe)
        file_map[symbol] = fp
        bars = _load_bars(fp)
        if len(bars) < 2:
            raise ValueError(f"Need at least 2 bars for {symbol}")
        bars_by_symbol[symbol] = bars
        input_hash_builder.update(symbol.encode("utf-8"))
        input_hash_builder.update(timeframe.encode("utf-8"))
        input_hash_builder.update(_sha256_file(fp).encode("utf-8"))

    fee_rate = max(0.0, fee_bps) / 10_000.0
    slippage_rate = max(0.0, slippage_bps) / 10_000.0

    index_by_symbol: dict[str, dict[int, Bar]] = {
        s: {b.ts: b for b in bars} for s, bars in bars_by_symbol.items()
    }
    timeline = sorted({b.ts for bars in bars_by_symbol.values() for b in bars})

    positions: dict[str, Position | None] = {s: None for s in symbols}
    nav = float(starting_nav)
    total_fees = 0.0

    trades: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    veto_trace: list[dict[str, Any]] = []
    permit_trace: list[dict[str, Any]] = []

    for ts in timeline:
        for symbol in symbols:
            bar = index_by_symbol[symbol].get(ts)
            if bar is None:
                continue

            bars = bars_by_symbol[symbol]
            pos = positions[symbol]

            # Find previous bar by timeline index for this symbol.
            # Since bars are sorted, map lookup + index is deterministic and local.
            idx_lookup = next((i for i, b in enumerate(bars) if b.ts == ts), None)
            if idx_lookup is None or idx_lookup == 0:
                continue
            prev_bar = bars[idx_lookup - 1]

            signal = _strategy_signal(prev_bar, bar)

            if signal == "ENTER_LONG" and pos is None:
                intended_notional = max(0.0, nav * per_trade_nav_pct)
                intended_qty = intended_notional / max(bar.close, 1e-12)
                plan = plan_min_notional_action(
                    symbol=symbol,
                    intended_qty=intended_qty,
                    mark_price=bar.close,
                    intended_notional=intended_notional,
                    min_notional=min_notional,
                    nav_usd=nav,
                    max_nav_pct=max_trade_nav_pct,
                    leverage=1.0,
                    fee_rate=fee_rate,
                )

                if plan.action != MinNotionalAction.PASS:
                    veto_trace.append(
                        {
                            "ts": ts,
                            "symbol": symbol,
                            "reason": "min_notional",
                            "min_notional_action": plan.action.value,
                            "intended_notional": round(plan.intended_notional, 8),
                            "adjusted_notional": round(plan.adjusted_notional, 8),
                        }
                    )
                    permit_trace.append(
                        {
                            "ts": ts,
                            "symbol": symbol,
                            "signal": signal,
                            "permit": False,
                            "reason": plan.action.value,
                        }
                    )
                    continue

                entry_px = bar.close * (1.0 + slippage_rate)
                qty = intended_notional / max(entry_px, 1e-12)
                entry_fee = intended_notional * fee_rate
                nav -= entry_fee
                total_fees += entry_fee
                positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    entry_ts=ts,
                    entry_px=entry_px,
                    entry_fee=entry_fee,
                    entry_notional=intended_notional,
                )
                permit_trace.append(
                    {
                        "ts": ts,
                        "symbol": symbol,
                        "signal": signal,
                        "permit": True,
                        "reason": "allowed",
                    }
                )
                continue

            if signal == "EXIT_LONG" and pos is not None:
                exit_px = bar.close * (1.0 - slippage_rate)
                exit_notional = pos.qty * exit_px
                gross_pnl = (exit_px - pos.entry_px) * pos.qty
                exit_fee = exit_notional * fee_rate
                trade_fees = pos.entry_fee + exit_fee
                net_pnl = gross_pnl - trade_fees

                nav += gross_pnl - exit_fee
                total_fees += exit_fee

                trades.append(
                    {
                        "symbol": symbol,
                        "entry_ts": pos.entry_ts,
                        "exit_ts": ts,
                        "entry_px": round(pos.entry_px, 8),
                        "exit_px": round(exit_px, 8),
                        "qty": round(pos.qty, 8),
                        "gross_pnl": round(gross_pnl, 8),
                        "fees": round(trade_fees, 8),
                        "net_pnl": round(net_pnl, 8),
                        "exit_reason": "signal",
                    }
                )
                permit_trace.append(
                    {
                        "ts": ts,
                        "symbol": symbol,
                        "signal": signal,
                        "permit": True,
                        "reason": "exit",
                    }
                )
                positions[symbol] = None

        unrealized = 0.0
        for symbol in symbols:
            pos = positions[symbol]
            if pos is None:
                continue
            bar = index_by_symbol[symbol].get(ts)
            if bar is None:
                continue
            mark = bar.close
            unrealized += (mark - pos.entry_px) * pos.qty

        equity_curve.append(
            {
                "ts": ts,
                "nav": round(nav + unrealized, 8),
                "cash_nav": round(nav, 8),
                "unrealized": round(unrealized, 8),
            }
        )

    last_ts = timeline[-1]
    for symbol in symbols:
        pos = positions[symbol]
        if pos is None:
            continue
        bar = index_by_symbol[symbol][last_ts]
        exit_px = bar.close * (1.0 - slippage_rate)
        exit_notional = pos.qty * exit_px
        gross_pnl = (exit_px - pos.entry_px) * pos.qty
        exit_fee = exit_notional * fee_rate
        trade_fees = pos.entry_fee + exit_fee
        net_pnl = gross_pnl - trade_fees
        nav += gross_pnl - exit_fee
        total_fees += exit_fee
        trades.append(
            {
                "symbol": symbol,
                "entry_ts": pos.entry_ts,
                "exit_ts": last_ts,
                "entry_px": round(pos.entry_px, 8),
                "exit_px": round(exit_px, 8),
                "qty": round(pos.qty, 8),
                "gross_pnl": round(gross_pnl, 8),
                "fees": round(trade_fees, 8),
                "net_pnl": round(net_pnl, 8),
                "exit_reason": "end_of_replay",
            }
        )
        positions[symbol] = None

    gross_total = sum(_safe_float(t.get("gross_pnl")) for t in trades)
    net_total = sum(_safe_float(t.get("net_pnl")) for t in trades)
    summary = {
        "strategy": REPLAY_STRATEGY_ID,
        "symbols": list(symbols),
        "timeframe": timeframe,
        "starting_nav": float(starting_nav),
        "ending_nav": round(nav, 8),
        "gross_pnl": round(gross_total, 8),
        "fees": round(total_fees, 8),
        "net_pnl": round(net_total, 8),
        "trade_count": len(trades),
        "veto_count": len(veto_trace),
        "permit_count": len(permit_trace),
        "conviction_authority": "frozen",
        "doctrine_mutated": False,
        "live_exchange_calls": False,
    }

    trades_path = out_dir / "trades.csv"
    equity_path = out_dir / "equity_curve.csv"
    veto_path = out_dir / "veto_trace.csv"
    permit_path = out_dir / "permit_trace.csv"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "replay_manifest.json"

    _write_csv(
        trades_path,
        trades,
        [
            "symbol",
            "entry_ts",
            "exit_ts",
            "entry_px",
            "exit_px",
            "qty",
            "gross_pnl",
            "fees",
            "net_pnl",
            "exit_reason",
        ],
    )
    _write_csv(equity_path, equity_curve, ["ts", "nav", "cash_nav", "unrealized"])
    _write_csv(
        veto_path,
        veto_trace,
        ["ts", "symbol", "reason", "min_notional_action", "intended_notional", "adjusted_notional"],
    )
    _write_csv(permit_path, permit_trace, ["ts", "symbol", "signal", "permit", "reason"])
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")

    output_hash_builder = hashlib.sha256()
    for artifact in [trades_path, equity_path, veto_path, permit_path, summary_path]:
        output_hash_builder.update(artifact.name.encode("utf-8"))
        output_hash_builder.update(_sha256_file(artifact).encode("utf-8"))

    manifest = {
        "run_id": run_id_final,
        "git_sha": _resolve_git_sha(),
        "started_at": started_at,
        "completed_at": _now_iso(),
        "symbols": list(symbols),
        "timeframe": timeframe,
        "config_hash": _sha256_file(cfg_path),
        "input_data_hash": input_hash_builder.hexdigest(),
        "output_hash": output_hash_builder.hexdigest(),
        "fee_model": f"fixed_bps:{fee_bps}",
        "slippage_model": f"fixed_close_bps:{slippage_bps}",
        "conviction_authority": "frozen",
        "doctrine_mutated": False,
        "live_exchange_calls": False,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    return {
        "run_dir": str(out_dir),
        "manifest": manifest,
        "summary": summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V8 deterministic local replay engine")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--config", required=True, dest="config_path")
    parser.add_argument("--starting-nav", type=float, required=True)
    parser.add_argument("--fee-bps", type=float, required=True)
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--output-base-dir", default="data/replay_runs")
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_replay(
        data_dir=args.data_dir,
        symbols=args.symbols,
        timeframe=args.timeframe,
        config_path=args.config_path,
        starting_nav=args.starting_nav,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        output_base_dir=args.output_base_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result["manifest"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
