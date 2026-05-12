from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from research import backtest_engine_v8 as be


def _write_seed_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_config(path: Path, per_trade_nav_pct: float = 0.2, min_notional_usdt: float = 25.0) -> None:
    data = {
        "replay": {
            "per_trade_nav_pct": per_trade_nav_pct,
            "min_notional_usdt": min_notional_usdt,
            "max_trade_nav_pct": per_trade_nav_pct,
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _basic_rows() -> list[dict[str, str]]:
    return [
        {"timestamp": "1710000000", "open": "100", "high": "101", "low": "99", "close": "100", "volume": "10"},
        {"timestamp": "1710000900", "open": "100", "high": "103", "low": "99.5", "close": "102", "volume": "12"},
        {"timestamp": "1710001800", "open": "102", "high": "103", "low": "99", "close": "100", "volume": "11"},
        {"timestamp": "1710002700", "open": "100", "high": "101", "low": "98", "close": "99", "volume": "13"},
    ]


def test_replay_refuses_missing_local_data(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)

    with pytest.raises(FileNotFoundError):
        be.run_replay(
            data_dir=str(tmp_path / "missing"),
            symbols=["BTCUSDT"],
            timeframe="15m",
            config_path=str(cfg),
            starting_nav=10_000,
            fee_bps=5,
            slippage_bps=3,
            output_base_dir=str(tmp_path / "out"),
            run_id="r1",
        )


def test_replay_emits_required_artifacts_and_manifest_hashes(tmp_path: Path) -> None:
    data_dir = tmp_path / "seed"
    out_dir = tmp_path / "replay_runs"
    cfg = tmp_path / "cfg.json"
    _write_config(cfg)

    _write_seed_csv(data_dir / "BTCUSDT_15m.csv", _basic_rows())
    _write_seed_csv(data_dir / "ETHUSDT_15m.csv", _basic_rows())

    result = be.run_replay(
        data_dir=str(data_dir),
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="15m",
        config_path=str(cfg),
        starting_nav=10_000,
        fee_bps=5,
        slippage_bps=3,
        output_base_dir=str(out_dir),
        run_id="run_a",
    )

    run_path = Path(result["run_dir"])
    required = [
        "replay_manifest.json",
        "trades.csv",
        "equity_curve.csv",
        "veto_trace.csv",
        "permit_trace.csv",
        "summary.json",
    ]
    for name in required:
        assert (run_path / name).exists(), name

    manifest = json.loads((run_path / "replay_manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_hash"]
    assert manifest["input_data_hash"]
    assert manifest["output_hash"]
    assert manifest["conviction_authority"] == "frozen"
    assert manifest["doctrine_mutated"] is False
    assert manifest["live_exchange_calls"] is False


def test_output_hash_is_stable_across_identical_runs(tmp_path: Path) -> None:
    data_dir = tmp_path / "seed"
    cfg = tmp_path / "cfg.json"
    out_dir = tmp_path / "runs"
    _write_config(cfg)
    _write_seed_csv(data_dir / "SOLUSDT_15m.csv", _basic_rows())

    r1 = be.run_replay(
        data_dir=str(data_dir),
        symbols=["SOLUSDT"],
        timeframe="15m",
        config_path=str(cfg),
        starting_nav=10_000,
        fee_bps=5,
        slippage_bps=3,
        output_base_dir=str(out_dir),
        run_id="stable_1",
    )
    r2 = be.run_replay(
        data_dir=str(data_dir),
        symbols=["SOLUSDT"],
        timeframe="15m",
        config_path=str(cfg),
        starting_nav=10_000,
        fee_bps=5,
        slippage_bps=3,
        output_base_dir=str(out_dir),
        run_id="stable_2",
    )

    assert r1["manifest"]["output_hash"] == r2["manifest"]["output_hash"]


def test_no_live_exchange_module_is_imported(tmp_path: Path) -> None:
    data_dir = tmp_path / "seed"
    cfg = tmp_path / "cfg.json"
    out_dir = tmp_path / "runs"
    _write_config(cfg)
    _write_seed_csv(data_dir / "BTCUSDT_15m.csv", _basic_rows())

    for mod in be.LIVE_MODULE_BLOCKLIST:
        sys.modules.pop(mod, None)

    be.run_replay(
        data_dir=str(data_dir),
        symbols=["BTCUSDT"],
        timeframe="15m",
        config_path=str(cfg),
        starting_nav=10_000,
        fee_bps=5,
        slippage_bps=3,
        output_base_dir=str(out_dir),
        run_id="blocklist_import_test",
    )

    for mod in be.LIVE_MODULE_BLOCKLIST:
        assert mod not in sys.modules


def test_fees_reduce_gross_to_net_pnl(tmp_path: Path) -> None:
    data_dir = tmp_path / "seed"
    cfg = tmp_path / "cfg.json"
    out_dir = tmp_path / "runs"
    _write_config(cfg)
    _write_seed_csv(data_dir / "BTCUSDT_15m.csv", _basic_rows())

    result = be.run_replay(
        data_dir=str(data_dir),
        symbols=["BTCUSDT"],
        timeframe="15m",
        config_path=str(cfg),
        starting_nav=10_000,
        fee_bps=50,
        slippage_bps=3,
        output_base_dir=str(out_dir),
        run_id="fees_test",
    )

    trades_path = Path(result["run_dir"]) / "trades.csv"
    with trades_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows, "Expected at least one trade"
    gross = float(rows[0]["gross_pnl"])
    net = float(rows[0]["net_pnl"])
    assert net < gross


def test_min_notional_behavior_is_veto_traced(tmp_path: Path) -> None:
    data_dir = tmp_path / "seed"
    cfg = tmp_path / "cfg.json"
    out_dir = tmp_path / "runs"
    _write_config(cfg, per_trade_nav_pct=0.001, min_notional_usdt=25.0)
    _write_seed_csv(data_dir / "BTCUSDT_15m.csv", _basic_rows())

    result = be.run_replay(
        data_dir=str(data_dir),
        symbols=["BTCUSDT"],
        timeframe="15m",
        config_path=str(cfg),
        starting_nav=1000,
        fee_bps=5,
        slippage_bps=3,
        output_base_dir=str(out_dir),
        run_id="min_notional_veto",
    )

    veto_path = Path(result["run_dir"]) / "veto_trace.csv"
    with veto_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows, "Expected min-notional veto rows"
    assert rows[0]["reason"] == "min_notional"
    assert rows[0]["min_notional_action"] != "PASS"
