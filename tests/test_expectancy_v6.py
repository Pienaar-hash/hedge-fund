import json
from datetime import datetime, timezone

from execution.intel import expectancy_v6 as exp


def _ts(hour: int) -> float:
    return datetime(2024, 1, 1, hour, 0, tzinfo=timezone.utc).timestamp()


def test_build_expectancy_symbol_hour_regimes():
    trades = [
        {"symbol": "BTCUSDT", "pnl_usd": 10.0, "ts": _ts(1), "router_metric": {"is_maker_final": True}},
        {"symbol": "BTCUSDT", "pnl_usd": -5.0, "ts": _ts(1), "router_metric": {"used_fallback": True}},
        {"symbol": "ETHUSDT", "pnl_usd": 3.0, "ts": _ts(5), "router_metric": {"slippage_bps": 6.0}},
    ]
    snapshot = exp.build_expectancy({"trades": trades, "lookback_hours": 24})
    assert snapshot["symbols"]["BTCUSDT"]["count"] == 2
    assert snapshot["hours"]["1"]["count"] == 2
    assert snapshot["regimes"]["maker_success"]["count"] == 1
    assert snapshot["regimes"]["fallback"]["count"] == 1
    assert snapshot["regimes"]["slip_high"]["count"] == 1


def test_merge_trades_with_policy():
    trades = [{"attempt_id": "a", "symbol": "BTCUSDT", "pnl_usd": 1.0}]
    metrics = {"a": {"policy": {"quality": "good"}}}
    merged = exp.merge_trades_with_policy(trades, metrics)
    assert merged[0]["router_policy"]["quality"] == "good"


def test_save_and_load_expectancy(tmp_path):
    snapshot = {"symbols": {"BTC": {"expectancy": 1.0}}, "updated_ts": 1.0}
    target = tmp_path / "expectancy.json"
    exp.save_expectancy(target, snapshot)
    loaded = exp.load_expectancy(target)
    assert loaded == snapshot


def test_load_inputs_merges_router(tmp_path):
    exec_dir = tmp_path / "execution"
    state_dir = tmp_path / "state"
    exec_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    fills_path = exec_dir / "orders_executed.jsonl"
    router_path = exec_dir / "order_metrics.jsonl"
    nav_path = state_dir / "nav.json"
    fills_path.write_text(
        json.dumps(
            {
                "event_type": "order_close",
                "symbol": "BTCUSDT",
                "realizedPnlUsd": 2.0,
                "ts_close": datetime.now(timezone.utc).isoformat(),
                "attempt_id": "abc",
            }
        )
        + "\n"
    )
    router_path.write_text(
        json.dumps(
            {
                "attempt_id": "abc",
                "ts": datetime.now(timezone.utc).isoformat(),
                "is_maker_final": True,
                "policy": {"quality": "good"},
            }
        )
        + "\n"
    )
    nav_path.write_text(json.dumps({"nav": 1000.0}))

    inputs = exp.load_inputs(tmp_path, lookback_days=1)
    trades = inputs["trades"]
    assert trades and trades[0]["router_metric"]["is_maker_final"] is True
