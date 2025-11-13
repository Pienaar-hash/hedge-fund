import execution.utils.metrics as metrics
import execution.router_metrics as router_metrics


def _sample_events():
    return [
        {
            "symbol": "BTCUSDC",
            "ts": 1707345600,
            "is_maker_final": True,
            "started_maker": True,
            "used_fallback": False,
            "slippage_bps": -1.0,
        },
        {
            "symbol": "BTCUSDC",
            "ts": 1707349200,
            "is_maker_final": False,
            "started_maker": True,
            "used_fallback": True,
            "slippage_bps": 3.0,
        },
        {
            "symbol": "BTCUSDC",
            "ts": 1707352800,
            "is_maker_final": True,
            "started_maker": False,
            "used_fallback": False,
            "slippage_bps": 0.5,
        },
        {
            "symbol": "BTCUSDC",
            "ts": 1707356400,
            "is_maker_final": False,
            "started_maker": True,
            "used_fallback": True,
            "slippage_bps": 5.0,
        },
    ]


def test_router_effectiveness_7d_ratios_and_quartiles_execution_hardening(monkeypatch):
    monkeypatch.setattr(
        router_metrics,
        "get_recent_router_events",
        lambda symbol=None, window_days=7: _sample_events(),
    )

    eff = metrics.router_effectiveness_7d(symbol="BTCUSDC")

    assert eff["maker_fill_ratio"] == 0.5
    assert eff["fallback_ratio"] == (2 / 3)

    q25 = eff["slip_q25"]
    q50 = eff["slip_q50"]
    q75 = eff["slip_q75"]

    assert q25 is not None and q50 is not None and q75 is not None
    assert q25 <= q50 <= q75
    assert q25 < q50 < q75
