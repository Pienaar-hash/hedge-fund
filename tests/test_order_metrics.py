from __future__ import annotations

import math

import pytest

from execution import order_router


def _approx_equal(a: float | None, b: float, tol: float = 1e-6) -> bool:
    if a is None:
        return False
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_route_intent_metrics_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_route_order(intent, ctx, dry_run):
        return {
            "accepted": True,
            "price": 100.5,
            "qty": 2.0,
            "latency_ms": 25.0,
            "raw": {"status": "FILLED", "commission": "0.12"},
        }

    monkeypatch.setattr(order_router, "route_order", fake_route_order)

    intent = {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": "2",
        "price": 100.0,
        "router_ctx": {"price": 100.0},
        "dry_run": False,
        "timing": {"decision": 150.0},
    }
    resp, metrics = order_router.route_intent(dict(intent), "attempt_1")

    assert resp["accepted"] is True
    assert metrics["attempt_id"] == "attempt_1"
    assert metrics["prices"]["avg_fill"] == pytest.approx(100.5)
    assert metrics["qty"]["contracts"] == pytest.approx(2.0)
    assert metrics["qty"]["notional_usd"] == pytest.approx(200.0)
    expected_slippage = ((100.5 - 100.0) / 100.0) * 10_000
    assert _approx_equal(metrics["slippage_bps"], expected_slippage)
    assert metrics["timing_ms"]["decision"] == pytest.approx(150.0)
    assert metrics["result"]["status"] == "filled"


def test_route_intent_metrics_sell_slippage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_route_order(intent, ctx, dry_run):
        return {
            "accepted": True,
            "price": 99.0,
            "qty": 1.0,
            "latency_ms": 12.0,
            "raw": {"status": "FILLED"},
        }

    monkeypatch.setattr(order_router, "route_order", fake_route_order)

    intent = {
        "symbol": "ETHUSDT",
        "side": "SELL",
        "quantity": "1",
        "price": 100.0,
        "router_ctx": {"mark_price": 100.0},
        "dry_run": False,
        "timing": {"decision": 80.0},
    }
    _, metrics = order_router.route_intent(dict(intent), "attempt_2")

    assert metrics["prices"]["mark"] == pytest.approx(100.0)
    assert metrics["prices"]["avg_fill"] == pytest.approx(99.0)
    # Better fill for SELL should give positive slippage
    assert metrics["slippage_bps"] == pytest.approx(100.0)
    assert metrics["timing_ms"]["decision"] == pytest.approx(80.0)
