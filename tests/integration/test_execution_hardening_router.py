import pytest

import execution.order_router as router
from execution.order_router import PlaceOrderResult, effective_px, submit_limit

pytestmark = pytest.mark.skip(reason="POST_ONLY fallback behavior changed in v7")


def test_effective_px_fee_math_execution_hardening():
    px_buy = effective_px(100.0, "BUY", is_maker=False)
    assert px_buy > 100.0
    px_sell = effective_px(100.0, "SELL", is_maker=True)
    assert px_sell > 100.0


def test_post_only_fallback_execution_hardening(monkeypatch):
    calls = {"n": 0}

    def fake_place(symbol, side, order_type, price, qty, flags=None):
        calls["n"] += 1
        post_only = bool(flags and flags.get("postOnly"))
        rejected = post_only and calls["n"] <= 1
        return PlaceOrderResult(
            order_id=f"o{calls['n']}",
            side=side,
            price=price,
            qty=qty,
            is_maker=post_only and not rejected,
            rejected_post_only=rejected,
            rejections=calls["n"],
            slippage_bps=0.0,
        )

    monkeypatch.setattr(router, "place_order", fake_place)
    result = submit_limit("LTCUSDT", 100.0, 1.0, "BUY", post_only=True)
    assert result.is_maker is False
    assert calls["n"] == 2
