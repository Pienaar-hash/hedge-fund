import pytest

from execution.pnl_tracker import Fill as PnlFill
from execution.pnl_tracker import PositionTracker


def test_position_tracker_fifo_long_close() -> None:
    tracker = PositionTracker()
    tracker.apply_fill(PnlFill(symbol="BTCUSDT", side="BUY", qty=2.0, price=100.0, position_side="LONG"))
    result = tracker.apply_fill(
        PnlFill(
            symbol="BTCUSDT",
            side="SELL",
            qty=1.0,
            price=110.0,
            fee=0.1,
            position_side="LONG",
            reduce_only=True,
        )
    )
    assert result is not None
    assert pytest.approx(result.closed_qty) == 1.0
    assert pytest.approx(result.realized_pnl) == pytest.approx(10.0)
    assert pytest.approx(result.fees) == pytest.approx(0.1)
    assert pytest.approx(result.position_before) == pytest.approx(2.0)
    assert pytest.approx(result.position_after) == pytest.approx(1.0)


def test_position_tracker_fifo_short_close() -> None:
    tracker = PositionTracker()
    tracker.apply_fill(PnlFill(symbol="ETHUSDT", side="SELL", qty=3.0, price=200.0, position_side="SHORT"))
    result = tracker.apply_fill(
        PnlFill(
            symbol="ETHUSDT",
            side="BUY",
            qty=3.0,
            price=180.0,
            fee=0.3,
            position_side="SHORT",
            reduce_only=True,
        )
    )
    assert result is not None
    assert pytest.approx(result.closed_qty) == pytest.approx(3.0)
    assert pytest.approx(result.realized_pnl) == pytest.approx(60.0)
    assert pytest.approx(result.fees) == pytest.approx(0.3)
    assert pytest.approx(result.position_before) == pytest.approx(-3.0)
    assert pytest.approx(result.position_after) == pytest.approx(0.0)
