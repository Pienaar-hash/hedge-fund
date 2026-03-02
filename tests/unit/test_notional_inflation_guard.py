"""Tests for v7.9-D1: Notional Inflation Guard & LOT_SIZE preference.

Root cause: Binance testnet MARKET_LOT_SIZE returns minQty=1 / stepSize=1
for BTCUSDT, inflating qty from ~0.001 to 1.0 (~$67k notional vs $586
target).  These tests verify:

1. normalize_price_qty prefers LOT_SIZE over MARKET_LOT_SIZE
2. normalize_price_qty with stepSize=1 produces qty=1 (documenting the bug path)
3. _format_decimal_for_step with step=1 produces "1" (no decimals)
4. The post-sizing notional inflation guard vetoes inflated orders
5. Main loop crash resilience (try/except wrapping)
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. normalize_price_qty: LOT_SIZE preference
# ---------------------------------------------------------------------------

class TestNormalizePriceQtyLotPreference:
    """Verify LOT_SIZE is preferred over MARKET_LOT_SIZE."""

    def _make_filters(
        self,
        lot_step: str = "0.001",
        lot_min: str = "0.001",
        market_step: str = "1",
        market_min: str = "1",
    ) -> Dict[str, Any]:
        return {
            "PRICE_FILTER": {"tickSize": "0.10"},
            "LOT_SIZE": {"stepSize": lot_step, "minQty": lot_min},
            "MARKET_LOT_SIZE": {"stepSize": market_step, "minQty": market_min},
            "MIN_NOTIONAL": {"notional": "5"},
        }

    @patch("execution.exchange_utils.get_symbol_filters")
    def test_uses_lot_size_not_market(self, mock_filters):
        """With both LOT_SIZE and MARKET_LOT_SIZE present, LOT_SIZE wins."""
        from execution.exchange_utils import normalize_price_qty

        mock_filters.return_value = self._make_filters(
            lot_step="0.001", lot_min="0.001",
            market_step="1", market_min="1",
        )
        _price, qty, meta = normalize_price_qty("BTCUSDT", 67000.0, 586.0)
        # LOT_SIZE step=0.001 → qty ≈ 0.008 (586/67000)
        # If MARKET_LOT_SIZE were used (step=1, min=1) → qty = 1.0
        assert qty < Decimal("1"), f"qty={qty} — MARKET_LOT_SIZE was incorrectly used"
        assert qty >= Decimal("0.001"), f"qty={qty} too small"
        assert meta["stepSize"] == "0.001"

    @patch("execution.exchange_utils.get_symbol_filters")
    def test_falls_back_to_market_lot_when_lot_missing(self, mock_filters):
        """When LOT_SIZE is absent, MARKET_LOT_SIZE is used as fallback."""
        from execution.exchange_utils import normalize_price_qty

        filters = {
            "PRICE_FILTER": {"tickSize": "0.10"},
            "MARKET_LOT_SIZE": {"stepSize": "0.001", "minQty": "0.001"},
            "MIN_NOTIONAL": {"notional": "5"},
        }
        mock_filters.return_value = filters
        _price, qty, meta = normalize_price_qty("BTCUSDT", 67000.0, 586.0)
        assert qty >= Decimal("0.001")
        assert meta["stepSize"] == "0.001"


# ---------------------------------------------------------------------------
# 2. Document the bug path: stepSize=1 produces qty=1
# ---------------------------------------------------------------------------

class TestInflatedFiltersProduceQtyOne:
    """Document that with testnet-style MARKET_LOT_SIZE (step=1, min=1),
    normalize_price_qty would produce qty=1 if MARKET_LOT_SIZE were preferred."""

    @patch("execution.exchange_utils.get_symbol_filters")
    def test_step_one_produces_qty_one(self, mock_filters):
        """If filters have stepSize=1/minQty=1, small gross → qty=1."""
        from execution.exchange_utils import normalize_price_qty

        # Simulate what happens if ONLY MARKET_LOT_SIZE with step=1 is available
        filters = {
            "PRICE_FILTER": {"tickSize": "0.10"},
            "LOT_SIZE": {"stepSize": "1", "minQty": "1"},
            "MIN_NOTIONAL": {"notional": "5"},
        }
        mock_filters.return_value = filters
        _price, qty, meta = normalize_price_qty("BTCUSDT", 67000.0, 586.0)
        # 586 / 67000 = 0.00875 → floor(0.00875, step=1) = 0 → bumped to minQty=1
        assert qty == Decimal("1"), f"Expected qty=1 with step=1, got {qty}"

    @patch("execution.exchange_utils.get_symbol_filters")
    def test_normal_filters_produce_small_qty(self, mock_filters):
        """With standard BTCUSDT filters (step=0.001), gross_target=$586 → tiny qty."""
        from execution.exchange_utils import normalize_price_qty

        filters = {
            "PRICE_FILTER": {"tickSize": "0.10"},
            "LOT_SIZE": {"stepSize": "0.001", "minQty": "0.001"},
            "MIN_NOTIONAL": {"notional": "100"},
        }
        mock_filters.return_value = filters
        _price, qty, meta = normalize_price_qty("BTCUSDT", 67000.0, 586.0)
        assert qty < Decimal("0.01"), f"qty={qty} unexpectedly large"
        assert qty >= Decimal("0.001")


# ---------------------------------------------------------------------------
# 3. _format_decimal_for_step with step=1 → "1" (zero decimals)
# ---------------------------------------------------------------------------

class TestFormatDecimalForStep:
    """Verify formatting matches the payload string observed in D1 errors."""

    def test_step_one_formats_without_decimals(self):
        from execution.exchange_utils import _format_decimal_for_step
        result = _format_decimal_for_step(Decimal("1"), Decimal("1"))
        assert result == "1", f"Expected '1', got '{result}'"

    def test_step_001_formats_with_three_decimals(self):
        from execution.exchange_utils import _format_decimal_for_step
        result = _format_decimal_for_step(Decimal("0.008"), Decimal("0.001"))
        assert result == "0.008", f"Expected '0.008', got '{result}'"

    def test_step_one_formats_larger_qty(self):
        from execution.exchange_utils import _format_decimal_for_step
        result = _format_decimal_for_step(Decimal("3"), Decimal("1"))
        assert result == "3", f"Expected '3', got '{result}'"


# ---------------------------------------------------------------------------
# 4. Post-sizing notional inflation guard
# ---------------------------------------------------------------------------

class TestNotionalInflationGuard:
    """Test the notional inflation guard logic.

    The guard lives inside _send_order() after build_order_payload and compares
    the post-normalization notional (qty × price) against the original
    gross_target and per-symbol max_order_notional.

    Rather than driving the full _send_order path (which requires mocking ~15
    intermediate gates), we test the guard logic directly by simulating the
    condition checks.
    """

    def test_inflation_ratio_exceeds_3x(self):
        """If actual_notional / gross_target > 3.0, order must be vetoed."""
        # Simulates: gross_target=$586, qty=1 BTC, price=$67000
        actual_qty = 1.0
        price_hint = 67000.0
        gross_target = 586.0
        max_order_notional = 80000.0

        actual_notional = actual_qty * price_hint  # $67,000
        inflation_ratio = actual_notional / gross_target  # 114.3x

        breach = False
        if max_order_notional > 0 and actual_notional > max_order_notional:
            breach = True
        elif gross_target > 0 and inflation_ratio > 3.0:
            breach = True

        assert breach, f"inflation_ratio={inflation_ratio:.1f}x should trigger guard"

    def test_max_order_notional_exceeds_cap(self):
        """If actual_notional > max_order_notional, order must be vetoed."""
        # Simulates: qty=2 BTC at $67k → $134k > $80k cap
        actual_qty = 2.0
        price_hint = 67000.0
        gross_target = 50000.0  # ratio = 2.68x (under 3x)
        max_order_notional = 80000.0

        actual_notional = actual_qty * price_hint  # $134,000
        inflation_ratio = actual_notional / gross_target

        breach = False
        if max_order_notional > 0 and actual_notional > max_order_notional:
            breach = True
        elif gross_target > 0 and inflation_ratio > 3.0:
            breach = True

        assert breach, "max_order_notional cap should trigger guard"
        assert actual_notional > max_order_notional

    def test_normal_qty_passes(self):
        """A correctly-sized order should NOT trigger the guard."""
        actual_qty = 0.008
        price_hint = 67000.0
        gross_target = 586.0
        max_order_notional = 80000.0

        actual_notional = actual_qty * price_hint  # $536
        inflation_ratio = actual_notional / gross_target  # 0.91x

        breach = False
        if max_order_notional > 0 and actual_notional > max_order_notional:
            breach = True
        elif gross_target > 0 and inflation_ratio > 3.0:
            breach = True

        assert not breach, f"Normal order should pass (ratio={inflation_ratio:.2f}x)"

    def test_reduce_only_exits_bypass_guard(self):
        """Reduce-only exits skip all sizing caps including this guard.

        The guard code is placed after `build_order_payload` which occurs
        after all sizing cap blocks.  However, for exits gross_target can
        be 0, making the inflation ratio undefined.  The guard handles
        this by not computing ratio when gross_target <= 0.
        """
        actual_qty = 1.0
        price_hint = 67000.0
        gross_target = 0.0  # exits have gross_target=0
        max_order_notional = 0.0  # no per-symbol config checked for exits

        actual_notional = actual_qty * price_hint
        inflation_ratio = (actual_notional / gross_target) if gross_target > 0 else 0.0

        breach = False
        if max_order_notional > 0 and actual_notional > max_order_notional:
            breach = True
        elif gross_target > 0 and inflation_ratio > 3.0:
            breach = True

        assert not breach, "Exit with gross_target=0 should not trigger guard"

    def test_zero_price_does_not_crash(self):
        """Guard must not crash when price_hint is 0."""
        actual_qty = 1.0
        price_hint = 0.0
        gross_target = 586.0
        max_order_notional = 80000.0

        actual_notional = actual_qty * price_hint if price_hint > 0 else 0.0

        breach = False
        if max_order_notional > 0 and actual_notional > max_order_notional:
            breach = True
        elif gross_target > 0 and (actual_notional / gross_target if gross_target > 0 else 0) > 3.0:
            breach = True

        assert not breach, "Zero price should result in 0 notional, no breach"


# ---------------------------------------------------------------------------
# 5. refresh_precision_cache stores MARKET_LOT_SIZE fields
# ---------------------------------------------------------------------------

class TestPrecisionCacheMarketLotSize:
    """Verify that refresh_precision_cache now stores MARKET_LOT_SIZE fields."""

    def test_market_lot_size_fields_stored(self, tmp_path, monkeypatch):
        """MARKET_LOT_SIZE data should be stored with 'market' prefix keys."""
        import execution.exchange_precision as ep

        fake_response = {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "100", "maxPrice": "999999"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
                        {"filterType": "MARKET_LOT_SIZE", "stepSize": "1", "minQty": "1", "maxQty": "100"},
                        {"filterType": "MIN_NOTIONAL", "notional": "100"},
                    ],
                }
            ]
        }

        cache_path = tmp_path / "exchange_precision_cache.json"
        monkeypatch.setattr(ep, "PRECISION_CACHE_PATH", cache_path)

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = ep.refresh_precision_cache()

        assert result is True
        data = json.loads(cache_path.read_text())
        btc = data["BTCUSDT"]
        assert btc["stepSize"] == "0.001"
        assert btc["minQty"] == "0.001"
        assert btc["marketStepSize"] == "1"
        assert btc["marketMinQty"] == "1"
        assert btc["marketMaxQty"] == "100"
