"""Tests for execution.order_dispatch — extracted dispatch functions."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from execution.order_dispatch import (
    DispatchRetryContext,
    attempt_maker_first,
    build_maker_metrics,
    dispatch_to_exchange,
    dispatch_with_retry,
    meta_float,
)


# ── meta_float ────────────────────────────────────────────────────────


class TestMetaFloat:
    def test_valid_float(self):
        assert meta_float("3.14", 0.0) == 3.14

    def test_valid_int(self):
        assert meta_float(42, 0.0) == 42.0

    def test_none_returns_fallback(self):
        assert meta_float(None, 9.9) == 9.9

    def test_garbage_string_returns_fallback(self):
        assert meta_float("abc", 1.5) == 1.5


# ── dispatch_to_exchange ──────────────────────────────────────────────


class TestDispatchToExchange:
    def test_calls_send_fn_with_payload(self):
        send_fn = MagicMock(return_value={"orderId": 123})
        close_fn = MagicMock(return_value=(False, 0))
        payload = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": "0.01",
            "positionSide": "LONG",
        }
        result = dispatch_to_exchange(payload, [], send_fn, close_fn)
        assert result == {"orderId": 123}
        assert send_fn.call_count == 1

    def test_converts_close_position_for_non_market(self):
        send_fn = MagicMock(return_value={"orderId": 456})
        close_fn = MagicMock(return_value=(True, 0.5))
        payload = {
            "symbol": "ETHUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "quantity": "1.0",
            "positionSide": "LONG",
            "reduceOnly": True,
            "price": "3000",
        }
        result = dispatch_to_exchange(payload, [], send_fn, close_fn)
        assert result == {"orderId": 456}
        # closePosition should be set, quantity and reduceOnly removed
        call_kwargs = send_fn.call_args[1]
        assert call_kwargs["closePosition"] is True
        assert call_kwargs.get("quantity") is None

    def test_does_not_convert_close_for_market(self):
        """Even if close_fn says convert, MARKET type should pass through."""
        send_fn = MagicMock(return_value={"orderId": 789})
        close_fn = MagicMock(return_value=(True, 0.5))
        payload = {
            "symbol": "ETHUSDT",
            "side": "SELL",
            "type": "MARKET",
            "quantity": "1.0",
            "positionSide": "LONG",
        }
        dispatch_to_exchange(payload, [], send_fn, close_fn)
        call_kwargs = send_fn.call_args[1]
        # MARKET type — closePosition should NOT be set
        assert call_kwargs.get("closePosition") is None

    def test_string_reduce_only_coerced(self):
        send_fn = MagicMock(return_value={})
        close_fn = MagicMock(return_value=(False, 0))
        payload = {
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "MARKET",
            "quantity": "0.01",
            "positionSide": "SHORT",
            "reduceOnly": "true",
        }
        dispatch_to_exchange(payload, [], send_fn, close_fn)
        call_kwargs = send_fn.call_args[1]
        assert call_kwargs["reduceOnly"] is True

    def test_positions_passed_through(self):
        positions = [{"symbol": "BTCUSDT", "positionAmt": "0.1"}]
        send_fn = MagicMock(return_value={})
        close_fn = MagicMock(return_value=(False, 0))
        payload = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "quantity": "0.01",
        }
        dispatch_to_exchange(payload, positions, send_fn, close_fn)
        # close_fn should receive positions
        assert close_fn.call_args[1]["positions"] == positions
        # send_fn should receive positions
        assert send_fn.call_args[1]["positions"] == positions


# ── attempt_maker_first ───────────────────────────────────────────────


class TestAttemptMakerFirst:
    def test_returns_none_when_fns_missing(self):
        assert attempt_maker_first(100.0, 1.0, "BTCUSDT", "BUY") is None

    def test_returns_none_for_zero_price(self):
        assert attempt_maker_first(
            0.0, 1.0, "BTCUSDT", "BUY",
            submit_limit_fn=MagicMock(),
            effective_px_fn=MagicMock(),
        ) is None

    def test_returns_none_for_zero_qty(self):
        assert attempt_maker_first(
            100.0, 0.0, "BTCUSDT", "BUY",
            submit_limit_fn=MagicMock(),
            effective_px_fn=MagicMock(),
        ) is None

    def test_happy_path(self):
        mock_result = MagicMock()
        submit = MagicMock(return_value=mock_result)
        eff_px = MagicMock(return_value=99.5)
        result = attempt_maker_first(
            100.0, 1.0, "BTCUSDT", "BUY",
            submit_limit_fn=submit,
            effective_px_fn=eff_px,
        )
        assert result is mock_result
        eff_px.assert_called_once_with(100.0, "BUY", is_maker=True)
        submit.assert_called_once_with("BTCUSDT", 99.5, 1.0, "BUY")

    def test_exception_returns_none(self):
        submit = MagicMock(side_effect=RuntimeError("fail"))
        eff_px = MagicMock(return_value=99.5)
        result = attempt_maker_first(
            100.0, 1.0, "BTCUSDT", "BUY",
            submit_limit_fn=submit,
            effective_px_fn=eff_px,
        )
        assert result is None


# ── build_maker_metrics ──────────────────────────────────────────────


class TestBuildMakerMetrics:
    def test_basic_structure(self):
        result = MagicMock()
        result.price = 100.0
        result.filled_qty = 0.5
        result.qty = 0.5
        result.rejections = 0
        result.slippage_bps = 1.2
        metrics = build_maker_metrics(result, "att-1", 99.8, 15.0)
        assert metrics["attempt_id"] == "att-1"
        assert metrics["route"] == "maker_first"
        assert metrics["prices"]["mark"] == 99.8
        assert metrics["timing_ms"]["decision"] == 15.0
        assert metrics["result"]["retries"] == 0
        assert metrics["slippage_bps"] == 1.2

    def test_none_price_uses_none(self):
        result = MagicMock()
        result.price = None
        result.filled_qty = 0.0
        result.qty = 1.0
        result.rejections = 1
        result.slippage_bps = None
        metrics = build_maker_metrics(result, "att-2", 50.0, 10.0)
        assert metrics["prices"]["avg_fill"] is None
        assert metrics["result"]["status"] == "NEW"


# ── DispatchRetryContext guards ──────────────────────────────────────


class TestDispatchRetryContextGuards:
    def _make_ctx(self, **overrides):
        defaults = dict(
            symbol="BTCUSDT",
            side="BUY",
            pos_side="LONG",
            gross_target=1000.0,
            meta={},
            payload_view={},
            normalized_ctx={},
            max_retries=1,
            retry_backoff_s=0.0,
            note_error_fn=MagicMock(),
            log_order_error_fn=MagicMock(),
            publish_audit_fn=MagicMock(),
            classify_error_fn=MagicMock(return_value={}),
        )
        defaults.update(overrides)
        return DispatchRetryContext(**defaults)

    def test_empty_symbol_raises(self):
        ctx = self._make_ctx(symbol="")
        with pytest.raises(AssertionError, match="non-empty"):
            dispatch_with_retry(MagicMock(), {}, ctx)

    def test_invalid_side_raises(self):
        ctx = self._make_ctx(side="LONG")
        with pytest.raises(AssertionError, match="BUY or SELL"):
            dispatch_with_retry(MagicMock(), {}, ctx)


# ── dispatch_with_retry ──────────────────────────────────────────────


class TestDispatchWithRetry:
    def _make_ctx(self, **overrides):
        defaults = dict(
            symbol="BTCUSDT",
            side="BUY",
            pos_side="LONG",
            gross_target=1000.0,
            meta={},
            payload_view={},
            normalized_ctx={},
            max_retries=1,
            retry_backoff_s=0.0,
            note_error_fn=MagicMock(),
            log_order_error_fn=MagicMock(),
            publish_audit_fn=MagicMock(),
            classify_error_fn=MagicMock(return_value={}),
        )
        defaults.update(overrides)
        return DispatchRetryContext(**defaults)

    def test_success_returns_response(self):
        dispatch_fn = MagicMock(return_value={"orderId": 1})
        ctx = self._make_ctx()
        result = dispatch_with_retry(dispatch_fn, {"symbol": "BTCUSDT"}, ctx)
        assert result == {"orderId": 1}

    def test_http_error_retriable(self):
        """Retriable HTTP error should retry, then succeed."""
        response_mock = MagicMock()
        response_mock.json.return_value = {"code": -1000}
        exc = requests.HTTPError(response=response_mock)
        exc.response = response_mock

        dispatch_fn = MagicMock(side_effect=[exc, {"orderId": 2}])
        classify = MagicMock(return_value={"retriable": True})
        ctx = self._make_ctx(max_retries=2, classify_error_fn=classify)

        result = dispatch_with_retry(dispatch_fn, {}, ctx)
        assert result == {"orderId": 2}
        assert dispatch_fn.call_count == 2

    def test_http_error_non_retriable_raises(self):
        response_mock = MagicMock()
        response_mock.json.return_value = {"code": -2000}
        exc = requests.HTTPError(response=response_mock)
        exc.response = response_mock

        dispatch_fn = MagicMock(side_effect=exc)
        classify = MagicMock(return_value={"retriable": False})
        ctx = self._make_ctx(classify_error_fn=classify)

        with pytest.raises(requests.HTTPError):
            dispatch_with_retry(dispatch_fn, {}, ctx)

    def test_precision_error_returns_none(self):
        """Error code -1111 should return None (abort signal)."""
        response_mock = MagicMock()
        response_mock.json.return_value = {"code": -1111}
        exc = requests.HTTPError(response=response_mock)
        exc.response = response_mock

        dispatch_fn = MagicMock(side_effect=exc)
        classify = MagicMock(return_value={})
        ctx = self._make_ctx(classify_error_fn=classify)

        result = dispatch_with_retry(dispatch_fn, {}, ctx)
        assert result is None

    def test_generic_exception_raises(self):
        dispatch_fn = MagicMock(side_effect=RuntimeError("boom"))
        ctx = self._make_ctx()
        with pytest.raises(RuntimeError, match="boom"):
            dispatch_with_retry(dispatch_fn, {}, ctx)
        # note_error should have been called
        ctx.note_error_fn.assert_called_once()

    def test_callbacks_invoked_on_http_error(self):
        response_mock = MagicMock()
        response_mock.json.return_value = {"code": -9999}
        exc = requests.HTTPError(response=response_mock)
        exc.response = response_mock

        dispatch_fn = MagicMock(side_effect=exc)
        classify = MagicMock(return_value={"retriable": False})
        ctx = self._make_ctx(classify_error_fn=classify)

        with pytest.raises(requests.HTTPError):
            dispatch_with_retry(dispatch_fn, {}, ctx)

        ctx.note_error_fn.assert_called_once()
        ctx.log_order_error_fn.assert_called_once()
        ctx.publish_audit_fn.assert_called_once()
        classify.assert_called_once()
