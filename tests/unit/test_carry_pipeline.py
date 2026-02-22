"""Tests for carry data pipeline: funding + basis snapshot publishing (v7.9).

Verifies that:
  - exchange_utils.build_funding_and_basis_snapshots() produces valid snapshots
  - Snapshots contain non-null values for BTC/ETH/SOL
  - publish_funding_snapshot / publish_basis_snapshot write atomically
  - carry_score() returns non-neutral when fed real snapshot data
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_premium_index_response() -> list[dict]:
    """Simulated Binance premiumIndex response for BTC, ETH, SOL."""
    now_ms = int(time.time() * 1000)
    return [
        {
            "symbol": "BTCUSDT",
            "markPrice": "97500.00",
            "indexPrice": "97400.00",
            "lastFundingRate": "0.00015",
            "nextFundingTime": now_ms + 28800000,
            "time": now_ms,
        },
        {
            "symbol": "ETHUSDT",
            "markPrice": "3200.00",
            "indexPrice": "3195.00",
            "lastFundingRate": "-0.00008",
            "nextFundingTime": now_ms + 28800000,
            "time": now_ms,
        },
        {
            "symbol": "SOLUSDT",
            "markPrice": "185.50",
            "indexPrice": "185.00",
            "lastFundingRate": "0.00025",
            "nextFundingTime": now_ms + 28800000,
            "time": now_ms,
        },
    ]


class TestBuildFundingAndBasisSnapshots:
    """Test exchange_utils.build_funding_and_basis_snapshots()."""

    def test_produces_valid_snapshots(self):
        from execution.exchange_utils import build_funding_and_basis_snapshots

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_premium_index_response()

        with patch("execution.exchange_utils._req", return_value=mock_resp):
            with patch("execution.exchange_utils.is_dry_run", return_value=False):
                funding, basis = build_funding_and_basis_snapshots()

        assert "symbols" in funding
        assert "symbols" in basis
        assert "BTCUSDT" in funding["symbols"]
        assert "ETHUSDT" in funding["symbols"]
        assert "SOLUSDT" in funding["symbols"]

    def test_funding_rates_non_null(self):
        from execution.exchange_utils import build_funding_and_basis_snapshots

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_premium_index_response()

        with patch("execution.exchange_utils._req", return_value=mock_resp):
            with patch("execution.exchange_utils.is_dry_run", return_value=False):
                funding, _ = build_funding_and_basis_snapshots()

        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            entry = funding["symbols"][sym]
            assert entry["rate"] is not None
            assert isinstance(entry["rate"], float)
            assert entry["funding_rate"] == entry["rate"]

    def test_basis_pct_non_null(self):
        from execution.exchange_utils import build_funding_and_basis_snapshots

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_premium_index_response()

        with patch("execution.exchange_utils._req", return_value=mock_resp):
            with patch("execution.exchange_utils.is_dry_run", return_value=False):
                _, basis = build_funding_and_basis_snapshots()

        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            entry = basis["symbols"][sym]
            assert entry["basis_pct"] is not None
            assert isinstance(entry["basis_pct"], float)
            assert entry["mark_price"] > 0
            assert entry["index_price"] > 0

    def test_basis_pct_computed_correctly(self):
        from execution.exchange_utils import build_funding_and_basis_snapshots

        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_premium_index_response()

        with patch("execution.exchange_utils._req", return_value=mock_resp):
            with patch("execution.exchange_utils.is_dry_run", return_value=False):
                _, basis = build_funding_and_basis_snapshots()

        btc = basis["symbols"]["BTCUSDT"]
        expected = (97500.0 - 97400.0) / 97400.0
        assert abs(btc["basis_pct"] - expected) < 1e-8

    def test_dry_run_returns_mock_data(self):
        from execution.exchange_utils import build_funding_and_basis_snapshots

        with patch("execution.exchange_utils.is_dry_run", return_value=True):
            funding, basis = build_funding_and_basis_snapshots()

        assert "BTCUSDT" in funding["symbols"]
        assert "ETHUSDT" in funding["symbols"]
        assert funding["symbols"]["BTCUSDT"]["rate"] == 0.0001


class TestPublishSnapshots:
    """Test state_publish.publish_funding_snapshot / publish_basis_snapshot."""

    def test_publish_funding_snapshot_writes_file(self, tmp_path):
        from execution.state_publish import publish_funding_snapshot

        data = {"symbols": {"BTCUSDT": {"rate": 0.0003}}}
        publish_funding_snapshot(data, state_dir=tmp_path)

        path = tmp_path / "funding_snapshot.json"
        assert path.exists()
        content = json.loads(path.read_text())
        assert content["symbols"]["BTCUSDT"]["rate"] == 0.0003
        assert "updated_ts" in content

    def test_publish_basis_snapshot_writes_file(self, tmp_path):
        from execution.state_publish import publish_basis_snapshot

        data = {"symbols": {"BTCUSDT": {"basis_pct": 0.001}}}
        publish_basis_snapshot(data, state_dir=tmp_path)

        path = tmp_path / "basis_snapshot.json"
        assert path.exists()
        content = json.loads(path.read_text())
        assert content["symbols"]["BTCUSDT"]["basis_pct"] == 0.001
        assert "updated_ts" in content


class TestCarryScoreWithRealData:
    """Verify carry_score returns non-neutral when fed populated snapshots."""

    def test_positive_funding_short_above_neutral(self):
        from execution.intel.symbol_score_v6 import carry_score

        funding = {"symbols": {"BTCUSDT": {"rate": 0.00015}}}
        basis = {"symbols": {"BTCUSDT": {"basis_pct": 0.001}}}

        result = carry_score("BTCUSDT", "SHORT", funding, basis)
        assert result["score"] > 0.5, "Positive funding should favor SHORT"

    def test_negative_funding_long_above_neutral(self):
        from execution.intel.symbol_score_v6 import carry_score

        funding = {"symbols": {"BTCUSDT": {"rate": -0.00015}}}
        basis = {"symbols": {"BTCUSDT": {"basis_pct": -0.001}}}

        result = carry_score("BTCUSDT", "LONG", funding, basis)
        assert result["score"] > 0.5, "Negative funding should favor LONG"

    def test_zero_rates_neutral(self):
        from execution.intel.symbol_score_v6 import carry_score

        funding = {"symbols": {"BTCUSDT": {"rate": 0.0}}}
        basis = {"symbols": {"BTCUSDT": {"basis_pct": 0.0}}}

        result = carry_score("BTCUSDT", "LONG", funding, basis)
        assert abs(result["score"] - 0.5) < 0.01
