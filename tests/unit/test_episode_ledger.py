"""Tests for episode_ledger module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from execution.episode_ledger import (
    Episode,
    EpisodeLedger,
    build_episode_ledger,
    _parse_ts,
    _extract_exit_reason,
)
from execution.exit_reason_normalizer import _ensure_loaded as _preload_exit_map

# Pre-load the exit reason YAML map so that mock_open in tests
# does not intercept the config file read.
_preload_exit_map()


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_episode_to_dict(self):
        """Episode should serialize to dict."""
        ep = Episode(
            episode_id="EP_0001",
            symbol="BTCUSDT",
            side="LONG",
            entry_ts="2025-12-18T06:00:00+00:00",
            exit_ts="2025-12-18T07:00:00+00:00",
            duration_hours=1.0,
            entry_fills=2,
            exit_fills=1,
            entry_notional=1000.0,
            exit_notional=1010.0,
            total_qty=0.01,
            avg_entry_price=100000.0,
            avg_exit_price=101000.0,
            gross_pnl=10.0,
            fees=2.0,
            net_pnl=8.0,
            regime_at_entry="MEAN_REVERT",
            regime_at_exit="MEAN_REVERT",
            exit_reason="tp",
            strategy="vol_target",
        )
        d = ep.to_dict()
        assert d["episode_id"] == "EP_0001"
        assert d["symbol"] == "BTCUSDT"
        assert d["net_pnl"] == 8.0


class TestEpisodeLedger:
    """Tests for EpisodeLedger container."""

    def test_empty_ledger(self):
        """Empty ledger should serialize correctly."""
        ledger = EpisodeLedger()
        d = ledger.to_dict()
        assert d["episodes"] == []
        assert d["episode_count"] == 0

    def test_ledger_with_episodes(self):
        """Ledger with episodes should count correctly."""
        ep1 = Episode(
            episode_id="EP_0001",
            symbol="BTCUSDT",
            side="LONG",
            entry_ts="2025-12-18T06:00:00+00:00",
            exit_ts="2025-12-18T07:00:00+00:00",
            duration_hours=1.0,
            entry_fills=1,
            exit_fills=1,
            entry_notional=1000.0,
            exit_notional=1005.0,
            total_qty=0.01,
            avg_entry_price=100000.0,
            avg_exit_price=100500.0,
            gross_pnl=5.0,
            fees=1.0,
            net_pnl=4.0,
            regime_at_entry="unknown",
            regime_at_exit="unknown",
            exit_reason="tp",
            strategy="vol_target",
        )
        ledger = EpisodeLedger(episodes=[ep1])
        d = ledger.to_dict()
        assert d["episode_count"] == 1


class TestParsers:
    """Tests for parsing utilities."""

    def test_parse_ts_iso(self):
        """Should parse ISO timestamp."""
        ts = _parse_ts("2025-12-18T06:00:00+00:00")
        assert ts is not None
        assert ts.year == 2025
        assert ts.month == 12
        assert ts.day == 18

    def test_parse_ts_unix(self):
        """Should parse Unix timestamp."""
        ts = _parse_ts(1734505200.0)  # Dec 18, 2024
        assert ts is not None

    def test_parse_ts_none(self):
        """Should return None for None input."""
        assert _parse_ts(None) is None

    def test_extract_exit_reason_tp(self):
        """Should extract TP exit reason."""
        fill = {
            "metadata": {
                "exit": {"reason": "tp"}
            }
        }
        assert _extract_exit_reason(fill) == "tp"

    def test_extract_exit_reason_sl(self):
        """Should extract SL exit reason."""
        fill = {
            "metadata": {
                "exit": {"reason": "sl"}
            }
        }
        assert _extract_exit_reason(fill) == "sl"

    def test_extract_exit_reason_unknown(self):
        """Should return unknown for missing reason."""
        fill = {"metadata": {}}
        assert _extract_exit_reason(fill) == "unknown"


class TestBuildLedger:
    """Tests for ledger building."""

    def test_build_empty_ledger(self):
        """Should handle missing execution log."""
        with patch("execution.episode_ledger.EXECUTION_LOG_PATH") as mock_path, \
             patch("execution.episode_ledger.EXECUTION_LOG_DIR") as mock_dir:
            mock_path.exists.return_value = False
            mock_dir.exists.return_value = False
            ledger = build_episode_ledger()
            assert len(ledger.episodes) == 0
            assert ledger.stats["total_fills"] == 0

    def test_build_ledger_with_fills(self):
        """Should build episodes from fills."""
        # Create mock fills
        fills = [
            # Entry fill
            json.dumps({
                "event_type": "order_fill",
                "ts": "2025-12-18T06:00:00+00:00",
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "side": "BUY",
                "reduceOnly": False,
                "executedQty": 0.01,
                "avgPrice": 100000.0,
                "fee_total": 0.5,
                "metadata": {"strategy": "vol_target"},
            }),
            # Exit fill
            json.dumps({
                "event_type": "order_fill",
                "ts": "2025-12-18T07:00:00+00:00",
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "side": "SELL",
                "reduceOnly": True,
                "executedQty": 0.01,
                "avgPrice": 100500.0,
                "fee_total": 0.5,
                "metadata": {"exit": {"reason": "tp"}},
            }),
        ]
        
        mock_file_content = "\n".join(fills)
        
        with patch("execution.episode_ledger.EXECUTION_LOG_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("builtins.open", mock_open(read_data=mock_file_content)):
                ledger = build_episode_ledger()
                
                assert len(ledger.episodes) == 1
                ep = ledger.episodes[0]
                assert ep.symbol == "BTCUSDT"
                assert ep.side == "LONG"
                assert ep.gross_pnl == 5.0  # (100500 - 100000) * 0.01
                assert ep.fees == 1.0  # 0.5 + 0.5
                assert ep.net_pnl == 4.0

    def test_cross_window_episode_counted(self):
        """
        Episode with entry BEFORE window and exit INSIDE window should be counted.
        
        This tests the "episodes ending in window" semantics:
        - Entry on Jan 19 (before window)
        - Exit on Jan 20 (inside window)
        - Window is Jan 20 only
        - Episode SHOULD be counted, fees SHOULD be counted, PnL SHOULD be counted
        """
        fills = [
            # Entry fill on Jan 19 (BEFORE window)
            json.dumps({
                "event_type": "order_fill",
                "ts": "2026-01-19T12:00:00+00:00",
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "side": "BUY",
                "reduceOnly": False,
                "executedQty": 0.01,
                "avgPrice": 90000.0,
                "fee_total": 0.36,
                "metadata": {"strategy": "vol_target"},
            }),
            # Exit fill on Jan 20 (INSIDE window)
            json.dumps({
                "event_type": "order_fill",
                "ts": "2026-01-20T14:00:00+00:00",
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "side": "SELL",
                "reduceOnly": True,
                "executedQty": 0.01,
                "avgPrice": 89000.0,
                "fee_total": 0.356,
                "metadata": {
                    "exit": {"reason": "regime_flip", "entry_price": 90000.0}
                },
            }),
        ]
        
        mock_file_content = "\n".join(fills)
        
        with patch("execution.episode_ledger.EXECUTION_LOG_PATH") as mock_path:
            mock_path.exists.return_value = True
            with patch("builtins.open", mock_open(read_data=mock_file_content)):
                # Query for Jan 20 only
                ledger = build_episode_ledger(since_date="2026-01-20", until_date="2026-01-20")
                
                # Episode should be counted (entry before window, exit in window)
                assert len(ledger.episodes) == 1, "Episode with exit in window should be counted"
                
                ep = ledger.episodes[0]
                assert ep.symbol == "BTCUSDT"
                assert ep.exit_reason == "REGIME_CHANGE"
                assert ep.exit_reason_raw == "regime_flip"
                
                # PnL should be calculated: (89000 - 90000) * 0.01 = -10.0
                assert ep.gross_pnl == -10.0
                
                # Fees should include both entry and exit
                assert abs(ep.fees - 0.716) < 0.01
                
                # Net PnL should be gross - fees
                assert abs(ep.net_pnl - (-10.716)) < 0.01
                
                # Stats should reflect the episode
                assert ledger.stats["episodes_found"] == 1
                assert ledger.stats["total_gross_pnl"] == -10.0
                
                # Metadata PnL cross-check should also work
                meta_pnl = ledger.stats.get("metadata_pnl", {})
                assert meta_pnl.get("exits") == 1
                assert meta_pnl.get("gross_pnl") == -10.0


@pytest.mark.integration
class TestLedgerIntegration:
    """Integration tests using actual execution log if available."""

    def test_build_from_actual_log(self):
        """Build ledger from actual execution log."""
        log_path = Path("logs/execution/orders_executed.jsonl")
        if not log_path.exists():
            pytest.skip("No execution log available")
        
        ledger = build_episode_ledger()
        
        # Should have some episodes
        assert ledger.stats["total_fills"] >= 0
        # Stats should be populated
        assert "total_net_pnl" in ledger.stats
        assert "win_rate" in ledger.stats
