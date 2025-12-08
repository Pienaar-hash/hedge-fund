"""
Test suite for execution/correlation_groups.py — correlation group resolver.

Tests symbol-to-group mapping and exposure calculations.
"""

import pytest
from decimal import Decimal
from typing import Set
from unittest.mock import patch, MagicMock

from execution.correlation_groups import (
    build_symbol_to_groups_index,
    compute_group_exposure_nav_pct,
    compute_hypothetical_group_exposure_nav_pct,
    get_groups_for_symbol,
)
from execution.risk_loader import CorrelationGroupConfig, CorrelationGroupsConfig


class TestBuildSymbolToGroupsIndex:
    """Tests for symbol-to-group index building."""
    
    def test_builds_correct_index(self) -> None:
        """Index should map each symbol to its group names (as set)."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
                "L2_alts": CorrelationGroupConfig(
                    symbols=["DOGEUSDT", "SOLUSDT"],
                    max_group_nav_pct=0.25,
                ),
            }
        )
        
        index = build_symbol_to_groups_index(config)
        
        assert index["BTCUSDT"] == {"L1_bluechips"}
        assert index["ETHUSDT"] == {"L1_bluechips"}
        assert index["DOGEUSDT"] == {"L2_alts"}
        assert index["SOLUSDT"] == {"L2_alts"}
    
    def test_symbol_in_multiple_groups(self) -> None:
        """Symbol can belong to multiple groups."""
        config = CorrelationGroupsConfig(
            groups={
                "group_a": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.30,
                ),
                "group_b": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "SOLUSDT"],
                    max_group_nav_pct=0.20,
                ),
            }
        )
        
        index = build_symbol_to_groups_index(config)
        
        # BTCUSDT is in both groups
        assert index["BTCUSDT"] == {"group_a", "group_b"}
        assert index["ETHUSDT"] == {"group_a"}
        assert index["SOLUSDT"] == {"group_b"}
    
    def test_empty_groups(self) -> None:
        """Empty groups config produces empty index."""
        config = CorrelationGroupsConfig(groups={})
        
        index = build_symbol_to_groups_index(config)
        
        assert index == {}


class TestGetGroupsForSymbol:
    """Tests for get_groups_for_symbol."""
    
    def test_returns_groups_for_known_symbol(self) -> None:
        """Should return set of groups for known symbol."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        groups = get_groups_for_symbol("BTCUSDT", config)
        
        assert groups == {"L1_bluechips"}
    
    def test_returns_empty_for_unknown_symbol(self) -> None:
        """Should return empty set for symbol not in any group."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        groups = get_groups_for_symbol("XYZUSDT", config)
        
        assert groups == set()


class TestComputeGroupExposureNavPct:
    """Tests for group exposure calculation."""
    
    def test_computes_gross_exposure_for_group(self) -> None:
        """Should sum absolute position sizes for group symbols."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        positions = [
            {"symbol": "BTCUSDT", "notional": 10000},  # long
            {"symbol": "ETHUSDT", "notional": -5000},  # short (uses abs)
            {"symbol": "DOGEUSDT", "notional": 2000},  # not in group
        ]
        
        nav_usd = 50000.0
        
        result = compute_group_exposure_nav_pct(
            positions=positions,
            nav_total_usd=nav_usd,
            corr_cfg=config,
        )
        
        # (|10000| + |-5000|) / 50000 = 15000 / 50000 = 0.30
        assert result["L1_bluechips"] == pytest.approx(0.30)
    
    def test_returns_zero_for_empty_positions(self) -> None:
        """Should return 0 when no positions exist."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        result = compute_group_exposure_nav_pct(
            positions=[],
            nav_total_usd=50000.0,
            corr_cfg=config,
        )
        
        assert result["L1_bluechips"] == 0.0
    
    def test_returns_zero_for_zero_nav(self) -> None:
        """Should return 0 when NAV is zero to avoid division by zero."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        positions = [
            {"symbol": "BTCUSDT", "notional": 10000},
        ]
        
        result = compute_group_exposure_nav_pct(
            positions=positions,
            nav_total_usd=0.0,
            corr_cfg=config,
        )
        
        assert result["L1_bluechips"] == 0.0


class TestComputeHypotheticalGroupExposureNavPct:
    """Tests for hypothetical group exposure with proposed order."""
    
    def test_adds_proposed_order_to_exposure(self) -> None:
        """Should add proposed order size to existing exposure."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        positions = [
            {"symbol": "BTCUSDT", "notional": 10000},
        ]
        
        nav_usd = 50000.0
        
        result = compute_hypothetical_group_exposure_nav_pct(
            positions=positions,
            nav_total_usd=nav_usd,
            corr_cfg=config,
            order_symbol="ETHUSDT",
            order_notional_usd=5000,
        )
        
        # (|10000| + |5000|) / 50000 = 15000 / 50000 = 0.30
        assert result["L1_bluechips"] == pytest.approx(0.30)
    
    def test_handles_short_proposed_order(self) -> None:
        """Should use absolute value for negative (short) proposed order."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        positions = [
            {"symbol": "BTCUSDT", "notional": 10000},
        ]
        
        nav_usd = 50000.0
        
        result = compute_hypothetical_group_exposure_nav_pct(
            positions=positions,
            nav_total_usd=nav_usd,
            corr_cfg=config,
            order_symbol="ETHUSDT",
            order_notional_usd=-5000,  # short
        )
        
        # (|10000| + |-5000|) / 50000 = 15000 / 50000 = 0.30
        assert result["L1_bluechips"] == pytest.approx(0.30)
    
    def test_proposed_order_not_in_group(self) -> None:
        """Should not add order if proposed symbol is not in the group."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        positions = [
            {"symbol": "BTCUSDT", "notional": 10000},
        ]
        
        nav_usd = 50000.0
        
        result = compute_hypothetical_group_exposure_nav_pct(
            positions=positions,
            nav_total_usd=nav_usd,
            corr_cfg=config,
            order_symbol="DOGEUSDT",  # not in group
            order_notional_usd=5000,
        )
        
        # Only existing position: 10000 / 50000 = 0.20
        assert result["L1_bluechips"] == pytest.approx(0.20)
    
    def test_zero_proposed_notional(self) -> None:
        """Should handle zero proposed notional gracefully."""
        config = CorrelationGroupsConfig(
            groups={
                "L1_bluechips": CorrelationGroupConfig(
                    symbols=["BTCUSDT", "ETHUSDT"],
                    max_group_nav_pct=0.35,
                ),
            }
        )
        
        positions = [
            {"symbol": "BTCUSDT", "notional": 10000},
        ]
        
        nav_usd = 50000.0
        
        result = compute_hypothetical_group_exposure_nav_pct(
            positions=positions,
            nav_total_usd=nav_usd,
            corr_cfg=config,
            order_symbol="ETHUSDT",
            order_notional_usd=0,
        )
        
        # Only existing position: 10000 / 50000 = 0.20
        assert result["L1_bluechips"] == pytest.approx(0.20)
