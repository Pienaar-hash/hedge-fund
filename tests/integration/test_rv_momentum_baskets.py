"""
Tests for v7.5_C1 — RV Momentum Basket Loading.

Tests:
- Correct loading of rv_momo_baskets.json
- Correct classification of symbols into baskets
- Edge cases for symbol membership
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.rv_momentum import (
    load_baskets_config,
    get_symbol_baskets,
)


class TestLoadBasketsConfig:
    """Test basket config loading."""

    def test_loads_from_file(self, tmp_path):
        """Loads basket config from file."""
        baskets_file = tmp_path / "rv_momo_baskets.json"
        baskets_file.write_text(json.dumps({
            "pairs": {
                "btc_vs_eth": {"long": "BTCUSDT", "short": "ETHUSDT"}
            },
            "baskets": {
                "l1": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                "alts": ["LTCUSDT", "LINKUSDT"],
                "meme": ["DOGEUSDT", "WIFUSDT"],
            }
        }))
        
        cfg = load_baskets_config(baskets_file)
        
        assert "pairs" in cfg
        assert "baskets" in cfg
        assert cfg["pairs"]["btc_vs_eth"]["long"] == "BTCUSDT"
        assert "SOLUSDT" in cfg["baskets"]["l1"]

    def test_returns_default_on_missing_file(self):
        """Returns default config when file doesn't exist."""
        cfg = load_baskets_config(Path("/nonexistent/path.json"))
        
        assert "pairs" in cfg
        assert "baskets" in cfg
        assert "btc_vs_eth" in cfg["pairs"]

    def test_returns_default_on_invalid_json(self, tmp_path):
        """Returns default config on invalid JSON."""
        baskets_file = tmp_path / "invalid.json"
        baskets_file.write_text("not valid json {{{")
        
        cfg = load_baskets_config(baskets_file)
        
        assert "pairs" in cfg
        assert "baskets" in cfg


class TestGetSymbolBaskets:
    """Test symbol basket membership."""

    @pytest.fixture
    def sample_baskets_cfg(self):
        return {
            "pairs": {
                "btc_vs_eth": {"long": "BTCUSDT", "short": "ETHUSDT"}
            },
            "baskets": {
                "l1": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                "alts": ["LTCUSDT", "LINKUSDT", "SUIUSDT"],
                "meme": ["DOGEUSDT", "WIFUSDT"],
            }
        }

    def test_symbol_in_single_basket(self, sample_baskets_cfg):
        """Symbol correctly identified in single basket."""
        baskets = get_symbol_baskets("LTCUSDT", sample_baskets_cfg)
        
        assert baskets == ["alts"]

    def test_symbol_in_multiple_baskets(self):
        """Symbol in multiple baskets returns all."""
        cfg = {
            "baskets": {
                "l1": ["BTCUSDT", "ETHUSDT"],
                "majors": ["BTCUSDT"],  # BTC in both
            }
        }
        
        baskets = get_symbol_baskets("BTCUSDT", cfg)
        
        assert "l1" in baskets
        assert "majors" in baskets

    def test_symbol_not_in_any_basket(self, sample_baskets_cfg):
        """Symbol not in any basket returns empty list."""
        baskets = get_symbol_baskets("UNKNOWNUSDT", sample_baskets_cfg)
        
        assert baskets == []

    def test_case_insensitive_lookup(self, sample_baskets_cfg):
        """Symbol lookup is case-insensitive."""
        baskets_upper = get_symbol_baskets("BTCUSDT", sample_baskets_cfg)
        baskets_lower = get_symbol_baskets("btcusdt", sample_baskets_cfg)
        baskets_mixed = get_symbol_baskets("BtCuSdT", sample_baskets_cfg)
        
        assert baskets_upper == baskets_lower == baskets_mixed == ["l1"]

    def test_empty_baskets_config(self):
        """Empty baskets config returns empty list."""
        cfg = {"baskets": {}}
        
        baskets = get_symbol_baskets("BTCUSDT", cfg)
        
        assert baskets == []

    def test_meme_basket_symbols(self, sample_baskets_cfg):
        """Meme basket symbols correctly identified."""
        doge_baskets = get_symbol_baskets("DOGEUSDT", sample_baskets_cfg)
        wif_baskets = get_symbol_baskets("WIFUSDT", sample_baskets_cfg)
        
        assert doge_baskets == ["meme"]
        assert wif_baskets == ["meme"]


class TestBasketsIntegration:
    """Integration tests for basket classification."""

    def test_all_l1_symbols_classified(self):
        """All L1 basket symbols are correctly classified."""
        cfg = load_baskets_config()  # Use default
        
        l1_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        
        for symbol in l1_symbols:
            baskets = get_symbol_baskets(symbol, cfg)
            assert "l1" in baskets, f"{symbol} should be in l1 basket"

    def test_all_alts_symbols_classified(self):
        """All alts basket symbols are correctly classified."""
        cfg = load_baskets_config()  # Use default
        
        alt_symbols = ["LTCUSDT", "LINKUSDT", "SUIUSDT"]
        
        for symbol in alt_symbols:
            baskets = get_symbol_baskets(symbol, cfg)
            assert "alts" in baskets, f"{symbol} should be in alts basket"

    def test_all_meme_symbols_classified(self):
        """All meme basket symbols are correctly classified."""
        cfg = load_baskets_config()  # Use default
        
        meme_symbols = ["DOGEUSDT", "WIFUSDT"]
        
        for symbol in meme_symbols:
            baskets = get_symbol_baskets(symbol, cfg)
            assert "meme" in baskets, f"{symbol} should be in meme basket"
