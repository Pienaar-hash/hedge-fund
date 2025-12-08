"""
Tests for liquidity_model.py (v7.5_B1)
"""

import json
import pytest
import tempfile
from pathlib import Path


# ===========================================================================
# Tests: LiquidityBucketConfig and LiquidityModel
# ===========================================================================

class TestLiquidityBucketConfig:
    """Test LiquidityBucketConfig dataclass."""
    
    def test_basic_instantiation(self):
        """Should create bucket config with required fields."""
        from execution.liquidity_model import LiquidityBucketConfig
        
        bucket = LiquidityBucketConfig(
            name="A_HIGH",
            max_spread_bps=5.0,
            default_maker_bias=0.8,
        )
        
        assert bucket.name == "A_HIGH"
        assert bucket.max_spread_bps == 5.0
        assert bucket.default_maker_bias == 0.8

    def test_symbols_default_empty(self):
        """Symbols should default to empty list."""
        from execution.liquidity_model import LiquidityBucketConfig
        
        bucket = LiquidityBucketConfig(
            name="TEST",
            max_spread_bps=10.0,
            default_maker_bias=0.5,
        )
        
        assert bucket.symbols == []


class TestLiquidityModel:
    """Test LiquidityModel dataclass."""
    
    def test_get_bucket_returns_mapped_bucket(self):
        """Should return mapped bucket for known symbol."""
        from execution.liquidity_model import LiquidityModel, LiquidityBucketConfig
        
        high_bucket = LiquidityBucketConfig("A_HIGH", 5.0, 0.8)
        model = LiquidityModel(
            symbol_to_bucket={"BTCUSDT": high_bucket},
            default_bucket=LiquidityBucketConfig("GENERIC", 15.0, 0.5),
        )
        
        bucket = model.get_bucket("BTCUSDT")
        assert bucket.name == "A_HIGH"

    def test_get_bucket_returns_default_for_unknown(self):
        """Should return default bucket for unknown symbol."""
        from execution.liquidity_model import LiquidityModel, LiquidityBucketConfig
        
        model = LiquidityModel(
            symbol_to_bucket={},
            default_bucket=LiquidityBucketConfig("GENERIC", 15.0, 0.5),
        )
        
        bucket = model.get_bucket("UNKNOWNUSDT")
        assert bucket.name == "GENERIC"

    def test_get_bucket_is_case_insensitive(self):
        """Symbol lookup should be case-insensitive (normalized to upper)."""
        from execution.liquidity_model import LiquidityModel, LiquidityBucketConfig
        
        high_bucket = LiquidityBucketConfig("A_HIGH", 5.0, 0.8)
        model = LiquidityModel(
            symbol_to_bucket={"BTCUSDT": high_bucket},
            default_bucket=LiquidityBucketConfig("GENERIC", 15.0, 0.5),
        )
        
        bucket = model.get_bucket("btcusdt")
        assert bucket.name == "A_HIGH"


# ===========================================================================
# Tests: load_liquidity_model
# ===========================================================================

class TestLoadLiquidityModel:
    """Test load_liquidity_model function."""
    
    def test_loads_valid_config(self):
        """Should load valid config file."""
        from execution.liquidity_model import load_liquidity_model
        
        config = {
            "buckets": {
                "A_HIGH": {
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "max_spread_bps": 5,
                    "default_maker_bias": 0.8,
                },
                "B_MEDIUM": {
                    "symbols": ["SOLUSDT"],
                    "max_spread_bps": 12,
                    "default_maker_bias": 0.6,
                },
            },
            "defaults": {
                "max_spread_bps": 15,
                "default_maker_bias": 0.5,
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "liquidity_buckets.json"
            with path.open("w") as f:
                json.dump(config, f)
            
            model = load_liquidity_model(path)
            
            assert model.get_bucket("BTCUSDT").name == "A_HIGH"
            assert model.get_bucket("ETHUSDT").name == "A_HIGH"
            assert model.get_bucket("SOLUSDT").name == "B_MEDIUM"
            assert model.get_bucket("UNKNOWNUSDT").name == "GENERIC"

    def test_handles_missing_file(self):
        """Should return default model if file missing."""
        from execution.liquidity_model import load_liquidity_model
        
        model = load_liquidity_model("/nonexistent/path.json")
        
        assert model.default_bucket.name == "GENERIC"
        assert model.default_bucket.max_spread_bps == 15.0

    def test_handles_duplicate_symbols(self):
        """Should handle duplicate symbols (last wins)."""
        from execution.liquidity_model import load_liquidity_model
        
        config = {
            "buckets": {
                "A_HIGH": {
                    "symbols": ["BTCUSDT"],
                    "max_spread_bps": 5,
                    "default_maker_bias": 0.8,
                },
                "B_MEDIUM": {
                    "symbols": ["BTCUSDT"],  # Duplicate!
                    "max_spread_bps": 12,
                    "default_maker_bias": 0.6,
                },
            },
            "defaults": {},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "liquidity_buckets.json"
            with path.open("w") as f:
                json.dump(config, f)
            
            model = load_liquidity_model(path)
            
            # B_MEDIUM comes after A_HIGH alphabetically, so it should win
            assert model.get_bucket("BTCUSDT").name == "B_MEDIUM"

    def test_handles_empty_config(self):
        """Should handle empty config gracefully."""
        from execution.liquidity_model import load_liquidity_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "liquidity_buckets.json"
            with path.open("w") as f:
                json.dump({}, f)
            
            model = load_liquidity_model(path)
            
            assert model.default_bucket.name == "GENERIC"


# ===========================================================================
# Tests: Helper Functions
# ===========================================================================

class TestHelperFunctions:
    """Test module-level helper functions."""
    
    def test_get_max_spread_bps(self):
        """get_max_spread_bps should return bucket's max spread."""
        from execution.liquidity_model import (
            get_max_spread_bps,
            reload_liquidity_model,
        )
        
        # Reload with actual config
        reload_liquidity_model()
        
        # BTCUSDT should be A_HIGH with 5 bps
        spread = get_max_spread_bps("BTCUSDT")
        assert spread == 5.0

    def test_get_default_maker_bias(self):
        """get_default_maker_bias should return bucket's maker bias."""
        from execution.liquidity_model import (
            get_default_maker_bias,
            reload_liquidity_model,
        )
        
        reload_liquidity_model()
        
        # BTCUSDT should be A_HIGH with 0.8 bias
        bias = get_default_maker_bias("BTCUSDT")
        assert bias == 0.8

    def test_is_high_liquidity(self):
        """is_high_liquidity should return True for A_HIGH bucket."""
        from execution.liquidity_model import (
            is_high_liquidity,
            reload_liquidity_model,
        )
        
        reload_liquidity_model()
        
        assert is_high_liquidity("BTCUSDT") is True
        assert is_high_liquidity("DOGEUSDT") is False

    def test_is_low_liquidity(self):
        """is_low_liquidity should return True for C_LOW bucket."""
        from execution.liquidity_model import (
            is_low_liquidity,
            reload_liquidity_model,
        )
        
        reload_liquidity_model()
        
        assert is_low_liquidity("WIFUSDT") is True
        assert is_low_liquidity("BTCUSDT") is False


# ===========================================================================
# Tests: build_liquidity_snapshot
# ===========================================================================

class TestBuildLiquiditySnapshot:
    """Test build_liquidity_snapshot function."""
    
    def test_returns_symbol_to_bucket_mapping(self):
        """Should return dict mapping symbol to bucket info."""
        from execution.liquidity_model import (
            build_liquidity_snapshot,
            reload_liquidity_model,
        )
        
        reload_liquidity_model()
        snapshot = build_liquidity_snapshot()
        
        assert "BTCUSDT" in snapshot
        assert snapshot["BTCUSDT"]["bucket"] == "A_HIGH"
        assert "max_spread_bps" in snapshot["BTCUSDT"]
        assert "default_maker_bias" in snapshot["BTCUSDT"]

    def test_empty_model_returns_empty_snapshot(self):
        """Should return empty dict if no symbols mapped."""
        from execution.liquidity_model import load_liquidity_model, build_liquidity_snapshot
        import execution.liquidity_model as lm
        
        # Temporarily set empty model
        old_model = lm._LIQUIDITY_MODEL
        lm._LIQUIDITY_MODEL = load_liquidity_model("/nonexistent/path.json")
        
        try:
            snapshot = build_liquidity_snapshot()
            assert snapshot == {}
        finally:
            lm._LIQUIDITY_MODEL = old_model
