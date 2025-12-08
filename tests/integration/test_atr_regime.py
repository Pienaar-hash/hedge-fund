"""
Tests for ATR regime classification (v7).

Test cases:
- ATR bucket boundaries
- percentile handling
- stability across calls
"""

from execution.utils.vol import (
    ATR_REGIME_LOW,
    ATR_REGIME_NORMAL,
    ATR_REGIME_ELEVATED,
    ATR_REGIME_EXTREME,
    DD_REGIME_LOW,
    DD_REGIME_MODERATE,
    DD_REGIME_HIGH,
    DD_REGIME_CRITICAL,
    DEFAULT_ATR_PERCENTILES,
    classify_atr_regime,
    classify_dd_regime,
    compute_atr_regime,
    get_atr_regime_name,
    get_dd_regime_name,
    build_regime_snapshot,
)


class TestClassifyAtrRegime:
    """Unit tests for classify_atr_regime function."""

    def test_low_regime_at_boundary(self):
        """ATR at p20 boundary should be Low (<=)"""
        result = classify_atr_regime(0.15)  # Default p20 = 0.15
        assert result == ATR_REGIME_LOW

    def test_low_regime_below_boundary(self):
        """ATR below p20 should be Low"""
        result = classify_atr_regime(0.10)
        assert result == ATR_REGIME_LOW

    def test_low_regime_at_zero(self):
        """ATR at 0 should be Low"""
        result = classify_atr_regime(0.0)
        assert result == ATR_REGIME_LOW

    def test_normal_regime_just_above_p20(self):
        """ATR just above p20 should be Normal"""
        result = classify_atr_regime(0.16)  # Just above p20=0.15
        assert result == ATR_REGIME_NORMAL

    def test_normal_regime_at_p50(self):
        """ATR at p50 boundary should be Normal (<=)"""
        result = classify_atr_regime(0.25)  # Default p50 = 0.25
        assert result == ATR_REGIME_NORMAL

    def test_elevated_regime_just_above_p50(self):
        """ATR just above p50 should be Elevated"""
        result = classify_atr_regime(0.26)  # Just above p50=0.25
        assert result == ATR_REGIME_ELEVATED

    def test_elevated_regime_at_p80(self):
        """ATR at p80 boundary should be Elevated (<=)"""
        result = classify_atr_regime(0.40)  # Default p80 = 0.40
        assert result == ATR_REGIME_ELEVATED

    def test_extreme_regime_above_p80(self):
        """ATR above p80 should be Extreme"""
        result = classify_atr_regime(0.41)  # Just above p80=0.40
        assert result == ATR_REGIME_EXTREME

    def test_extreme_regime_very_high(self):
        """Very high ATR should be Extreme"""
        result = classify_atr_regime(1.0)
        assert result == ATR_REGIME_EXTREME


class TestCustomPercentiles:
    """Test custom percentile thresholds."""

    def test_custom_percentiles_low(self):
        """Custom percentiles should affect classification"""
        custom = {"p20": 0.10, "p50": 0.20, "p80": 0.30}
        result = classify_atr_regime(0.10, custom)
        assert result == ATR_REGIME_LOW

    def test_custom_percentiles_normal(self):
        """Custom percentiles - Normal range"""
        custom = {"p20": 0.10, "p50": 0.20, "p80": 0.30}
        result = classify_atr_regime(0.15, custom)
        assert result == ATR_REGIME_NORMAL

    def test_custom_percentiles_elevated(self):
        """Custom percentiles - Elevated range"""
        custom = {"p20": 0.10, "p50": 0.20, "p80": 0.30}
        result = classify_atr_regime(0.25, custom)
        assert result == ATR_REGIME_ELEVATED

    def test_custom_percentiles_extreme(self):
        """Custom percentiles - Extreme range"""
        custom = {"p20": 0.10, "p50": 0.20, "p80": 0.30}
        result = classify_atr_regime(0.35, custom)
        assert result == ATR_REGIME_EXTREME

    def test_partial_percentiles_uses_defaults(self):
        """Missing percentile keys should use defaults"""
        custom = {"p20": 0.10}  # Only p20 provided
        result = classify_atr_regime(0.30, custom)
        # p50=0.25 (default), p80=0.40 (default)
        # 0.30 > 0.25 and <= 0.40, so Elevated
        assert result == ATR_REGIME_ELEVATED


class TestClassifyDdRegime:
    """Unit tests for classify_dd_regime function."""

    def test_low_regime_below_threshold(self):
        """DD below 0.05 should be Low"""
        result = classify_dd_regime(0.03)
        assert result == DD_REGIME_LOW

    def test_low_regime_at_zero(self):
        """DD at 0 should be Low"""
        result = classify_dd_regime(0.0)
        assert result == DD_REGIME_LOW

    def test_moderate_regime_at_boundary(self):
        """DD at 0.05 boundary should be Moderate (>=)"""
        result = classify_dd_regime(0.05)
        assert result == DD_REGIME_MODERATE

    def test_moderate_regime_mid_range(self):
        """DD in moderate range"""
        result = classify_dd_regime(0.10)
        assert result == DD_REGIME_MODERATE

    def test_high_regime_at_boundary(self):
        """DD at 0.15 boundary should be High (>=)"""
        result = classify_dd_regime(0.15)
        assert result == DD_REGIME_HIGH

    def test_high_regime_mid_range(self):
        """DD in high range"""
        result = classify_dd_regime(0.22)
        assert result == DD_REGIME_HIGH

    def test_critical_regime_at_boundary(self):
        """DD at 0.30 boundary should be Critical (>=)"""
        result = classify_dd_regime(0.30)
        assert result == DD_REGIME_CRITICAL

    def test_critical_regime_high_value(self):
        """DD above 0.30 should be Critical"""
        result = classify_dd_regime(0.50)
        assert result == DD_REGIME_CRITICAL


class TestStabilityAcrossCalls:
    """Test stability of classification across multiple calls."""

    def test_atr_regime_stability(self):
        """Same input should always produce same output"""
        results = [classify_atr_regime(0.20) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_dd_regime_stability(self):
        """Same input should always produce same output"""
        results = [classify_dd_regime(0.12) for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_boundary_stability(self):
        """Boundary values should be consistent"""
        # Test each boundary multiple times
        boundaries = [0.15, 0.25, 0.40]  # p20, p50, p80
        for b in boundaries:
            results = [classify_atr_regime(b) for _ in range(5)]
            assert all(r == results[0] for r in results)


class TestGetRegimeNames:
    """Test regime name functions."""

    def test_atr_regime_names(self):
        """ATR regime names should be correct"""
        assert get_atr_regime_name(ATR_REGIME_LOW) == "low"
        assert get_atr_regime_name(ATR_REGIME_NORMAL) == "normal"
        assert get_atr_regime_name(ATR_REGIME_ELEVATED) == "elevated"
        assert get_atr_regime_name(ATR_REGIME_EXTREME) == "extreme"

    def test_dd_regime_names(self):
        """DD regime names should be correct"""
        assert get_dd_regime_name(DD_REGIME_LOW) == "low"
        assert get_dd_regime_name(DD_REGIME_MODERATE) == "moderate"
        assert get_dd_regime_name(DD_REGIME_HIGH) == "high"
        assert get_dd_regime_name(DD_REGIME_CRITICAL) == "critical"

    def test_invalid_regime_name(self):
        """Invalid regime should return unknown"""
        assert get_atr_regime_name(99) == "unknown"
        assert get_dd_regime_name(-1) == "unknown"


class TestComputeAtrRegime:
    """Test compute_atr_regime function."""

    def test_returns_dict_with_required_fields(self):
        """compute_atr_regime should return dict with all required fields"""
        result = compute_atr_regime("BTCUSDT", atr_value=0.20)
        
        assert "symbol" in result
        assert "atr_value" in result
        assert "atr_regime" in result
        assert "atr_regime_name" in result
        assert "percentiles" in result

    def test_uses_provided_atr_value(self):
        """Should use provided ATR value"""
        result = compute_atr_regime("BTCUSDT", atr_value=0.30)
        assert result["atr_value"] == 0.30
        assert result["atr_regime"] == ATR_REGIME_ELEVATED

    def test_includes_percentiles(self):
        """Should include percentiles in result"""
        result = compute_atr_regime("BTCUSDT", atr_value=0.20)
        assert result["percentiles"] == DEFAULT_ATR_PERCENTILES


class TestBuildRegimeSnapshot:
    """Test build_regime_snapshot function."""

    def test_returns_complete_snapshot(self):
        """build_regime_snapshot should return complete snapshot"""
        result = build_regime_snapshot(atr_value=0.20, dd_frac=0.08)
        
        assert "atr_regime" in result
        assert "atr_regime_name" in result
        assert "atr_value" in result
        assert "dd_regime" in result
        assert "dd_regime_name" in result
        assert "dd_frac" in result
        assert "regime_matrix" in result
        assert "updated_ts" in result

    def test_correct_regime_assignment(self):
        """Should assign correct regimes"""
        result = build_regime_snapshot(atr_value=0.20, dd_frac=0.08)
        
        assert result["atr_regime"] == ATR_REGIME_NORMAL  # 0.20 is in Normal range
        assert result["dd_regime"] == DD_REGIME_MODERATE  # 0.08 is in Moderate range

    def test_defaults_to_zero(self):
        """Should default to zero when values not provided"""
        result = build_regime_snapshot()
        
        assert result["atr_value"] == 0.0
        assert result["dd_frac"] == 0.0
        assert result["atr_regime"] == ATR_REGIME_LOW
        assert result["dd_regime"] == DD_REGIME_LOW
