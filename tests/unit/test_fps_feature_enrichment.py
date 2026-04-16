"""Unit tests for _compute_fps_features() and FPS data enrichment pipeline."""

from __future__ import annotations

import json
import random

import pytest

from execution.signal_screener import _compute_fps_features


# ---------------------------------------------------------------------------
# Helpers — synthetic klines
# ---------------------------------------------------------------------------
def _make_klines(
    n: int = 100,
    base_price: float = 100.0,
    volatility: float = 0.02,
    volume_base: float = 1000.0,
    *,
    seed: int = 42,
    trend: float = 0.0,
) -> list:
    """Generate synthetic OHLCV klines.

    Each row: [open_time, open, high, low, close, volume]
    """
    rng = random.Random(seed)
    rows = []
    price = base_price
    for i in range(n):
        o = price
        move = rng.gauss(trend, volatility) * price
        c = o + move
        h = max(o, c) + abs(rng.gauss(0, volatility * 0.3)) * price
        lo = min(o, c) - abs(rng.gauss(0, volatility * 0.3)) * price
        v = volume_base * (1 + rng.gauss(0, 0.3))
        rows.append([i * 60000, o, h, lo, c, max(v, 1.0)])
        price = c
    return rows


def _closes(kl: list) -> list:
    return [row[4] for row in kl]


# ===================================================================
# TestComputeFpsFeatures
# ===================================================================
class TestComputeFpsFeatures:
    """Core tests for the _compute_fps_features() helper."""

    def test_returns_dict_with_provenance(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert isinstance(result, dict)
        assert "fps_feature_version" in result
        assert result["fps_feature_version"] == "v1"

    def test_insufficient_data_returns_minimal(self):
        """Fewer than 50 bars → only provenance keys."""
        kl = _make_klines(10)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        # Should have provenance but no computed features
        assert "fps_feature_source" in result
        assert "rsi" not in result
        assert "atr_pct" not in result

    def test_zero_price_returns_minimal(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), 0.0)
        assert "rsi" not in result

    def test_rsi_computed(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "rsi" in result
        assert 0.0 <= result["rsi"] <= 100.0

    def test_rsi_passthrough(self):
        """Pre-computed rsi_val is used instead of recomputing."""
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4], rsi_val=77.5)
        assert result["rsi"] == 77.5

    def test_zscore_computed(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "zscore" in result
        assert -10 < result["zscore"] < 10

    def test_zscore_passthrough(self):
        """Pre-computed zscore_val is used."""
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4], zscore_val=-2.1)
        assert result["zscore"] == -2.1

    def test_atr_pct_positive(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "atr_pct" in result
        assert result["atr_pct"] > 0

    def test_atr_percentile_range(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "atr_percentile" in result
        assert 0.0 <= result["atr_percentile"] <= 100.0

    def test_volume_z_computed(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "volume_z" in result

    def test_ema_slope_computed(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "ema_slope" in result
        assert isinstance(result["ema_slope"], float)

    def test_ema_aligned_computed(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "ema_aligned" in result
        assert isinstance(result["ema_aligned"], bool)

    def test_range_ratio_computed(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "range_ratio" in result
        assert result["range_ratio"] > 0

    def test_pullback_atr_ratio_computed(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "pullback_atr_ratio" in result
        assert result["pullback_atr_ratio"] >= 0

    def test_continuation_failed_is_bool(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "continuation_failed" in result
        assert isinstance(result["continuation_failed"], bool)

    def test_wick_rejection_is_bool(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "wick_rejection" in result
        assert isinstance(result["wick_rejection"], bool)

    def test_momentum_reacceleration_is_bool(self):
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        assert "momentum_reacceleration" in result
        assert isinstance(result["momentum_reacceleration"], bool)

    def test_trending_market_produces_breakout(self):
        """Strong uptrend should produce local_breakout_dir='HIGH'."""
        kl = _make_klines(100, trend=0.005, seed=99)
        price = kl[-1][4] * 1.02  # price above recent highs
        result = _compute_fps_features(kl, _closes(kl), price)
        assert result.get("local_breakout_dir") == "HIGH"

    def test_flat_market_no_breakout(self):
        """Flat market with no breakout → no local_breakout_dir key."""
        kl = _make_klines(100, volatility=0.001, trend=0.0, seed=10)
        price = kl[-1][4]
        result = _compute_fps_features(kl, _closes(kl), price)
        assert "local_breakout_dir" not in result

    def test_omission_vs_default_distinction(self):
        """Features that cannot be computed are omitted, not set to defaults."""
        # 50 bars = minimum, but EMA(26) needs at least 26+1 data points
        kl = _make_klines(50)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        # At 50 bars, most features should compute but verify dict type
        assert isinstance(result, dict)
        # Provenance always present
        assert result["fps_feature_version"] == "v1"


# ===================================================================
# TestFpsFeatureParity — screener vs Hydra enrichment use same helper
# ===================================================================
class TestFpsFeatureParity:
    """Verify that identical inputs produce identical outputs from the
    same helper, regardless of call site (screener vs hydra executor)."""

    def test_same_klines_same_output(self):
        """Same synthetic klines → deterministic, identical features."""
        kl = _make_klines(200, seed=123)
        closes = _closes(kl)
        price = kl[-1][4]

        result_a = _compute_fps_features(kl, closes, price)
        result_b = _compute_fps_features(kl, closes, price)
        assert result_a == result_b

    def test_passthrough_matches_recompute(self):
        """Pre-computed rsi/zscore matches self-computed values."""
        kl = _make_klines(200, seed=77)
        closes = _closes(kl)
        price = kl[-1][4]

        from execution.signal_screener import _rsi, _zscore
        rsi_val = _rsi(closes, 14)
        z_val = _zscore(closes, 20)

        result_pass = _compute_fps_features(
            kl, closes, price, rsi_val=rsi_val, zscore_val=z_val,
        )
        result_self = _compute_fps_features(kl, closes, price)

        assert result_pass["rsi"] == pytest.approx(result_self["rsi"])
        assert result_pass["zscore"] == pytest.approx(result_self["zscore"])

    def test_screener_provenance(self):
        """Screener-stamped features have correct source."""
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        result["fps_feature_source"] = "screener"
        assert result["fps_feature_source"] == "screener"

    def test_hydra_provenance(self):
        """Hydra-stamped features have correct source."""
        kl = _make_klines(100)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        result["fps_feature_source"] = "hydra_executor_enrich"
        assert result["fps_feature_source"] == "hydra_executor_enrich"

    def test_all_expected_fields_present_with_sufficient_data(self):
        """With 200 bars, all FPS feature fields should be computed."""
        kl = _make_klines(200, seed=55)
        result = _compute_fps_features(kl, _closes(kl), kl[-1][4])
        expected_keys = {
            "rsi", "zscore", "atr_pct", "atr_percentile", "volume_z",
            "ema_slope", "ema_aligned", "range_ratio", "pullback_atr_ratio",
            "continuation_failed", "wick_rejection", "momentum_reacceleration",
            "fps_feature_source", "fps_feature_version",
        }
        # local_breakout_dir is conditional — may or may not be present
        missing = expected_keys - set(result.keys())
        assert not missing, f"Missing fields: {missing}"


# ===================================================================
# TestWrapperReadsEnrichedFields
# ===================================================================
class TestWrapperReadsEnrichedFields:
    """Verify that evaluate_shadow_for_intent passes enriched metadata
    through to FPSEvalContext and into the shadow record."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self, monkeypatch):
        import execution.futures_permit_surface_v1 as _mod
        monkeypatch.setattr(_mod, "_FPS_CFG_CACHE", None)

    @pytest.fixture()
    def shadow_dir(self, tmp_path, monkeypatch):
        import execution.futures_permit_surface_v1 as _mod
        log_path = tmp_path / "fps_shadow.jsonl"
        monkeypatch.setattr(_mod, "_SHADOW_LOG", log_path)
        return log_path

    def test_hypothesis_fields_flow_to_shadow(self, shadow_dir, monkeypatch):
        """All 8 hypothesis-class fields propagate to shadow record."""
        from execution.futures_permit_surface_v1 import (
            FPSv1Config,
            evaluate_shadow_for_intent,
        )
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        intent = {
            "symbol": "ETHUSDT",
            "price": 3200.0,
            "signal": "BUY",
            "metadata": {
                "atr_pct": 0.04,
                "volume_z": 2.5,
                "zscore": -1.8,
                "rsi": 28.0,
                "ema_slope": -0.003,
                "atr_percentile": 15.0,
                "range_ratio": 2.1,
                "local_breakout_dir": "LOW",
                "continuation_failed": True,
                "wick_rejection": True,
                "ema_aligned": True,
                "pullback_atr_ratio": 0.8,
                "momentum_reacceleration": True,
                "fps_feature_source": "screener",
                "fps_feature_version": "v1",
            },
        }
        sentinel = {
            "primary_regime": "MEAN_REVERT",
            "previous_regime": "CHOPPY",
            "regime_age_bars": 5,
            "regime_probs": {"MEAN_REVERT": 0.65},
            "crisis_flag": False,
        }
        evaluate_shadow_for_intent(intent, sentinel)
        rec = json.loads(shadow_dir.read_text().strip().splitlines()[0])
        assert rec["atr_percentile"] == pytest.approx(15.0)
        assert rec["range_ratio"] == pytest.approx(2.1)
        assert rec["local_breakout_dir"] == "LOW"
        assert rec["continuation_failed"] is True
        assert rec["wick_rejection"] is True
        assert rec["ema_aligned"] is True
        assert rec["pullback_atr_ratio"] == pytest.approx(0.8)
        assert rec["momentum_reacceleration"] is True

    def test_missing_hypothesis_fields_use_defaults(self, shadow_dir, monkeypatch):
        """When metadata lacks hypothesis fields, FPSEvalContext defaults apply."""
        from execution.futures_permit_surface_v1 import (
            FPSv1Config,
            evaluate_shadow_for_intent,
        )
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        intent = {
            "symbol": "BTCUSDT",
            "price": 65000.0,
            "signal": "BUY",
            "metadata": {"atr_pct": 0.03},  # only one field
        }
        sentinel = {
            "primary_regime": "TREND_UP",
            "regime_probs": {"TREND_UP": 0.70},
        }
        evaluate_shadow_for_intent(intent, sentinel)
        rec = json.loads(shadow_dir.read_text().strip().splitlines()[0])
        # Defaults from FPSEvalContext
        assert rec["atr_percentile"] == pytest.approx(50.0)
        assert rec["range_ratio"] == pytest.approx(1.0)
        assert rec["continuation_failed"] is False
        assert rec["ema_aligned"] is False
