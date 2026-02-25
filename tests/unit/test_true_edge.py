"""
Tests for True Edge v1: ATR × confidence mapping (v7.9-TE1).

Validates:
- Edge increases with confidence (monotonic)
- Edge increases with ATR (monotonic)
- Symmetry BUY vs SELL (same confidence → same edge pct)
- Missing ATR uses fallback proxy and emits edge_source=fallback_proxy
- expected_edge_usd matches formula exactly
- Config loading and per-timeframe k_atr resolution
- Spread/slippage deduction
- fee_gate v2 integration path
"""

import pytest
from execution.true_edge import (
    TrueEdgeConfig,
    TrueEdgeResult,
    compute_true_edge,
    load_true_edge_config,
    DEFAULT_ADV_CAP,
    DEFAULT_K_ATR,
    DEFAULT_MIN_ATR_PCT,
    K_ATR_DEFAULTS,
    _clamp,
    _resolve_k_atr,
)
from execution.fee_gate import (
    FeeGateConfig,
    check_fee_edge_v2,
)


# ── Config loading ─────────────────────────────────────────────────────────

class TestTrueEdgeConfig:
    def test_defaults(self):
        cfg = TrueEdgeConfig()
        assert cfg.adv_cap == DEFAULT_ADV_CAP
        assert cfg.k_atr == DEFAULT_K_ATR
        assert cfg.min_atr_pct == DEFAULT_MIN_ATR_PCT
        assert cfg.enabled is True
        assert cfg.k_atr_by_timeframe == K_ATR_DEFAULTS

    def test_load_from_dict(self):
        runtime = {
            "true_edge": {
                "adv_cap": 0.30,
                "k_atr": 0.8,
                "min_atr_pct": 0.001,
                "spread_slippage_bps": 2.0,
                "enabled": False,
            }
        }
        cfg = load_true_edge_config(runtime)
        assert cfg.adv_cap == 0.30
        assert cfg.k_atr == 0.8
        assert cfg.min_atr_pct == 0.001
        assert cfg.spread_slippage_bps == 2.0
        assert cfg.enabled is False

    def test_load_with_timeframe_overrides(self):
        runtime = {
            "true_edge": {
                "k_atr_by_timeframe": {"m15": 0.7, "m5": 0.3},
            }
        }
        cfg = load_true_edge_config(runtime)
        assert cfg.k_atr_by_timeframe["m15"] == 0.7
        assert cfg.k_atr_by_timeframe["m5"] == 0.3
        # h1 still has default
        assert cfg.k_atr_by_timeframe["h1"] == K_ATR_DEFAULTS["h1"]

    def test_load_missing_section(self):
        cfg = load_true_edge_config({})
        assert cfg.adv_cap == DEFAULT_ADV_CAP

    def test_load_none_section(self):
        cfg = load_true_edge_config({"true_edge": None})
        assert cfg.enabled is True


# ── Clamp utility ──────────────────────────────────────────────────────────

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.1, 0.0, 0.25) == 0.1

    def test_below_lo(self):
        assert _clamp(-0.1, 0.0, 0.25) == 0.0

    def test_above_hi(self):
        assert _clamp(0.5, 0.0, 0.25) == 0.25

    def test_at_boundary(self):
        assert _clamp(0.25, 0.0, 0.25) == 0.25
        assert _clamp(0.0, 0.0, 0.25) == 0.0


# ── k_atr resolution ──────────────────────────────────────────────────────

class TestResolveKAtr:
    def test_global_default(self):
        cfg = TrueEdgeConfig(k_atr=0.6)
        assert _resolve_k_atr(cfg) == 0.6
        assert _resolve_k_atr(cfg, None) == 0.6

    def test_per_timeframe(self):
        cfg = TrueEdgeConfig(k_atr=0.6, k_atr_by_timeframe={"m15": 0.65, "h1": 0.9})
        assert _resolve_k_atr(cfg, "m15") == 0.65
        assert _resolve_k_atr(cfg, "h1") == 0.9

    def test_unknown_timeframe_uses_global(self):
        cfg = TrueEdgeConfig(k_atr=0.6)
        assert _resolve_k_atr(cfg, "w1") == 0.6


# ── Core edge computation ─────────────────────────────────────────────────

class TestComputeTrueEdge:
    CFG = TrueEdgeConfig(adv_cap=0.25, k_atr=0.6, min_atr_pct=0.0001)

    def test_formula_exact(self):
        """Verify step-by-step: confidence=0.65, ATR=50 on $100k price, $1000 notional."""
        # adv = 0.65 - 0.5 = 0.15
        # atr_pct = 50 / 100000 = 0.0005
        # move_pct = 0.6 * 0.0005 = 0.0003
        # edge_pct = 0.15 * 0.0003 = 0.000045
        # edge_usd = 1000 * 0.000045 = 0.045
        result = compute_true_edge(
            confidence=0.65,
            price=100_000.0,
            atr=50.0,
            notional_usd=1000.0,
            config=self.CFG,
        )
        assert result.source == "atr_conf_v1"
        assert result.fallback_reason == ""
        assert result.adv == pytest.approx(0.15, abs=1e-6)
        assert result.atr_pct == pytest.approx(0.0005, abs=1e-8)
        assert result.expected_edge_pct == pytest.approx(0.000045, abs=1e-8)
        assert result.expected_edge_usd == pytest.approx(0.045, abs=1e-4)

    def test_formula_eth_example(self):
        """ETH-like: confidence=0.70, ATR=$5 on $3000 price, $500 notional."""
        # adv = 0.20
        # atr_pct = 5/3000 = 0.001667
        # move_pct = 0.6 * 0.001667 = 0.001
        # edge_pct = 0.20 * 0.001 = 0.0002
        # edge_usd = 500 * 0.0002 = 0.10
        result = compute_true_edge(
            confidence=0.70,
            price=3000.0,
            atr=5.0,
            notional_usd=500.0,
            config=self.CFG,
        )
        assert result.source == "atr_conf_v1"
        assert result.adv == pytest.approx(0.20, abs=1e-6)
        assert result.atr_pct == pytest.approx(5.0 / 3000.0, abs=1e-8)
        assert result.expected_edge_pct == pytest.approx(0.20 * 0.6 * (5.0 / 3000.0), abs=1e-8)
        assert result.expected_edge_usd == pytest.approx(500.0 * 0.20 * 0.6 * (5.0 / 3000.0), abs=1e-4)

    def test_edge_increases_with_confidence_monotonic(self):
        """Higher confidence → higher edge (monotonic)."""
        edges = []
        for conf in [0.55, 0.60, 0.65, 0.70, 0.75]:
            r = compute_true_edge(
                confidence=conf, price=50000.0, atr=100.0,
                notional_usd=1000.0, config=self.CFG,
            )
            edges.append(r.expected_edge_usd)
        for i in range(1, len(edges)):
            assert edges[i] > edges[i - 1], f"edge[{i}]={edges[i]} not > edge[{i-1}]={edges[i-1]}"

    def test_edge_increases_with_atr_monotonic(self):
        """Higher ATR → higher edge (monotonic)."""
        edges = []
        for atr in [10.0, 50.0, 100.0, 200.0, 500.0]:
            r = compute_true_edge(
                confidence=0.65, price=50000.0, atr=atr,
                notional_usd=1000.0, config=self.CFG,
            )
            edges.append(r.expected_edge_usd)
        for i in range(1, len(edges)):
            assert edges[i] > edges[i - 1], f"edge[{i}]={edges[i]} not > edge[{i-1}]={edges[i-1]}"

    def test_symmetry_buy_vs_sell(self):
        """Same confidence → same edge_pct regardless of side.
        
        True Edge is direction-agnostic. Side is not an input.
        """
        r_buy = compute_true_edge(
            confidence=0.65, price=50000.0, atr=100.0,
            notional_usd=1000.0, config=self.CFG,
        )
        r_sell = compute_true_edge(
            confidence=0.65, price=50000.0, atr=100.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r_buy.expected_edge_pct == r_sell.expected_edge_pct
        assert r_buy.expected_edge_usd == r_sell.expected_edge_usd

    def test_confidence_at_50_gives_zero_edge(self):
        """50% confidence = no advantage → zero edge."""
        r = compute_true_edge(
            confidence=0.50, price=50000.0, atr=100.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.expected_edge_pct == 0.0
        assert r.expected_edge_usd == 0.0
        assert r.adv == 0.0

    def test_confidence_below_50_clamped_to_zero(self):
        """Below 50% confidence → adv clamped to 0."""
        r = compute_true_edge(
            confidence=0.40, price=50000.0, atr=100.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.adv == 0.0
        assert r.expected_edge_pct == 0.0

    def test_adv_cap_limits_extreme_confidence(self):
        """Confidence 0.90 → adv capped at adv_cap (0.25), not 0.40."""
        r = compute_true_edge(
            confidence=0.90, price=50000.0, atr=100.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.adv == pytest.approx(0.25, abs=1e-6)  # capped

    def test_timeframe_overrides_k_atr(self):
        """Per-timeframe k_atr is used when timeframe is specified."""
        cfg = TrueEdgeConfig(k_atr=0.6, k_atr_by_timeframe={"h1": 0.8})
        r_m15 = compute_true_edge(
            confidence=0.65, price=50000.0, atr=100.0,
            notional_usd=1000.0, timeframe="m15", config=cfg,
        )
        r_h1 = compute_true_edge(
            confidence=0.65, price=50000.0, atr=100.0,
            notional_usd=1000.0, timeframe="h1", config=cfg,
        )
        assert r_h1.k_atr == 0.8
        assert r_h1.expected_edge_usd > r_m15.expected_edge_usd


# ── Fallback behavior ─────────────────────────────────────────────────────

class TestFallbackProxy:
    CFG = TrueEdgeConfig(adv_cap=0.25, k_atr=0.6, min_atr_pct=0.0001)

    def test_none_atr_uses_fallback(self):
        """ATR=None → fallback proxy."""
        r = compute_true_edge(
            confidence=0.65, price=50000.0, atr=None,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.source == "fallback_proxy"
        assert r.fallback_reason == "atr_missing"
        assert r.atr_pct == 0.0
        # Proxy: confidence - 0.5 = 0.15
        assert r.expected_edge_pct == pytest.approx(0.15, abs=1e-6)
        assert r.expected_edge_usd == pytest.approx(150.0, abs=0.01)

    def test_zero_atr_uses_fallback(self):
        """ATR=0 → fallback proxy."""
        r = compute_true_edge(
            confidence=0.65, price=50000.0, atr=0.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.source == "fallback_proxy"
        assert r.fallback_reason == "atr_missing"

    def test_negative_atr_uses_fallback(self):
        """ATR<0 → fallback proxy."""
        r = compute_true_edge(
            confidence=0.65, price=50000.0, atr=-10.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.source == "fallback_proxy"
        assert r.fallback_reason == "atr_missing"

    def test_stale_atr_below_min_uses_fallback(self):
        """ATR/price below min_atr_pct → fallback."""
        # min_atr_pct = 0.0001 (0.01%)
        # ATR = 1.0, price = 50000 → atr_pct = 0.00002 < 0.0001
        r = compute_true_edge(
            confidence=0.65, price=50000.0, atr=1.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.source == "fallback_proxy"
        assert r.fallback_reason == "atr_stale"

    def test_zero_price_uses_fallback(self):
        """price=0 → fallback (avoid division by zero)."""
        r = compute_true_edge(
            confidence=0.65, price=0.0, atr=100.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.source == "fallback_proxy"
        assert r.fallback_reason == "price_missing"

    def test_fallback_proxy_value_matches_legacy(self):
        """Fallback edge should match legacy signal_generator: confidence - 0.5."""
        for conf in [0.55, 0.60, 0.70, 0.75]:
            r = compute_true_edge(
                confidence=conf, price=50000.0, atr=None,
                notional_usd=100.0, config=self.CFG,
            )
            assert r.source == "fallback_proxy"
            expected = conf - 0.5
            assert r.expected_edge_pct == pytest.approx(expected, abs=1e-6)

    def test_atr_unit_guard_wrong_units_forces_fallback(self):
        """ATR/price > 1.0 means caller passed atr_pct, not price units.

        E.g. atr=2.5 (percentage) on price=$100k → ratio=0.000025.
        But atr=200000 on price=$100k → ratio=2.0 → suspect → fallback.
        """
        r = compute_true_edge(
            confidence=0.65, price=100000.0, atr=200000.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.source == "fallback_proxy"
        assert r.fallback_reason == "atr_unit_suspect"

    def test_atr_unit_guard_borderline_valid(self):
        """ATR/price = 0.99 is unusual but valid (extreme vol day)."""
        r = compute_true_edge(
            confidence=0.65, price=100.0, atr=99.0,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.source == "atr_conf_v1"
        assert r.fallback_reason == ""

    def test_atr_unit_guard_exactly_1_forces_fallback(self):
        """ATR/price = 1.0001 → just over threshold → fallback."""
        r = compute_true_edge(
            confidence=0.65, price=100.0, atr=100.01,
            notional_usd=1000.0, config=self.CFG,
        )
        assert r.source == "fallback_proxy"
        assert r.fallback_reason == "atr_unit_suspect"


# ── Spread/slippage deduction ─────────────────────────────────────────────

class TestSpreadSlippage:
    def test_spread_deducted_from_edge_usd(self):
        """Spread/slippage reduces expected_edge_usd."""
        cfg = TrueEdgeConfig(adv_cap=0.25, k_atr=0.6, spread_slippage_bps=5.0)
        r = compute_true_edge(
            confidence=0.65, price=50000.0, atr=100.0,
            notional_usd=1000.0, config=cfg,
        )
        # Slippage = 1000 * (5/10000) = $0.50
        assert r.spread_slippage_usd == pytest.approx(0.50, abs=0.01)
        # Edge_usd should be reduced by slippage
        assert r.expected_edge_usd < 1000.0 * r.expected_edge_pct

    def test_override_spread_in_call(self):
        """Per-call spread_slippage_bps overrides config."""
        cfg = TrueEdgeConfig(spread_slippage_bps=0.0)
        r = compute_true_edge(
            confidence=0.65, price=50000.0, atr=100.0,
            notional_usd=1000.0, spread_slippage_bps=10.0, config=cfg,
        )
        # Slippage = 1000 * (10/10000) = $1.00
        assert r.spread_slippage_usd == pytest.approx(1.0, abs=0.01)

    def test_edge_floored_at_zero_with_large_slippage(self):
        """Large slippage can't make edge negative."""
        cfg = TrueEdgeConfig(spread_slippage_bps=1000.0)  # 10% slippage!
        r = compute_true_edge(
            confidence=0.55, price=50000.0, atr=50.0,
            notional_usd=100.0, config=cfg,
        )
        assert r.expected_edge_usd >= 0.0


# ── TrueEdgeResult serialization ──────────────────────────────────────────

class TestTrueEdgeResult:
    def test_to_dict_prefixes_keys(self):
        """to_dict() adds true_edge_v1.* prefix to all fields."""
        r = TrueEdgeResult(
            expected_edge_pct=0.001,
            expected_edge_usd=0.50,
            atr_pct=0.002,
            k_atr=0.6,
            adv=0.15,
            confidence=0.65,
            notional_usd=500.0,
            source="atr_conf_v1",
        )
        d = r.to_dict()
        assert "true_edge_v1.expected_edge_pct" in d
        assert "true_edge_v1.expected_edge_usd" in d
        assert "true_edge_v1.atr_pct" in d
        assert "true_edge_v1.k_atr" in d
        assert "true_edge_v1.adv" in d
        assert "true_edge_v1.source" in d
        assert d["true_edge_v1.source"] == "atr_conf_v1"
        # fallback_reason present and empty for normal path
        assert "true_edge_v1.fallback_reason" in d
        assert d["true_edge_v1.fallback_reason"] == ""

    def test_to_dict_fallback_source(self):
        r = TrueEdgeResult(
            expected_edge_pct=0.10,
            expected_edge_usd=10.0,
            atr_pct=0.0,
            k_atr=0.6,
            adv=0.10,
            confidence=0.60,
            notional_usd=100.0,
            source="fallback_proxy",
            fallback_reason="atr_missing",
        )
        d = r.to_dict()
        assert d["true_edge_v1.source"] == "fallback_proxy"
        assert d["true_edge_v1.atr_pct"] == 0.0
        assert d["true_edge_v1.fallback_reason"] == "atr_missing"


# ── Fee gate v2 integration ───────────────────────────────────────────────

class TestFeeGateV2Integration:
    FG_CFG = FeeGateConfig(taker_fee_rate=0.0004, fee_buffer_mult=1.5)
    TE_CFG = TrueEdgeConfig(adv_cap=0.25, k_atr=0.6)

    def test_pass_with_sufficient_true_edge(self):
        """True edge that beats fee threshold → pass.

        confidence=0.75, ATR=$30 on $3000 price, $500 notional.
        adv=0.25, atr_pct=0.01, move_pct=0.006, edge_pct=0.0015
        edge_usd = 500*0.0015 = $0.75 > required $0.60 → pass.
        """
        te = compute_true_edge(
            confidence=0.75, price=3000.0, atr=30.0,
            notional_usd=500.0, config=self.TE_CFG,
        )
        allowed, details = check_fee_edge_v2(te, config=self.FG_CFG)
        assert allowed is True
        assert details["gate_status"] == "pass"
        assert "true_edge_v1.adv" in details
        assert details["edge_source"] == "atr_conf_v1"

    def test_veto_with_insufficient_edge(self):
        """Very low confidence + small ATR → edge below fee threshold.

        confidence=0.51, ATR=$20 on $100k price, $20 notional.
        adv=0.01, atr_pct=0.0002, move_pct=0.00012, edge_pct=0.0000012
        edge_usd = 20*0.0000012 ≈ $0.000024 < required $0.024 → veto.
        """
        te = compute_true_edge(
            confidence=0.51, price=100000.0, atr=20.0,
            notional_usd=20.0, config=self.TE_CFG,
        )
        allowed, details = check_fee_edge_v2(te, config=self.FG_CFG)
        assert allowed is False
        assert "shortfall_usd" in details

    def test_fallback_proxy_details_in_v2(self):
        """Fallback proxy result passes through fee gate v2 correctly."""
        te = compute_true_edge(
            confidence=0.70, price=50000.0, atr=None,
            notional_usd=1000.0, config=self.TE_CFG,
        )
        assert te.source == "fallback_proxy"
        allowed, details = check_fee_edge_v2(te, config=self.FG_CFG)
        assert allowed is True  # 20% edge on $1000 >> fee threshold
        assert details["edge_source"] == "fallback_proxy"
        assert details["true_edge_v1.source"] == "fallback_proxy"

    def test_v2_details_contain_all_true_edge_fields(self):
        """Fee gate v2 details contain all diagnostic fields."""
        te = compute_true_edge(
            confidence=0.65, price=50000.0, atr=100.0,
            notional_usd=1000.0, config=self.TE_CFG,
        )
        _, details = check_fee_edge_v2(te, config=self.FG_CFG)
        # Fee gate fields
        assert "notional_usd" in details
        assert "round_trip_fee_usd" in details
        assert "required_edge_usd" in details
        # True edge fields
        assert "true_edge_v1.expected_edge_pct" in details
        assert "true_edge_v1.expected_edge_usd" in details
        assert "true_edge_v1.k_atr" in details
        assert "true_edge_v1.adv" in details
        assert "true_edge_v1.confidence" in details
