"""Unit tests for Futures Permit Surface v1."""

from __future__ import annotations

import json

import pytest

from execution.futures_permit_surface_v1 import (
    ABSTAIN,
    BREAKOUT_RETEST_CONFIRM,
    DENY_STRUCTURAL,
    DISLOCATION_REVERSION,
    EXHAUSTION_REVERSAL,
    LIQUIDITY_VACUUM_RECLAIM,
    LONG,
    PERMIT_CANDIDATE,
    POST_CRISIS_RESTABILIZATION,
    REGIME_TRANSITION_BREAK,
    SHORT,
    TREND_PULLBACK,
    VOL_EXPANSION_BREAKOUT,
    FPSEvalContext,
    FPSResult,
    FPSv1Config,
    _MAX_SETUP_CLASSES,
    _VALID_SETUP_CLASSES,
    compute_burst_risk,
    compute_sparsity,
    evaluate,
    load_fps_config,
    _gate1_structural_admissibility,
    _gate2_direction_validity,
    _gate3_fee_bridge,
    _hash_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_ctx(**overrides) -> FPSEvalContext:
    """Build a default FPSEvalContext with sane defaults, overridable."""
    defaults = dict(
        symbol="BTCUSDT",
        timestamp=1712966400.0,
        regime_current="TREND_UP",
        regime_previous="CHOPPY",
        regime_age_bars=2,
        regime_confidence=0.70,
        crisis_flag=False,
        atr_pct=0.03,
        volume_z=2.0,
        spread_bps=1.0,
        price=60000.0,
        proposed_direction="LONG",
    )
    defaults.update(overrides)
    return FPSEvalContext(**defaults)


# ===================================================================
# TestFPSConfig — authority, mode, influence flags always locked
# ===================================================================
class TestFPSConfig:
    def test_default_authority_none(self):
        cfg = FPSv1Config()
        assert cfg.authority == "none"

    def test_default_mode_shadow_only(self):
        cfg = FPSv1Config()
        assert cfg.mode == "shadow_only"

    def test_authority_forced_on_override_attempt(self):
        """Even if someone passes authority='full', it stays 'none'."""
        cfg = FPSv1Config(authority="full")
        assert cfg.authority == "none"

    def test_mode_forced_on_override_attempt(self):
        cfg = FPSv1Config(mode="live")
        assert cfg.mode == "shadow_only"

    def test_influence_flags_always_false(self):
        cfg = FPSv1Config(
            can_influence_sizing=True,
            can_influence_entry=True,
            can_influence_exit=True,
            can_influence_routing=True,
        )
        assert cfg.can_influence_sizing is False
        assert cfg.can_influence_entry is False
        assert cfg.can_influence_exit is False
        assert cfg.can_influence_routing is False

    def test_load_fps_config_none(self):
        cfg = load_fps_config(None)
        assert cfg.authority == "none"
        assert cfg.mode == "shadow_only"

    def test_load_fps_config_raw_overrides_ignored(self):
        raw = {
            "enabled": True,
            "mode": "live",
            "authority": "full",
            "can_influence_sizing": True,
        }
        cfg = load_fps_config(raw)
        assert cfg.authority == "none"
        assert cfg.mode == "shadow_only"
        assert cfg.can_influence_sizing is False

    def test_load_fps_config_fee_params(self):
        raw = {"taker_fee_rate": 0.001, "fee_buffer_mult": 2.0}
        cfg = load_fps_config(raw)
        assert cfg.taker_fee_rate == 0.001
        assert cfg.fee_buffer_mult == 2.0

    def test_load_fps_config_invalid_setup_classes_ignored(self):
        raw = {"setup_classes": ["REGIME_TRANSITION_BREAK", "INVALID_CLASS"]}
        cfg = load_fps_config(raw)
        assert REGIME_TRANSITION_BREAK in cfg.setup_classes
        assert "INVALID_CLASS" not in cfg.setup_classes

    def test_load_fps_config_all_invalid_gives_defaults(self):
        raw = {"setup_classes": ["BOGUS"]}
        cfg = load_fps_config(raw)
        assert len(cfg.setup_classes) == 8  # all defaults


# ===================================================================
# TestGate1 — structural admissibility (5 setup classes)
# ===================================================================
class TestGate1:
    def test_regime_transition_break(self):
        ctx = _make_ctx(
            regime_previous="CHOPPY",
            regime_current="TREND_UP",
            regime_age_bars=2,
            regime_confidence=0.60,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == REGIME_TRANSITION_BREAK

    def test_regime_transition_break_age_too_old(self):
        ctx = _make_ctx(regime_age_bars=4, regime_confidence=0.60, spread_bps=3.0)
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_regime_transition_break_same_regime(self):
        ctx = _make_ctx(regime_previous="TREND_UP", regime_current="TREND_UP")
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_regime_transition_break_low_confidence(self):
        ctx = _make_ctx(regime_confidence=0.40, spread_bps=3.0)
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_post_crisis_restabilization(self):
        ctx = _make_ctx(
            regime_previous="CRISIS",
            regime_current="TREND_UP",
            crisis_flag=False,
            regime_confidence=0.55,
            regime_age_bars=10,  # prevent transition break
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == POST_CRISIS_RESTABILIZATION

    def test_post_crisis_still_in_crisis(self):
        ctx = _make_ctx(
            regime_previous="CRISIS",
            regime_current="CRISIS",
            crisis_flag=False,
            regime_confidence=0.55,
            regime_age_bars=10,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_post_crisis_flag_still_set(self):
        ctx = _make_ctx(
            regime_previous="CRISIS",
            regime_current="TREND_UP",
            crisis_flag=True,
            regime_confidence=0.55,
            regime_age_bars=10,
            spread_bps=3.0,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_dislocation_reversion(self):
        ctx = _make_ctx(
            regime_current="MEAN_REVERT",
            regime_previous="MEAN_REVERT",  # same — no transition
            volume_z=2.0,
            regime_confidence=0.55,
            regime_age_bars=10,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == DISLOCATION_REVERSION

    def test_dislocation_reversion_low_volume(self):
        ctx = _make_ctx(
            regime_current="MEAN_REVERT",
            regime_previous="MEAN_REVERT",
            volume_z=1.0,
            regime_confidence=0.55,
            regime_age_bars=10,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_breakout_retest_confirm(self):
        ctx = _make_ctx(
            regime_current="BREAKOUT",
            regime_previous="BREAKOUT",  # same — no transition
            regime_age_bars=3,
            regime_confidence=0.65,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == BREAKOUT_RETEST_CONFIRM

    def test_breakout_retest_too_fresh(self):
        ctx = _make_ctx(
            regime_current="BREAKOUT",
            regime_previous="BREAKOUT",
            regime_age_bars=1,
            regime_confidence=0.65,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_liquidity_vacuum_reclaim(self):
        ctx = _make_ctx(
            regime_previous="CHOPPY",
            regime_current="TREND_UP",
            spread_bps=1.5,
            volume_z=1.2,
            regime_age_bars=10,  # prevent transition break match (age > 3)
            regime_confidence=0.40,  # prevent transition break match (conf < 0.55)
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == LIQUIDITY_VACUUM_RECLAIM

    def test_liquidity_vacuum_wide_spread(self):
        ctx = _make_ctx(
            regime_previous="CHOPPY",
            regime_current="TREND_UP",
            spread_bps=3.0,
            volume_z=1.2,
            regime_age_bars=10,
            regime_confidence=0.40,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_no_match_returns_none(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP",
            regime_current="TREND_UP",
            regime_age_bars=10,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_disabled_setup_class_not_matched(self):
        cfg = FPSv1Config(setup_classes=frozenset({BREAKOUT_RETEST_CONFIRM}))
        ctx = _make_ctx(
            regime_previous="CHOPPY",
            regime_current="TREND_UP",
            regime_age_bars=2,
            regime_confidence=0.60,
        )
        # REGIME_TRANSITION_BREAK would match but is not in setup_classes
        assert _gate1_structural_admissibility(ctx, cfg) is None


# ===================================================================
# TestGate2 — direction validity
# ===================================================================
class TestGate2:
    def test_trend_up_long_pass(self):
        ctx = _make_ctx(regime_current="TREND_UP", proposed_direction="LONG")
        assert _gate2_direction_validity(ctx, REGIME_TRANSITION_BREAK) == LONG

    def test_trend_up_short_fail(self):
        ctx = _make_ctx(regime_current="TREND_UP", proposed_direction="SHORT")
        assert _gate2_direction_validity(ctx, REGIME_TRANSITION_BREAK) is None

    def test_trend_down_short_pass(self):
        ctx = _make_ctx(regime_current="TREND_DOWN", proposed_direction="SHORT")
        assert _gate2_direction_validity(ctx, REGIME_TRANSITION_BREAK) == SHORT

    def test_trend_down_long_fail(self):
        ctx = _make_ctx(regime_current="TREND_DOWN", proposed_direction="LONG")
        assert _gate2_direction_validity(ctx, REGIME_TRANSITION_BREAK) is None

    def test_breakout_either_direction(self):
        ctx_long = _make_ctx(regime_current="BREAKOUT", proposed_direction="LONG")
        ctx_short = _make_ctx(regime_current="BREAKOUT", proposed_direction="SHORT")
        assert _gate2_direction_validity(ctx_long, BREAKOUT_RETEST_CONFIRM) == LONG
        assert _gate2_direction_validity(ctx_short, BREAKOUT_RETEST_CONFIRM) == SHORT

    def test_mean_revert_either_direction(self):
        ctx = _make_ctx(regime_current="MEAN_REVERT", proposed_direction="LONG")
        assert _gate2_direction_validity(ctx, DISLOCATION_REVERSION) == LONG


# ===================================================================
# TestGate3 — fee bridge
# ===================================================================
class TestGate3:
    def test_fee_clears_high_atr(self):
        ctx = _make_ctx(atr_pct=0.03)  # 3% ATR
        cfg = FPSv1Config()
        assert _gate3_fee_bridge(ctx, cfg) is True

    def test_fee_denies_low_atr(self):
        # round_trip = 2 * 0.0004 * 1.5 = 0.0012
        # edge = 0.001 * 0.5 = 0.0005 < 0.0012
        ctx = _make_ctx(atr_pct=0.001)
        cfg = FPSv1Config()
        assert _gate3_fee_bridge(ctx, cfg) is False

    def test_fee_boundary_exact(self):
        # round_trip = 2 * 0.0004 * 1.5 = 0.0012
        # edge = atr * 0.5; need edge > 0.0012 → atr > 0.0024
        ctx_below = _make_ctx(atr_pct=0.0024)  # edge = 0.0012, not > 0.0012
        ctx_above = _make_ctx(atr_pct=0.0025)  # edge = 0.00125 > 0.0012
        cfg = FPSv1Config()
        assert _gate3_fee_bridge(ctx_below, cfg) is False
        assert _gate3_fee_bridge(ctx_above, cfg) is True


# ===================================================================
# TestEvaluate — full pipeline
# ===================================================================
class TestEvaluate:
    def test_permit_candidate_full_pass(self):
        ctx = _make_ctx(
            regime_previous="CHOPPY",
            regime_current="TREND_UP",
            regime_age_bars=2,
            regime_confidence=0.65,
            atr_pct=0.03,
            proposed_direction="LONG",
        )
        result = evaluate(ctx)
        assert result.verdict == PERMIT_CANDIDATE
        assert result.setup_class == REGIME_TRANSITION_BREAK
        assert result.direction == LONG
        assert result.snapshot_hash != ""

    def test_abstain_no_setup_match(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP",
            regime_current="TREND_UP",
            regime_age_bars=10,
        )
        result = evaluate(ctx)
        assert result.verdict == ABSTAIN
        assert result.setup_class is None

    def test_abstain_direction_mismatch(self):
        ctx = _make_ctx(
            regime_previous="CHOPPY",
            regime_current="TREND_UP",
            regime_age_bars=2,
            regime_confidence=0.65,
            proposed_direction="SHORT",  # wrong direction for TREND_UP
        )
        result = evaluate(ctx)
        assert result.verdict == ABSTAIN
        assert "g2:direction_mismatch" in result.gate_trace

    def test_deny_structural_fee_bridge(self):
        ctx = _make_ctx(
            regime_previous="CHOPPY",
            regime_current="TREND_UP",
            regime_age_bars=2,
            regime_confidence=0.65,
            atr_pct=0.001,  # too low for fee bridge
            proposed_direction="LONG",
        )
        result = evaluate(ctx)
        assert result.verdict == DENY_STRUCTURAL
        assert result.deny_reason == "fee_bridge_insufficient"

    def test_disabled_returns_abstain(self):
        ctx = _make_ctx()
        cfg = FPSv1Config(enabled=False)
        result = evaluate(ctx, cfg)
        assert result.verdict == ABSTAIN
        assert "disabled" in result.gate_trace


# ===================================================================
# TestDeterminism — same ctx → identical result
# ===================================================================
class TestDeterminism:
    def test_same_input_same_output(self):
        ctx = _make_ctx()
        cfg = FPSv1Config()
        r1 = evaluate(ctx, cfg)
        r2 = evaluate(ctx, cfg)
        assert r1.verdict == r2.verdict
        assert r1.setup_class == r2.setup_class
        assert r1.direction == r2.direction
        assert r1.gate_trace == r2.gate_trace
        assert r1.snapshot_hash == r2.snapshot_hash
        assert r1.snapshot_hash != ""

    def test_hash_changes_with_input(self):
        ctx1 = _make_ctx(price=60000.0)
        ctx2 = _make_ctx(price=61000.0)
        h1 = _hash_context(ctx1)
        h2 = _hash_context(ctx2)
        assert h1 != h2


# ===================================================================
# TestFPSResult — validation
# ===================================================================
class TestFPSResult:
    def test_invalid_verdict_raises(self):
        with pytest.raises(ValueError, match="Invalid verdict"):
            FPSResult(verdict="BUY_NOW")

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="Invalid direction"):
            FPSResult(verdict=PERMIT_CANDIDATE, direction="UP")

    def test_valid_verdicts_accepted(self):
        for v in (PERMIT_CANDIDATE, ABSTAIN, DENY_STRUCTURAL):
            r = FPSResult(verdict=v)
            assert r.verdict == v


# ===================================================================
# TestNonAuthority — import isolation
# ===================================================================
class TestNonAuthority:
    """FPS v1 must NOT import execution authority modules."""

    FORBIDDEN_MODULES = [
        "order_router",
        "doctrine_kernel",
        "sizing",
        "order_dispatch",
    ]

    def test_no_forbidden_imports(self):
        """Source of FPS v1 must not contain imports of forbidden modules."""
        import ast
        import inspect
        import execution.futures_permit_surface_v1 as fps_mod

        source = inspect.getsource(fps_mod)
        tree = ast.parse(source)
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_names.add(node.module)

        for forbidden in self.FORBIDDEN_MODULES:
            for imp in imported_names:
                assert forbidden not in imp, (
                    f"FPS v1 source imports forbidden module: {imp}"
                )


# ===================================================================
# TestSetupSparsityInvariant — prevent categorical entropy (safeguard §6)
# ===================================================================
class TestSetupSparsityInvariant:
    """Setup classes must remain sparse, distinct, and bounded."""

    def _write_shadow_log(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_empty_log_returns_no_data(self, tmp_path):
        log = tmp_path / "empty.jsonl"
        report = compute_sparsity(log)
        assert report.total_evals == 0
        assert "no_data" in report.alerts

    def test_healthy_permit_rate(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # 5% permit rate: 5 permits, 95 abstains
        records = []
        for i in range(95):
            records.append({"verdict": ABSTAIN, "symbol": "BTCUSDT", "regime_current": "TREND_UP", "ts": float(i)})
        for i in range(5):
            records.append({"verdict": PERMIT_CANDIDATE, "symbol": "BTCUSDT", "regime_current": "TREND_UP",
                            "setup_class": REGIME_TRANSITION_BREAK, "ts": float(100 + i)})
        self._write_shadow_log(log, records)
        report = compute_sparsity(log)
        assert report.permit_rate == 0.05
        assert not any("DRIFT" in a for a in report.alerts)
        assert not any("DEAD" in a for a in report.alerts)

    def test_drift_alert_when_too_loose(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # 20% permit rate → above 15% ceiling
        records = []
        for i in range(80):
            records.append({"verdict": ABSTAIN, "symbol": "ETHUSDT", "regime_current": "BREAKOUT", "ts": float(i)})
        for i in range(20):
            records.append({"verdict": PERMIT_CANDIDATE, "symbol": "ETHUSDT", "regime_current": "BREAKOUT",
                            "setup_class": BREAKOUT_RETEST_CONFIRM, "ts": float(100 + i)})
        self._write_shadow_log(log, records)
        report = compute_sparsity(log)
        assert report.permit_rate == 0.20
        assert any("DRIFT" in a for a in report.alerts)

    def test_dead_alert_when_too_strict(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # 0.5% permit rate → below 1% floor
        records = []
        for i in range(199):
            records.append({"verdict": ABSTAIN, "symbol": "SOLUSDT", "regime_current": "TREND_UP", "ts": float(i)})
        records.append({"verdict": PERMIT_CANDIDATE, "symbol": "SOLUSDT", "regime_current": "TREND_UP",
                        "setup_class": REGIME_TRANSITION_BREAK, "ts": 200.0})
        self._write_shadow_log(log, records)
        report = compute_sparsity(log)
        assert report.permit_rate == pytest.approx(0.005, abs=0.001)
        assert any("DEAD" in a for a in report.alerts)

    def test_overlap_violation_detected(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # Two permits for same symbol at same timestamp with different classes
        records = [
            {"verdict": PERMIT_CANDIDATE, "symbol": "BTCUSDT", "ts": 100.0,
             "setup_class": REGIME_TRANSITION_BREAK, "regime_current": "TREND_UP"},
            {"verdict": PERMIT_CANDIDATE, "symbol": "BTCUSDT", "ts": 100.5,
             "setup_class": LIQUIDITY_VACUUM_RECLAIM, "regime_current": "TREND_UP"},
        ]
        self._write_shadow_log(log, records)
        report = compute_sparsity(log)
        assert len(report.overlap_violations) == 1
        assert any("OVERLAP" in a for a in report.alerts)

    def test_per_symbol_drift_alert(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # BTC: 3% (healthy), ETH: 25% (drift)
        records = []
        for i in range(97):
            records.append({"verdict": ABSTAIN, "symbol": "BTCUSDT", "regime_current": "TREND_UP", "ts": float(i)})
        for i in range(3):
            records.append({"verdict": PERMIT_CANDIDATE, "symbol": "BTCUSDT", "regime_current": "TREND_UP",
                            "setup_class": REGIME_TRANSITION_BREAK, "ts": float(100 + i)})
        for i in range(75):
            records.append({"verdict": ABSTAIN, "symbol": "ETHUSDT", "regime_current": "BREAKOUT", "ts": float(200 + i)})
        for i in range(25):
            records.append({"verdict": PERMIT_CANDIDATE, "symbol": "ETHUSDT", "regime_current": "BREAKOUT",
                            "setup_class": BREAKOUT_RETEST_CONFIRM, "ts": float(300 + i)})
        self._write_shadow_log(log, records)
        report = compute_sparsity(log)
        assert any("ETHUSDT" in a and "DRIFT" in a for a in report.alerts)
        assert not any("BTCUSDT" in a and "DRIFT" in a for a in report.alerts)


# ===================================================================
# TestNonOrdinalInvariant — prevent implicit scoring (safeguard §7)
# ===================================================================
class TestNonOrdinalInvariant:
    """
    Setup classes must be semantically distinct causal states,
    not quality tiers. No ordinal naming, no implicit ranking.
    """

    # Ordinal/quality patterns that would indicate disguised scoring
    ORDINAL_FRAGMENTS = [
        "strong", "weak", "primary", "secondary", "high_quality",
        "low_quality", "grade_a", "grade_b", "tier_1", "tier_2",
        "best", "worst", "better", "good", "bad", "premium",
    ]

    def test_setup_class_count_bounded(self):
        """Taxonomy must not grow unbounded."""
        assert len(_VALID_SETUP_CLASSES) <= _MAX_SETUP_CLASSES, (
            f"{len(_VALID_SETUP_CLASSES)} setup classes exceeds cap of {_MAX_SETUP_CLASSES}"
        )

    def test_no_ordinal_naming(self):
        """No setup class name may contain ordinal/quality fragments."""
        for cls in _VALID_SETUP_CLASSES:
            lower = cls.lower()
            for frag in self.ORDINAL_FRAGMENTS:
                assert frag not in lower, (
                    f"Setup class {cls!r} contains ordinal fragment {frag!r} — "
                    f"this indicates disguised scoring, not causal state"
                )

    def test_classes_are_semantically_distinct(self):
        """Each class must have a unique causal basis (no near-duplicates)."""
        # Heuristic: no two class names should share > 50% of their tokens
        classes = list(_VALID_SETUP_CLASSES)
        for i, a in enumerate(classes):
            tokens_a = set(a.lower().split("_"))
            for b in classes[i + 1:]:
                tokens_b = set(b.lower().split("_"))
                overlap = tokens_a & tokens_b
                union = tokens_a | tokens_b
                jaccard = len(overlap) / len(union) if union else 0
                assert jaccard < 0.5, (
                    f"Setup classes {a!r} and {b!r} share {jaccard:.0%} tokens — "
                    f"possible duplicate causal basis"
                )

    def test_fps_result_has_no_numeric_authority_field(self):
        """FPSResult must not contain any numeric score/rank/weight field."""
        import dataclasses
        numeric_names = {"score", "rank", "weight", "percentile", "band",
                         "confidence", "probability", "quality"}
        for f in dataclasses.fields(FPSResult):
            assert f.name not in numeric_names, (
                f"FPSResult has numeric authority field {f.name!r} — violates binary ontology"
            )

    def test_evaluate_never_returns_numeric_metadata(self):
        """The evaluate path must not smuggle numeric authority via gate_trace."""
        ctx = _make_ctx()
        result = evaluate(ctx)
        for entry in result.gate_trace:
            # gate_trace entries must be categorical (g1:CLASS, g2:DIR, g3:pass/deny)
            # not numeric (score=0.73, rank=2)
            assert "=" not in entry or not any(
                c.isdigit() for c in entry.split("=")[-1]
            ), f"gate_trace entry {entry!r} contains numeric value — potential implicit scoring"


# ===================================================================
# TestBurstDetection — temporal clustering (safeguard §7/final)
# ===================================================================
class TestBurstDetection:
    """Permits must be temporally sparse, not clustered in bursts."""

    def _write_shadow_log(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_empty_log(self, tmp_path):
        log = tmp_path / "empty.jsonl"
        report = compute_burst_risk(log)
        assert report.total_permits == 0
        assert "no_data" in report.alerts

    def test_evenly_distributed_no_bursts(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # 100 records spread over 100 hours, 5% permit, evenly spaced
        records = []
        base_ts = 1000000.0
        for i in range(100):
            ts = base_ts + i * 3600  # hourly
            verdict = PERMIT_CANDIDATE if i % 20 == 0 else ABSTAIN
            records.append({
                "verdict": verdict, "symbol": "BTCUSDT",
                "regime_current": "TREND_UP",
                "setup_class": REGIME_TRANSITION_BREAK if verdict == PERMIT_CANDIDATE else None,
                "ts": ts,
            })
        self._write_shadow_log(log, records)
        report = compute_burst_risk(log)
        assert not any("BURST" in a for a in report.alerts)

    def test_burst_detected_1h(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # 10 records in one hour: 8 permits (80% rate) → burst
        records = []
        base_ts = 1000000.0
        for i in range(10):
            ts = base_ts + i * 300  # every 5 min within 1 hour
            verdict = PERMIT_CANDIDATE if i < 8 else ABSTAIN
            records.append({
                "verdict": verdict, "symbol": "BTCUSDT",
                "regime_current": "TREND_UP",
                "setup_class": REGIME_TRANSITION_BREAK if verdict == PERMIT_CANDIDATE else None,
                "ts": ts,
            })
        self._write_shadow_log(log, records)
        report = compute_burst_risk(log)
        assert any("BURST_1H" in a for a in report.alerts)
        assert report.max_rate_1h >= 0.30

    def test_consecutive_streak_detected(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # 8 consecutive permits → exceeds default cap of 5
        records = []
        base_ts = 1000000.0
        for i in range(8):
            records.append({
                "verdict": PERMIT_CANDIDATE, "symbol": "BTCUSDT",
                "regime_current": "TREND_UP",
                "setup_class": REGIME_TRANSITION_BREAK,
                "ts": base_ts + i * 60,
            })
        # Then some abstains
        for i in range(10):
            records.append({
                "verdict": ABSTAIN, "symbol": "BTCUSDT",
                "regime_current": "TREND_UP", "ts": base_ts + 600 + i * 60,
            })
        self._write_shadow_log(log, records)
        report = compute_burst_risk(log)
        assert report.max_consecutive == 8
        assert len(report.consecutive_streaks) >= 1
        assert any("CONSECUTIVE" in a for a in report.alerts)

    def test_short_streak_no_alert(self, tmp_path):
        log = tmp_path / "shadow.jsonl"
        # 3 consecutive permits → below cap of 5, no alert
        records = []
        base_ts = 1000000.0
        for i in range(3):
            records.append({
                "verdict": PERMIT_CANDIDATE, "symbol": "BTCUSDT",
                "regime_current": "TREND_UP",
                "setup_class": REGIME_TRANSITION_BREAK,
                "ts": base_ts + i * 60,
            })
        for i in range(20):
            records.append({
                "verdict": ABSTAIN, "symbol": "BTCUSDT",
                "regime_current": "TREND_UP", "ts": base_ts + 300 + i * 60,
            })
        self._write_shadow_log(log, records)
        report = compute_burst_risk(log)
        assert report.max_consecutive == 3
        assert not any("CONSECUTIVE" in a for a in report.alerts)


# ===================================================================
# TestGate1Hypothesis — VEB, ERE, TCP detection (Step 10)
# ===================================================================
class TestGate1Hypothesis:
    """Gate 1 detection for hypothesis-driven setup classes."""

    # --- VOL_EXPANSION_BREAKOUT ---

    def test_veb_fires(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40,
            spread_bps=3.0,  # prevent LVR
            atr_percentile=15.0, range_ratio=2.0,
            local_breakout_dir="HIGH",
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == VOL_EXPANSION_BREAKOUT

    def test_veb_atr_percentile_too_high(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            atr_percentile=25.0, range_ratio=2.0, local_breakout_dir="HIGH",
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_veb_range_ratio_insufficient(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            atr_percentile=15.0, range_ratio=1.5, local_breakout_dir="HIGH",
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_veb_no_breakout_dir(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            atr_percentile=15.0, range_ratio=2.0, local_breakout_dir=None,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    # --- EXHAUSTION_REVERSAL ---

    def test_ere_fires(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            zscore=2.5, rsi=80.0, continuation_failed=True, wick_rejection=True,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == EXHAUSTION_REVERSAL

    def test_ere_zscore_insufficient(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            zscore=1.5, rsi=80.0, continuation_failed=True, wick_rejection=True,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_ere_rsi_not_extreme(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            zscore=2.5, rsi=60.0, continuation_failed=True, wick_rejection=True,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_ere_negative_zscore_low_rsi(self):
        ctx = _make_ctx(
            regime_previous="TREND_DOWN", regime_current="TREND_DOWN",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            zscore=-2.5, rsi=20.0, continuation_failed=True, wick_rejection=True,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == EXHAUSTION_REVERSAL

    # --- TREND_PULLBACK ---

    def test_tcp_fires(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            ema_aligned=True, pullback_atr_ratio=1.0,
            momentum_reacceleration=True, ema_slope=0.5,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == TREND_PULLBACK

    def test_tcp_pullback_too_deep(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            ema_aligned=True, pullback_atr_ratio=2.0,
            momentum_reacceleration=True, ema_slope=0.5,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_tcp_slope_regime_incoherent(self):
        """Negative slope with TREND_UP regime must not pass gate 1."""
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            ema_aligned=True, pullback_atr_ratio=1.0,
            momentum_reacceleration=True, ema_slope=-0.3,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) is None

    def test_tcp_trend_down(self):
        ctx = _make_ctx(
            regime_previous="TREND_DOWN", regime_current="TREND_DOWN",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            ema_aligned=True, pullback_atr_ratio=0.8,
            momentum_reacceleration=True, ema_slope=-0.4,
        )
        assert _gate1_structural_admissibility(ctx, FPSv1Config()) == TREND_PULLBACK


# ===================================================================
# TestGate2Hypothesis — direction validation for VEB, ERE, TCP (Step 11)
# ===================================================================
class TestGate2Hypothesis:
    """Gate 2 direction validation for hypothesis-driven setup classes."""

    def test_veb_high_long_pass(self):
        ctx = _make_ctx(local_breakout_dir="HIGH", proposed_direction="LONG")
        assert _gate2_direction_validity(ctx, VOL_EXPANSION_BREAKOUT) == LONG

    def test_veb_high_short_fail(self):
        ctx = _make_ctx(local_breakout_dir="HIGH", proposed_direction="SHORT")
        assert _gate2_direction_validity(ctx, VOL_EXPANSION_BREAKOUT) is None

    def test_veb_low_short_pass(self):
        ctx = _make_ctx(local_breakout_dir="LOW", proposed_direction="SHORT")
        assert _gate2_direction_validity(ctx, VOL_EXPANSION_BREAKOUT) == SHORT

    def test_ere_positive_zscore_short(self):
        ctx = _make_ctx(zscore=2.5, proposed_direction="SHORT")
        assert _gate2_direction_validity(ctx, EXHAUSTION_REVERSAL) == SHORT

    def test_ere_negative_zscore_long(self):
        ctx = _make_ctx(zscore=-2.5, proposed_direction="LONG")
        assert _gate2_direction_validity(ctx, EXHAUSTION_REVERSAL) == LONG

    def test_ere_wrong_reversal_direction(self):
        ctx = _make_ctx(zscore=2.5, proposed_direction="LONG")
        assert _gate2_direction_validity(ctx, EXHAUSTION_REVERSAL) is None

    def test_tcp_positive_slope_long(self):
        ctx = _make_ctx(ema_slope=0.5, proposed_direction="LONG")
        assert _gate2_direction_validity(ctx, TREND_PULLBACK) == LONG

    def test_tcp_negative_slope_short(self):
        ctx = _make_ctx(ema_slope=-0.5, proposed_direction="SHORT")
        assert _gate2_direction_validity(ctx, TREND_PULLBACK) == SHORT

    def test_tcp_slope_direction_mismatch(self):
        ctx = _make_ctx(ema_slope=0.5, proposed_direction="SHORT")
        assert _gate2_direction_validity(ctx, TREND_PULLBACK) is None


# ===================================================================
# TestHypothesisPipeline — full evaluate() for VEB, ERE, TCP (Step 12)
# ===================================================================
class TestHypothesisPipeline:
    """Full pipeline tests for hypothesis-driven setup classes."""

    def test_veb_full_pipeline_permit(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            atr_pct=0.03, atr_percentile=10.0, range_ratio=2.5,
            local_breakout_dir="HIGH", proposed_direction="LONG",
        )
        result = evaluate(ctx)
        assert result.verdict == PERMIT_CANDIDATE
        assert result.setup_class == VOL_EXPANSION_BREAKOUT
        assert result.direction == LONG

    def test_ere_full_pipeline_permit(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            atr_pct=0.03, zscore=2.5, rsi=80.0,
            continuation_failed=True, wick_rejection=True,
            proposed_direction="SHORT",
        )
        result = evaluate(ctx)
        assert result.verdict == PERMIT_CANDIDATE
        assert result.setup_class == EXHAUSTION_REVERSAL
        assert result.direction == SHORT

    def test_tcp_full_pipeline_permit(self):
        ctx = _make_ctx(
            regime_previous="TREND_UP", regime_current="TREND_UP",
            regime_age_bars=10, regime_confidence=0.40, spread_bps=3.0,
            atr_pct=0.03, ema_aligned=True, pullback_atr_ratio=1.0,
            momentum_reacceleration=True, ema_slope=0.5,
            proposed_direction="LONG",
        )
        result = evaluate(ctx)
        assert result.verdict == PERMIT_CANDIDATE
        assert result.setup_class == TREND_PULLBACK
        assert result.direction == LONG


# ===================================================================
# TestHypothesisNonOverlap — VEB/ERE/TCP vs regime-event classes (Step 13)
# ===================================================================
class TestHypothesisNonOverlap:
    """New hypothesis classes must not co-fire with structurally adjacent regime-event classes."""

    def test_veb_does_not_cofire_with_breakout_retest(self):
        """VEB requires atr_percentile<20 + range_ratio>1.8.
        BRC requires regime=BREAKOUT + age>=2 + conf>=0.60.
        A context satisfying BRC conditions should not also fire VEB
        (default atr_percentile=50 prevents it)."""
        ctx = _make_ctx(
            regime_current="BREAKOUT",
            regime_previous="BREAKOUT",
            regime_age_bars=3,
            regime_confidence=0.65,
            spread_bps=3.0,
            # BRC fires. VEB defaults: atr_percentile=50, range_ratio=1.0
        )
        result = _gate1_structural_admissibility(ctx, FPSv1Config())
        assert result == BREAKOUT_RETEST_CONFIRM
        # Now verify: even with VEB-like range_ratio, atr_percentile blocks it
        # because BRC fires first in priority order
        ctx2 = _make_ctx(
            regime_current="BREAKOUT",
            regime_previous="BREAKOUT",
            regime_age_bars=3,
            regime_confidence=0.65,
            spread_bps=3.0,
            atr_percentile=15.0, range_ratio=2.0, local_breakout_dir="HIGH",
        )
        result2 = _gate1_structural_admissibility(ctx2, FPSv1Config())
        # BRC fires first (priority order in gate 1)
        assert result2 == BREAKOUT_RETEST_CONFIRM

    def test_tcp_does_not_cofire_with_regime_transition_break(self):
        """RTB fires on fresh regime transition (age<=3, conf>=0.55, regime change).
        TCP fires on pullback in established trend (ema_aligned, slope-regime coherent).
        A fresh transition context should fire RTB, not TCP."""
        ctx = _make_ctx(
            regime_previous="CHOPPY",
            regime_current="TREND_UP",
            regime_age_bars=2,
            regime_confidence=0.60,
            spread_bps=3.0,
            # Also set TCP fields
            ema_aligned=True, pullback_atr_ratio=1.0,
            momentum_reacceleration=True, ema_slope=0.5,
        )
        result = _gate1_structural_admissibility(ctx, FPSv1Config())
        # RTB fires first (priority order): transition + young age + high conf
        assert result == REGIME_TRANSITION_BREAK

    def test_ere_does_not_cofire_with_dislocation_reversion(self):
        """DR fires on MEAN_REVERT + volume_z>=1.5 + conf>=0.50.
        ERE fires on |zscore|>=2 + extreme RSI + continuation_failed + wick_rejection.
        A context satisfying DR conditions should fire DR first."""
        ctx = _make_ctx(
            regime_current="MEAN_REVERT",
            regime_previous="MEAN_REVERT",
            regime_age_bars=10,
            regime_confidence=0.55,
            volume_z=2.0,
            spread_bps=3.0,
            # Also set ERE fields
            zscore=2.5, rsi=80.0, continuation_failed=True, wick_rejection=True,
        )
        result = _gate1_structural_admissibility(ctx, FPSv1Config())
        # DR fires first (priority order in gate 1)
        assert result == DISLOCATION_REVERSION


# ===================================================================
# TestEvaluateShadowForIntent — executor integration wrapper
# ===================================================================
class TestEvaluateShadowForIntent:
    """Tests for the evaluate_shadow_for_intent() executor integration."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self, monkeypatch):
        """Reset the module-level config cache before each test."""
        import execution.futures_permit_surface_v1 as _mod
        monkeypatch.setattr(_mod, "_FPS_CFG_CACHE", None)

    @pytest.fixture()
    def shadow_dir(self, tmp_path, monkeypatch):
        """Redirect shadow log to tmp_path so tests don't write to real logs."""
        import execution.futures_permit_surface_v1 as _mod
        log_path = tmp_path / "fps_shadow.jsonl"
        monkeypatch.setattr(_mod, "_SHADOW_LOG", log_path)
        return log_path

    @staticmethod
    def _intent(**overrides):
        base = {
            "symbol": "ETHUSDT",
            "price": 3200.0,
            "signal": "BUY",
        }
        base.update(overrides)
        return base

    @staticmethod
    def _sentinel(**overrides):
        base = {
            "primary_regime": "TREND_UP",
            "previous_regime": "CHOPPY",
            "regime_age_bars": 5,
            "regime_probs": {"TREND_UP": 0.70, "CHOPPY": 0.20},
            "crisis_flag": False,
        }
        base.update(overrides)
        return base

    def test_happy_path_writes_shadow_record(self, shadow_dir, tmp_path, monkeypatch):
        """Valid intent + sentinel produces a shadow JSONL line."""
        from execution.futures_permit_surface_v1 import evaluate_shadow_for_intent
        # Provide minimal config so it doesn't read real file
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        evaluate_shadow_for_intent(self._intent(), self._sentinel())
        assert shadow_dir.exists()
        lines = shadow_dir.read_text().strip().splitlines()
        assert len(lines) >= 1
        rec = json.loads(lines[0])
        assert rec["symbol"] == "ETHUSDT"
        assert rec["proposed_direction"] == "LONG"

    def test_missing_symbol_returns_silently(self, shadow_dir, monkeypatch):
        """No symbol → early return, no crash, no shadow line."""
        from execution.futures_permit_surface_v1 import evaluate_shadow_for_intent
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        evaluate_shadow_for_intent(self._intent(symbol=""), self._sentinel())
        assert not shadow_dir.exists()

    def test_missing_price_returns_silently(self, shadow_dir, monkeypatch):
        """price <= 0 → early return, no crash, no shadow line."""
        from execution.futures_permit_surface_v1 import evaluate_shadow_for_intent
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        evaluate_shadow_for_intent(self._intent(price=0), self._sentinel())
        assert not shadow_dir.exists()

    def test_signal_buy_maps_to_long(self, shadow_dir, monkeypatch):
        from execution.futures_permit_surface_v1 import evaluate_shadow_for_intent
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        evaluate_shadow_for_intent(self._intent(signal="BUY"), self._sentinel())
        rec = json.loads(shadow_dir.read_text().strip().splitlines()[0])
        assert rec["proposed_direction"] == "LONG"

    def test_signal_sell_maps_to_short(self, shadow_dir, monkeypatch):
        from execution.futures_permit_surface_v1 import evaluate_shadow_for_intent
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        evaluate_shadow_for_intent(self._intent(signal="SELL"), self._sentinel())
        rec = json.loads(shadow_dir.read_text().strip().splitlines()[0])
        assert rec["proposed_direction"] == "SHORT"

    def test_config_disabled_skips(self, shadow_dir, monkeypatch):
        """When config.enabled=False, no evaluation occurs."""
        from execution.futures_permit_surface_v1 import evaluate_shadow_for_intent
        cfg = FPSv1Config(enabled=False)
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            cfg,
        )
        evaluate_shadow_for_intent(self._intent(), self._sentinel())
        assert not shadow_dir.exists()

    def test_bad_sentinel_state_does_not_raise(self, shadow_dir, monkeypatch):
        """Completely broken sentinel state → fail-open, no crash."""
        from execution.futures_permit_surface_v1 import evaluate_shadow_for_intent
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        # Should not raise — fail-open
        evaluate_shadow_for_intent(self._intent(), {"garbage": True})

    def test_metadata_fields_propagate(self, shadow_dir, monkeypatch):
        """Metadata fields (atr_pct, volume_z, rsi) flow into context."""
        from execution.futures_permit_surface_v1 import evaluate_shadow_for_intent
        monkeypatch.setattr(
            "execution.futures_permit_surface_v1._FPS_CFG_CACHE",
            FPSv1Config(),
        )
        intent = self._intent(metadata={"atr_pct": 0.05, "volume_z": 3.0, "rsi": 75.0})
        evaluate_shadow_for_intent(intent, self._sentinel())
        rec = json.loads(shadow_dir.read_text().strip().splitlines()[0])
        assert rec["atr_pct"] == pytest.approx(0.05)
        assert rec["volume_z"] == pytest.approx(3.0)
        assert rec["rsi"] == pytest.approx(75.0)
