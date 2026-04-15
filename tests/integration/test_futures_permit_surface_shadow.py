"""Integration tests for FPS v1 shadow telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from execution.futures_permit_surface_v1 import (
    PERMIT_CANDIDATE,
    FPSEvalContext,
    FPSv1Config,
    append_shadow_record,
    build_dle_shadow_context,
    build_selector_mirror,
    evaluate,
    load_fps_config,
)


def _make_ctx(**overrides) -> FPSEvalContext:
    defaults = dict(
        symbol="ETHUSDT",
        timestamp=1712966400.0,
        regime_current="TREND_UP",
        regime_previous="CHOPPY",
        regime_age_bars=2,
        regime_confidence=0.70,
        crisis_flag=False,
        atr_pct=0.03,
        volume_z=2.0,
        spread_bps=1.0,
        price=3200.0,
        proposed_direction="LONG",
    )
    defaults.update(overrides)
    return FPSEvalContext(**defaults)


# ===================================================================
# Shadow JSONL append
# ===================================================================
class TestShadowAppend:
    def test_appends_valid_jsonl(self, tmp_path: Path):
        log_path = tmp_path / "fps_shadow.jsonl"
        ctx = _make_ctx()
        cfg = FPSv1Config()
        result = evaluate(ctx, cfg)
        append_shadow_record(result, ctx, cfg, log_path=log_path)

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["schema"] == "fps_shadow_v1"
        assert record["symbol"] == "ETHUSDT"
        assert record["verdict"] == result.verdict
        assert record["config_authority"] == "none"
        assert record["config_mode"] == "shadow_only"

    def test_multiple_appends(self, tmp_path: Path):
        log_path = tmp_path / "fps_shadow.jsonl"
        ctx = _make_ctx()
        cfg = FPSv1Config()
        result = evaluate(ctx, cfg)
        for _ in range(3):
            append_shadow_record(result, ctx, cfg, log_path=log_path)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_fail_open_bad_path(self):
        """Should not raise even with unwritable path."""
        ctx = _make_ctx()
        cfg = FPSv1Config()
        result = evaluate(ctx, cfg)
        # /dev/null/impossible is not writable — should fail silently
        append_shadow_record(result, ctx, cfg, log_path=Path("/dev/null/impossible/x.jsonl"))


# ===================================================================
# Selector mirror
# ===================================================================
class TestSelectorMirror:
    def test_keys_namespaced(self):
        ctx = _make_ctx()
        result = evaluate(ctx)
        mirror = build_selector_mirror(result, ctx)
        for key in mirror:
            assert key.startswith("fps_v1_"), f"Key {key!r} not namespaced"

    def test_contains_verdict(self):
        ctx = _make_ctx()
        result = evaluate(ctx)
        mirror = build_selector_mirror(result, ctx)
        assert mirror["fps_v1_verdict"] == result.verdict


# ===================================================================
# DLE shadow context
# ===================================================================
class TestDLEShadowContext:
    def test_contains_required_fields(self):
        ctx = _make_ctx()
        result = evaluate(ctx)
        dle_ctx = build_dle_shadow_context(result, ctx)
        assert "fps_v1_verdict" in dle_ctx
        assert "fps_v1_symbol" in dle_ctx
        assert "fps_v1_snapshot_hash" in dle_ctx
        assert "fps_v1_fee_cleared" in dle_ctx

    def test_fee_cleared_true_on_permit(self):
        ctx = _make_ctx(atr_pct=0.03)  # high ATR → fee clears
        result = evaluate(ctx)
        dle_ctx = build_dle_shadow_context(result, ctx)
        if result.verdict == PERMIT_CANDIDATE:
            assert dle_ctx["fps_v1_fee_cleared"] is True


# ===================================================================
# Freeze guard — config always shadow_only, no authority promotion
# ===================================================================
class TestFreezeGuard:
    def test_config_locked_shadow_only(self):
        cfg = load_fps_config({"mode": "live", "authority": "full"})
        assert cfg.mode == "shadow_only"
        assert cfg.authority == "none"

    def test_config_frozen_dataclass(self):
        cfg = FPSv1Config()
        with pytest.raises(AttributeError):
            cfg.authority = "full"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.mode = "live"  # type: ignore[misc]

    def test_influence_flags_locked(self):
        cfg = load_fps_config({
            "can_influence_sizing": True,
            "can_influence_entry": True,
            "can_influence_exit": True,
            "can_influence_routing": True,
        })
        assert cfg.can_influence_sizing is False
        assert cfg.can_influence_entry is False
        assert cfg.can_influence_exit is False
        assert cfg.can_influence_routing is False


# ===================================================================
# Import isolation
# ===================================================================
class TestImportIsolation:
    """FPS v1 must not pull in execution authority modules."""

    FORBIDDEN = [
        "order_router",
        "doctrine_kernel",
        "sizing",
        "order_dispatch",
        "executor_live",
    ]

    def test_no_forbidden_in_source(self):
        """FPS v1 source must not import forbidden authority modules."""
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

        for forbidden in self.FORBIDDEN:
            for imp in imported_names:
                assert forbidden not in imp, (
                    f"FPS v1 source imports forbidden module: {imp}"
                )
