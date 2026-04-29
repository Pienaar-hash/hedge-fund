"""
Regression guard: conviction authority must remain FROZEN.

The conviction surface was economically falsified on 2026-04-12.
These tests prevent silent re-enablement without explicit decision.
See config/CONVICTION_AUTHORITY_FROZEN.md for full rationale.
"""
import json
from pathlib import Path

import pytest

STRATEGY_CONFIG = Path("config/strategy_config.json")


@pytest.fixture()
def strategy_config():
    return json.loads(STRATEGY_CONFIG.read_text())


@pytest.mark.unit
class TestConvictionAuthorityFrozen:
    """Guard rails: conviction must not gate futures entries."""

    def test_conviction_enabled_is_false(self, strategy_config):
        conv = strategy_config.get("conviction", {})
        assert conv.get("enabled") is False, (
            "conviction.enabled must be false — conviction surface economically "
            "falsified (2026-04-12). Do not re-enable without a new validated surface."
        )

    def test_conviction_mode_is_off(self, strategy_config):
        conv = strategy_config.get("conviction", {})
        assert str(conv.get("mode", "off")).lower() == "off", (
            "conviction.mode must be 'off' — conviction band gate must not fire. "
            "See config/CONVICTION_AUTHORITY_FROZEN.md."
        )

    def test_alpha_decay_conviction_disabled(self, strategy_config):
        ad = strategy_config.get("alpha_decay", {})
        assert ad.get("conviction_enabled") is False, (
            "alpha_decay.conviction_enabled must be false."
        )

    def test_filter_below_threshold_not_enabled(self, strategy_config):
        """Hybrid score threshold filtering must not be active."""
        hs = strategy_config.get("hybrid_scoring", {})
        ir = hs.get("intent_ranking", {})
        assert ir.get("filter_below_threshold", False) is False, (
            "filter_below_threshold must remain False — hybrid scores are "
            "telemetry only, not entry gates."
        )

    def test_authority_frozen_marker_exists(self, strategy_config):
        conv = strategy_config.get("conviction", {})
        marker = conv.get("_authority_frozen", "")
        assert "frozen" in marker.lower() or "falsified" in marker.lower(), (
            "conviction._authority_frozen marker must exist to prevent silent re-enablement."
        )
