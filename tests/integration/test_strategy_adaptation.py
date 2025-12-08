import pytest

from execution.strategy_adaptation import (
    adaptive_factor,
    adaptive_sizing,
    attach_adaptive_metadata,
    load_regime_snapshot,
    load_risk_snapshot,
    strategy_enablement,
)


def test_atr_only_shrink():
    assert adaptive_factor(2, 0, "OK") == pytest.approx(0.7)


def test_dd_only_shrink():
    assert adaptive_factor(0, 2, "OK") == pytest.approx(0.6)


def test_risk_mode_shrink():
    assert adaptive_factor(0, 0, "WARN") == pytest.approx(0.8)


def test_combined_shrink():
    expected = 0.5 * 0.6 * 0.5
    assert adaptive_factor(3, 2, "DEFENSIVE") == pytest.approx(expected)


def test_strategy_enablement_respects_risk_and_dd():
    assert not strategy_enablement("test", 0, 0, "HALTED")
    assert not strategy_enablement("test", 0, 3, "OK")
    assert strategy_enablement("test", 0, 2, "OK")


def test_metadata_includes_final_factor():
    intent: dict[str, object] = {}
    _, final_factor = adaptive_sizing("BTCUSDT", 1000, 0, 0, "OK")
    attach_adaptive_metadata(intent, 0, 0, "OK", final_factor)
    assert intent["metadata"]["adaptive"]["final_factor"] == pytest.approx(final_factor)


def test_unlevered_qty_uses_adapted_gross():
    base_gross = 1_000.0
    price = 10.0
    adapted_gross, factor = adaptive_sizing("BTCUSDT", base_gross, 3, 0, "OK")
    qty = adapted_gross / price
    assert qty == pytest.approx((base_gross * factor) / price)


def test_missing_regimes_defaults(tmp_path):
    path = tmp_path / "regimes.json"
    assert not path.exists()
    regime_state = load_regime_snapshot(path)
    assert regime_state["atr_regime"] == 0
    assert regime_state["dd_regime"] == 0
    assert adaptive_factor(regime_state["atr_regime"], regime_state["dd_regime"], "OK") == pytest.approx(1.0)


def test_missing_risk_defaults(tmp_path):
    path = tmp_path / "risk_snapshot.json"
    assert not path.exists()
    risk_state = load_risk_snapshot(path)
    assert risk_state["risk_mode"] == "OK"
    assert adaptive_factor(0, 0, risk_state["risk_mode"]) == pytest.approx(1.0)
