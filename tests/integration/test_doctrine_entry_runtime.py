from __future__ import annotations

import time

from execution.doctrine_bridge import DoctrineEntryInputs


def _sentinel_state() -> dict[str, object]:
    return {
        "primary_regime": "TREND_UP",
        "regime": "TREND_UP",
        "confidence": 0.82,
        "smoothed_probs": {"TREND_UP": 0.82},
        "cycles_stable": 5,
        "updated_ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }


def test_doctrine_gate_vetoes_entries_when_inputs_degraded(monkeypatch) -> None:
    from execution import executor_live

    monkeypatch.setattr(executor_live, "_DOCTRINE_AVAILABLE", True)
    monkeypatch.setattr(executor_live, "_DETERMINISM_AVAILABLE", False)
    monkeypatch.setattr(executor_live, "_load_sentinel_x_state", _sentinel_state)
    monkeypatch.setattr(executor_live, "_load_execution_quality_state", lambda: {})
    monkeypatch.setattr(
        executor_live,
        "resolve_doctrine_entry_inputs",
        lambda **_: DoctrineEntryInputs(
            portfolio=None,
            alpha_health=None,
            telemetry={"healthy": False},
            degraded_details={
                "code": "HYDRA_STATE_STALE",
                "reason": "Hydra state stale",
                "dependency": "hydra",
            },
        ),
    )

    allowed, reason, details = executor_live._doctrine_gate(
        {
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "gross_usd": 100.0,
            "metadata": {"strategy": "TREND"},
        }
    )

    assert allowed is False
    assert reason == "ENVIRONMENT_DEGRADED"
    assert details["code"] == "HYDRA_STATE_STALE"


def test_doctrine_gate_reduce_only_bypasses_degraded_entry_inputs(monkeypatch) -> None:
    from execution import executor_live

    monkeypatch.setattr(executor_live, "_DOCTRINE_AVAILABLE", True)
    monkeypatch.setattr(executor_live, "resolve_doctrine_entry_inputs", lambda **_: (_ for _ in ()).throw(AssertionError("should not be called")))

    allowed, reason, details = executor_live._doctrine_gate(
        {
            "symbol": "BTCUSDT",
            "signal": "SELL",
            "reduceOnly": True,
            "metadata": {"strategy": "TREND"},
        }
    )

    assert allowed is True
    assert reason == "REDUCE_ONLY_BYPASS"
    assert details == {}
