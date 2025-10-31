from __future__ import annotations

import pytest

from execution import signal_doctor


def test_doctor_recent_fail_veto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signal_doctor, "is_nav_fresh", lambda threshold: True)
    verdict = signal_doctor.evaluate_signal(
        "BUY",
        "BTCUSDT",
        {"recent_fail_streak": 5, "fail_streak_limit": 3, "confidence": 1.0},
    )
    assert verdict["ok"] is False
    assert "recent_fail_streak" in verdict["reasons"]
    assert verdict["confidence"] <= 0.2


def test_doctor_nav_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signal_doctor, "is_nav_fresh", lambda threshold: False)
    verdict = signal_doctor.evaluate_signal(
        "SELL",
        "ETHUSDT",
        {"nav_stale_threshold_s": 60, "confidence": 0.8},
    )
    assert verdict["ok"] is False
    assert "nav_stale" in verdict["reasons"]


def test_doctor_spread_and_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signal_doctor, "is_nav_fresh", lambda threshold: True)
    verdict = signal_doctor.evaluate_signal(
        "BUY",
        "SOLUSDT",
        {
            "spread_bps": 10.0,
            "max_spread_bps": 5.0,
            "latency_ms": 20_000,
            "max_latency_ms": 10_000,
            "confidence": 0.7,
        },
    )
    assert verdict["ok"] is False
    assert set(verdict["reasons"]) == {"wide_spread", "stale_signal"}
