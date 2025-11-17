import json

import pytest

pytest.importorskip("pandas")

from dashboard import nav_helpers, live_helpers


def test_signal_attempts_summary_parses_latest():
    lines = [
        "[screener] attempted=5 emitted=1",
        "misc line",
        "[screener] attempted=7 emitted=3",
    ]
    msg = nav_helpers.signal_attempts_summary(lines)
    assert msg == "Signals: 7 attempted, 3 emitted (43%)"


def test_signal_attempts_summary_missing():
    lines: list[str] = ["no metrics here"]
    assert nav_helpers.signal_attempts_summary(lines) == "Signals: N/A"


def test_nav_snapshot_prefers_state(tmp_path, monkeypatch):
    state_path = tmp_path / "nav.json"
    payload = {"nav": 123.45, "equity": 130.0, "updated_ts": 999.0}
    state_path.write_text(json.dumps(payload))
    monkeypatch.setattr(live_helpers, "NAV_STATE_PATH", state_path)

    snapshot = live_helpers.get_nav_snapshot()
    assert snapshot["nav"] == pytest.approx(123.45)
    assert snapshot["equity"] == pytest.approx(130.0)
    assert snapshot["ts"] == pytest.approx(999.0)
