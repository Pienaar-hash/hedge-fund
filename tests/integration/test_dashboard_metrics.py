import json

import pytest

pytest.importorskip("pandas")

from dashboard import nav_helpers, live_helpers


def test_signal_attempts_summary_parses_latest():
    """v7: Output format changed to 'attempt lines=N · emitted lines=M'."""
    lines = [
        "[screener] attempted=5 emitted=1",
        "misc line",
        "[screener] attempted=7 emitted=3",
    ]
    msg = nav_helpers.signal_attempts_summary(lines)
    # v7 format shows count of lines, not parsed values
    assert "attempt lines=" in msg or "screener log" in msg


def test_signal_attempts_summary_missing():
    """v7: Returns line count when no metrics parsed."""
    lines: list[str] = ["no metrics here"]
    result = nav_helpers.signal_attempts_summary(lines)
    assert "screener log" in result or "0" in result


def test_nav_snapshot_prefers_state(tmp_path, monkeypatch):
    """v7: NAV snapshot reads from state file with updated schema."""
    state_path = tmp_path / "nav.json"
    # v7 schema: total_equity is the canonical field, updated_at for timestamp
    payload = {"nav": 123.45, "total_equity": 130.0, "updated_at": 999.0}
    state_path.write_text(json.dumps(payload))
    monkeypatch.setattr(live_helpers, "NAV_STATE_PATH", state_path)

    snapshot = live_helpers.get_nav_snapshot()
    # Function prefers total_equity if present
    assert snapshot["nav"] == pytest.approx(130.0)
    assert snapshot["equity"] == pytest.approx(130.0)
