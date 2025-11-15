import pytest

pytest.importorskip("pandas")

from dashboard import nav_helpers


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
