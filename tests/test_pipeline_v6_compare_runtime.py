from __future__ import annotations

from execution import executor_live
from execution.intel import pipeline_v6_compare


def test_pipeline_compare_scheduler(monkeypatch):
    calls = []

    def fake_compare():
        calls.append(1)

    monkeypatch.setattr(pipeline_v6_compare, "compare_pipeline_v6", fake_compare)
    monkeypatch.setattr(executor_live, "_PIPELINE_V6_COMPARE_INTERVAL_S", 0.0, raising=False)
    monkeypatch.setattr(executor_live, "_LAST_PIPELINE_V6_COMPARE", 0.0, raising=False)
    executor_live._maybe_run_pipeline_v6_compare()
    assert len(calls) == 1

    monkeypatch.setattr(executor_live.time, "time", lambda: 100.0)
    monkeypatch.setattr(executor_live, "_PIPELINE_V6_COMPARE_INTERVAL_S", 60.0, raising=False)
    monkeypatch.setattr(executor_live, "_LAST_PIPELINE_V6_COMPARE", 100.0, raising=False)
    executor_live._maybe_run_pipeline_v6_compare()
    assert len(calls) == 1
