"""Tests for execution/hydra_funnel.py — Hydra intent pipeline visibility tracking."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from execution.hydra_funnel import HydraFunnel, get_funnel, record, snapshot, flush


class TestHydraFunnel:
    def test_empty_snapshot(self):
        f = HydraFunnel()
        snap = f.snapshot()
        assert snap["visibility_rate"] == 0.0
        assert snap["stages"]["generated"] == 0
        assert snap["stages"]["executed"] == 0
        assert snap["regime_visibility"] == {}

    def test_record_and_snapshot(self):
        f = HydraFunnel()
        f.record("generated", 10, regime="TREND_UP")
        f.record("post_merge", 8, regime="TREND_UP")
        f.record("submitted", 5, regime="TREND_UP")
        f.record("post_doctrine", 4, regime="TREND_UP")
        f.record("executed", 3, regime="TREND_UP")

        snap = f.snapshot()
        assert snap["stages"]["generated"] == 10
        assert snap["stages"]["post_merge"] == 8
        assert snap["stages"]["submitted"] == 5
        assert snap["stages"]["post_doctrine"] == 4
        assert snap["stages"]["executed"] == 3
        assert snap["visibility_rate"] == 0.3  # 3/10

    def test_regime_visibility(self):
        f = HydraFunnel()
        f.record("generated", 10, regime="TREND_UP")
        f.record("executed", 5, regime="TREND_UP")
        f.record("generated", 20, regime="MEAN_REVERT")
        f.record("executed", 2, regime="MEAN_REVERT")

        snap = f.snapshot()
        assert snap["visibility_rate"] == pytest.approx(7 / 30, abs=0.001)
        assert snap["regime_visibility"]["TREND_UP"]["visibility_rate"] == 0.5
        assert snap["regime_visibility"]["MEAN_REVERT"]["visibility_rate"] == 0.1

    def test_multiple_records_accumulate(self):
        f = HydraFunnel()
        f.record("generated", 5, regime="TREND_UP")
        f.record("generated", 3, regime="TREND_UP")
        f.record("executed", 2, regime="TREND_UP")

        snap = f.snapshot()
        assert snap["stages"]["generated"] == 8
        assert snap["stages"]["executed"] == 2
        assert snap["regime_visibility"]["TREND_UP"]["stages"]["generated"] == 8

    def test_zero_count_ignored(self):
        f = HydraFunnel()
        f.record("generated", 0)
        snap = f.snapshot()
        assert snap["stages"]["generated"] == 0

    def test_no_regime_still_counts(self):
        f = HydraFunnel()
        f.record("generated", 5)
        f.record("executed", 2)
        snap = f.snapshot()
        assert snap["visibility_rate"] == 0.4
        assert snap["regime_visibility"] == {}

    def test_flush_writes_json(self, tmp_path: Path):
        f = HydraFunnel()
        f.record("generated", 10, regime="CHOPPY")
        f.record("executed", 1, regime="CHOPPY")

        out = tmp_path / "hydra_funnel.json"
        f.flush(out)

        data = json.loads(out.read_text())
        assert data["stages"]["generated"] == 10
        assert data["stages"]["executed"] == 1
        assert data["visibility_rate"] == 0.1
        assert "CHOPPY" in data["regime_visibility"]
        assert "updated_ts" in data

    def test_module_singleton(self):
        """get_funnel() returns the same singleton."""
        f1 = get_funnel()
        f2 = get_funnel()
        assert f1 is f2

    def test_snapshot_json_serializable(self):
        f = HydraFunnel()
        f.record("generated", 5, regime="BREAKOUT")
        f.record("post_doctrine", 3, regime="BREAKOUT")
        snap = f.snapshot()
        # Must not raise
        json.dumps(snap)

    def test_merge_win_rate(self):
        f = HydraFunnel()
        f.record("generated", 20, regime="TREND_UP")
        f.record("post_merge", 12, regime="TREND_UP")
        f.record("executed", 5, regime="TREND_UP")

        snap = f.snapshot()
        assert snap["merge_win_rate"] == pytest.approx(12 / 20, abs=0.001)
        assert snap["regime_visibility"]["TREND_UP"]["merge_win_rate"] == pytest.approx(
            12 / 20, abs=0.001
        )

    def test_merge_win_rate_empty(self):
        f = HydraFunnel()
        snap = f.snapshot()
        assert snap["merge_win_rate"] == 0.0
