"""Tests for execution.hydra_monotonicity module."""
import json
import os
import pytest
from execution.hydra_monotonicity import (
    compute_monotonicity,
    compute_monotonicity_by_head,
    persist_snapshot,
)


def _make_episode(score, entry, exit_, side="LONG", regime="MEAN_REVERT"):
    return {
        "hybrid_score": score,
        "avg_entry_price": entry,
        "avg_exit_price": exit_,
        "side": side,
        "symbol": "BTCUSDT",
        "regime_at_entry": regime,
    }


class TestComputeMonotonicity:
    def test_insufficient_data(self):
        result = compute_monotonicity([])
        assert result["slope"] == "insufficient_data"
        assert result["spearman"] is None
        assert result["buckets"] == []

    def test_perfectly_monotonic(self):
        """Higher score → higher return should give positive Spearman."""
        eps = [
            _make_episode(0.3, 100, 99),   # -1%
            _make_episode(0.35, 100, 99.5), # -0.5%
            _make_episode(0.4, 100, 100),   # 0%
            _make_episode(0.45, 100, 100.5),# +0.5%
            _make_episode(0.5, 100, 101),   # +1%
            _make_episode(0.55, 100, 101.5),# +1.5%
            _make_episode(0.6, 100, 102),   # +2%
        ]
        result = compute_monotonicity(eps, n_buckets=3)
        assert result["spearman"] is not None
        assert result["spearman"] > 0.5
        assert result["slope"] == "upward"
        assert result["n"] == 7
        assert len(result["buckets"]) >= 2

    def test_inverted_model(self):
        """Higher score → lower return should give negative Spearman."""
        eps = [
            _make_episode(0.3, 100, 102),  # +2%
            _make_episode(0.35, 100, 101.5),
            _make_episode(0.4, 100, 101),
            _make_episode(0.45, 100, 100.5),
            _make_episode(0.5, 100, 100),
            _make_episode(0.55, 100, 99.5),
            _make_episode(0.6, 100, 99),   # -1%
        ]
        result = compute_monotonicity(eps, n_buckets=3)
        assert result["spearman"] is not None
        assert result["spearman"] < -0.05
        assert result["slope"] == "inverted"

    def test_zero_scores_excluded(self):
        """Episodes with hybrid_score=0 should be filtered out."""
        eps = [
            _make_episode(0.0, 100, 101),  # excluded
            _make_episode(0.0, 100, 102),  # excluded
        ]
        result = compute_monotonicity(eps)
        assert result["n"] == 0
        assert result["slope"] == "insufficient_data"

    def test_short_side_return(self):
        """Short trades: entry > exit = positive return."""
        eps = [
            _make_episode(0.3, 100, 102, "SHORT"),  # -2%
            _make_episode(0.35, 100, 101, "SHORT"),  # -1%
            _make_episode(0.4, 100, 100, "SHORT"),   # 0%
            _make_episode(0.45, 100, 99, "SHORT"),   # +1%
            _make_episode(0.5, 100, 98, "SHORT"),    # +2%
        ]
        result = compute_monotonicity(eps, n_buckets=3)
        assert result["spearman"] is not None
        assert result["spearman"] > 0.5

    def test_bucket_structure(self):
        """Each bucket should have range, mean_score, mean_return, n."""
        eps = [_make_episode(0.3 + i * 0.03, 100, 100 + i * 0.1) for i in range(10)]
        result = compute_monotonicity(eps, n_buckets=5)
        for b in result["buckets"]:
            assert "range" in b
            assert "mean_score" in b
            assert "mean_return" in b
            assert "n" in b
            assert b["n"] >= 1


class TestPersistSnapshot:
    def test_writes_state_file(self, tmp_path):
        dest = str(tmp_path / "hydra_monotonicity.json")
        eps = [_make_episode(0.3 + i * 0.03, 100, 100 + i * 0.1) for i in range(10)]
        snap = persist_snapshot(eps, path=dest)
        assert os.path.exists(dest)
        data = json.loads(open(dest).read())
        assert data["n"] == 10
        assert "spearman" in data
        assert "buckets" in data
        assert "per_head" in data
        assert "head_contamination" in data


class TestPerHeadMonotonicity:
    def test_groups_by_regime(self):
        """Trades split by regime_at_entry into separate heads."""
        eps = [
            _make_episode(0.3 + i * 0.02, 100, 100 + i * 0.1, regime="MEAN_REVERT")
            for i in range(7)
        ] + [
            _make_episode(0.3 + i * 0.02, 100, 100 + i * 0.1, regime="TREND_UP")
            for i in range(6)
        ]
        results = compute_monotonicity_by_head(eps, n_buckets=2)
        heads = {r["head"] for r in results}
        assert "MEAN_REVERT" in heads
        assert "TREND" in heads

    def test_unknown_regime_excluded(self):
        """Episodes with unknown regime are excluded from per-head."""
        eps = [
            _make_episode(0.3 + i * 0.02, 100, 100 + i * 0.1, regime="unknown")
            for i in range(10)
        ]
        results = compute_monotonicity_by_head(eps)
        assert results == []

    def test_head_contamination_detected(self, tmp_path):
        """Contamination flag set when head upward but global flat."""
        # MR trades: monotonically positive (high score → high return)
        mr_eps = [
            _make_episode(0.30 + i * 0.02, 100, 100 + i * 0.4, regime="MEAN_REVERT")
            for i in range(8)
        ]
        # TREND trades: inverted (high score → low return), same score range
        # Enough inverted trades to wash out global monotonicity
        trend_eps = [
            _make_episode(0.30 + i * 0.02, 100, 102 - i * 0.4, regime="TREND_UP")
            for i in range(8)
        ]
        dest = str(tmp_path / "mono.json")
        snap = persist_snapshot(mr_eps + trend_eps, path=dest)
        per_head = {h["head"]: h for h in snap.get("per_head", [])}
        assert "MEAN_REVERT" in per_head
        assert "TREND" in per_head
        # MR head should be strongly upward
        assert per_head["MEAN_REVERT"]["slope"] == "upward"
        # Global should be flat or inverted (washed out by mixing)
        assert snap["slope"] in ("flat", "inverted")
        assert snap["head_contamination"] is True

    def test_trend_up_and_down_merge(self):
        """TREND_UP and TREND_DOWN both map to TREND head."""
        eps = [
            _make_episode(0.3 + i * 0.02, 100, 100 + i * 0.1, regime="TREND_UP")
            for i in range(3)
        ] + [
            _make_episode(0.4 + i * 0.02, 100, 100 + i * 0.1, regime="TREND_DOWN")
            for i in range(3)
        ]
        results = compute_monotonicity_by_head(eps, n_buckets=2)
        heads = [r["head"] for r in results]
        assert heads == ["TREND"]  # single entry, both merged
