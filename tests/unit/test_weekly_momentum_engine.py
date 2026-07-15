from __future__ import annotations

from datetime import datetime, timedelta, timezone

from execution.weekly_momentum_engine import (
    classify_conviction,
    compute_weekly_features,
    rank_weekly_features,
    select_candidates,
)


def _history(start: float, step: float, weeks: int = 60):
    start_dt = datetime(2025, 1, 6, tzinfo=timezone.utc)
    rows = []
    value = start
    for week in range(weeks):
        rows.append({"week_end": (start_dt + timedelta(weeks=week)).isoformat(), "close": round(value, 4)})
        value += step
    return rows


def test_tied_scores_resolve_by_symbol():
    features = [
        {"symbol": "ETHUSDT", "momentum_score": 0.9, "rank_26w": 1.0, "rank_52w": 1.0, "persistence_13": 1.0},
        {"symbol": "BTCUSDT", "momentum_score": 0.9, "rank_26w": 1.0, "rank_52w": 1.0, "persistence_13": 1.0},
    ]
    ranked = rank_weekly_features(features)
    assert [item["symbol"] for item in ranked] == ["BTCUSDT", "ETHUSDT"]


def test_high_candidates_selected_with_group_limit():
    history = {
        "BTCUSDT": _history(100.0, 4.0),
        "ETHUSDT": _history(100.0, 4.0),
        "SOLUSDT": _history(100.0, 4.0),
        "LINKUSDT": _history(100.0, 4.0),
        "DOGEUSDT": _history(100.0, 1.0),
    }
    features = classify_conviction(rank_weekly_features(compute_weekly_features(history)))
    for item in features:
        item["correlation_group"] = "majors" if item["symbol"] in {"BTCUSDT", "ETHUSDT"} else item["symbol"]
    selected = select_candidates(
        features,
        {item["symbol"]: item["correlation_group"] for item in features},
    )
    assert len(selected) == 3
    assert sum(1 for item in selected if item["correlation_group"] == "majors") == 1


def test_medium_candidates_not_selected():
    history = {
        "BTCUSDT": _history(100.0, 4.0),
        "ETHUSDT": _history(100.0, 3.0),
        "SOLUSDT": _history(100.0, 2.0),
        "DOGEUSDT": _history(100.0, 1.0),
    }
    features = classify_conviction(rank_weekly_features(compute_weekly_features(history)))
    assert any(item["conviction"] == "MEDIUM" for item in features)
    selected = select_candidates(features, {item["symbol"]: item["symbol"] for item in features})
    assert all(item["conviction"] == "HIGH" for item in selected)
