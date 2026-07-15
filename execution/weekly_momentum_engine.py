from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Mapping


ELIGIBILITY_PERSISTENCE_FLOOR = 0.6153846154


def _percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted((value, symbol) for symbol, value in values.items())
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0][1]: 1.0}
    grouped: dict[float, list[int]] = defaultdict(list)
    for index, (value, _) in enumerate(ordered):
        grouped[value].append(index)
    result: dict[str, float] = {}
    for value, indices in grouped.items():
        pct = (max(indices) + 1) / len(ordered)
        for symbol_value, symbol in ordered:
            if symbol_value == value:
                result[symbol] = pct
    return result


def compute_weekly_features(
    weekly_closes_by_symbol: Mapping[str, list[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = {}
    roc_4w: dict[str, float] = {}
    roc_13w: dict[str, float] = {}
    roc_26w: dict[str, float] = {}
    roc_52w: dict[str, float] = {}

    for symbol in sorted(weekly_closes_by_symbol):
        rows = list(weekly_closes_by_symbol[symbol])
        closes = [float(row["close"]) for row in rows]
        close_t = closes[-1]
        weekly_returns = [
            (closes[index] / closes[index - 1]) - 1.0
            for index in range(1, len(closes))
        ]
        entry = {
            "symbol": symbol,
            "roc_4w": close_t / closes[-5] - 1.0,
            "roc_13w": close_t / closes[-14] - 1.0,
            "roc_26w": close_t / closes[-27] - 1.0,
            "roc_52w": close_t / closes[-53] - 1.0,
            "positive_weeks_13": sum(1 for value in weekly_returns[-13:] if value > 0),
            "sma_40w": mean(closes[-40:]),
            "completed_week_end": rows[-1]["week_end"],
        }
        entry["persistence_13"] = entry["positive_weeks_13"] / 13.0
        entry["above_trend"] = close_t > entry["sma_40w"]
        raw[symbol] = entry
        roc_4w[symbol] = entry["roc_4w"]
        roc_13w[symbol] = entry["roc_13w"]
        roc_26w[symbol] = entry["roc_26w"]
        roc_52w[symbol] = entry["roc_52w"]

    rank_4w = _percentile_ranks(roc_4w)
    rank_13w = _percentile_ranks(roc_13w)
    rank_26w = _percentile_ranks(roc_26w)
    rank_52w = _percentile_ranks(roc_52w)

    features: list[dict[str, Any]] = []
    for symbol in sorted(raw):
        entry = raw[symbol]
        entry["rank_4w"] = rank_4w[symbol]
        entry["rank_13w"] = rank_13w[symbol]
        entry["rank_26w"] = rank_26w[symbol]
        entry["rank_52w"] = rank_52w[symbol]
        entry["momentum_score"] = (
            0.15 * entry["rank_4w"]
            + 0.35 * entry["rank_13w"]
            + 0.30 * entry["rank_26w"]
            + 0.20 * entry["rank_52w"]
        )
        entry["eligible"] = bool(
            entry["roc_13w"] > 0
            and entry["roc_26w"] > 0
            and entry["roc_52w"] > 0
            and entry["persistence_13"] >= ELIGIBILITY_PERSISTENCE_FLOOR
            and entry["above_trend"]
        )
        entry["conviction"] = "REJECT"
        entry["selected"] = False
        entry["rejection_reasons"] = []
        features.append(entry)
    return features


def rank_weekly_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        features,
        key=lambda item: (
            -item["momentum_score"],
            -item["rank_26w"],
            -item["rank_52w"],
            -item["persistence_13"],
            item["symbol"],
        ),
    )


def classify_conviction(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    for feature in features:
        item = dict(feature)
        if (
            item["eligible"]
            and item["momentum_score"] >= 0.80
            and item["rank_13w"] >= 0.75
            and item["rank_26w"] >= 0.80
            and item["rank_52w"] >= 0.70
            and item["persistence_13"] >= 0.69
        ):
            item["conviction"] = "HIGH"
        elif item["eligible"] and item["momentum_score"] >= 0.70:
            item["conviction"] = "MEDIUM"
        else:
            item["conviction"] = "REJECT"
        classified.append(item)
    return classified


def select_candidates(
    ranked_features: list[dict[str, Any]],
    correlation_groups: Mapping[str, str],
    *,
    max_positions: int = 3,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for feature in ranked_features:
        item = dict(feature)
        group = correlation_groups.get(item["symbol"], item["symbol"])
        if item["conviction"] != "HIGH":
            item["rejection_reasons"] = sorted(set(item["rejection_reasons"] + ["conviction"]))
            continue
        if group in seen_groups:
            item["rejection_reasons"] = sorted(set(item["rejection_reasons"] + ["correlation_group"]))
            continue
        item["selected"] = True
        selected.append(item)
        seen_groups.add(group)
        if len(selected) >= max_positions:
            break
    return selected
