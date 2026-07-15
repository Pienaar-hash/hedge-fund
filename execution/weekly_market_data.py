from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping


UTC = timezone.utc
MIN_COMPLETED_WEEKS = 53
PREFERRED_COMPLETED_WEEKS = 105


def _coerce_utc_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError(f"unsupported timestamp type: {type(value)!r}")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _week_start_utc(value: datetime) -> datetime:
    return (value - timedelta(days=value.weekday())).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


def _canonical_completed_week_end(as_of: datetime | None) -> datetime:
    reference = _coerce_utc_datetime(as_of) if as_of is not None else datetime.now(UTC)
    return _week_start_utc(reference)


def _iter_price_points(history: Iterable[Mapping[str, Any]]) -> list[tuple[datetime, float]]:
    points: list[tuple[datetime, float]] = []
    for row in history:
        ts = _coerce_utc_datetime(row["ts"])
        close = float(row["close"])
        points.append((ts, close))
    points.sort(key=lambda item: item[0])
    return points


def load_completed_weekly_closes(
    price_history_by_symbol: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    as_of: datetime | str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    completed_week_end = _canonical_completed_week_end(
        _coerce_utc_datetime(as_of) if as_of is not None else None
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for symbol in sorted(price_history_by_symbol):
        weekly_last: dict[datetime, tuple[datetime, float]] = {}
        for ts, close in _iter_price_points(price_history_by_symbol[symbol]):
            if not math.isfinite(close) or close <= 0:
                raise ValueError(f"{symbol}: non-positive or non-finite close")
            if ts >= completed_week_end:
                continue
            week_end = _week_start_utc(ts) + timedelta(days=7)
            prev = weekly_last.get(week_end)
            if prev is None or ts > prev[0]:
                weekly_last[week_end] = (ts, close)
        result[symbol] = [
            {"week_end": week_end.isoformat(), "close": close}
            for week_end, (_, close) in sorted(weekly_last.items())
        ]
    return result


def validate_weekly_history(
    weekly_closes_by_symbol: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    minimum_completed_weeks: int = MIN_COMPLETED_WEEKS,
    preferred_completed_weeks: int = PREFERRED_COMPLETED_WEEKS,
) -> dict[str, list[dict[str, Any]]]:
    validated: dict[str, list[dict[str, Any]]] = {}
    for symbol in sorted(weekly_closes_by_symbol):
        rows = list(weekly_closes_by_symbol[symbol])
        if len(rows) < minimum_completed_weeks:
            continue
        closes = [float(row["close"]) for row in rows]
        if any((not math.isfinite(close)) or close <= 0 for close in closes):
            continue
        if len(rows) > preferred_completed_weeks:
            rows = rows[-preferred_completed_weeks:]
        validated[symbol] = rows
    return validated


def build_weekly_input_manifest(
    weekly_closes_by_symbol: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    price_source: str,
    completed_week_end: str,
) -> dict[str, Any]:
    canonical_symbols = {
        symbol: list(weekly_closes_by_symbol[symbol])
        for symbol in sorted(weekly_closes_by_symbol)
    }
    payload = {
        "completed_week_end": completed_week_end,
        "price_source": price_source,
        "symbols": canonical_symbols,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "completed_week_end": completed_week_end,
        "price_source": price_source,
        "symbols": canonical_symbols,
        "input_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }
