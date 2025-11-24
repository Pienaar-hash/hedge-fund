from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

LOG = logging.getLogger("ml.telemetry")

TELEMETRY_CACHE_PATH = Path("logs/cache/ml_telemetry.json")
MAX_HISTORY = 400

__all__ = [
    "TelemetryPoint",
    "record_confidence",
    "load_history",
    "aggregate_history",
    "latest_point",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_confidence(value: object, *, default: float = 0.5) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if numeric != numeric or numeric < 0:
        return default
    if numeric > 1:
        return min(1.0, numeric)
    return numeric


def _normalize_features(features: Mapping[str, object] | Sequence[Tuple[str, object]] | None) -> Dict[str, float]:
    if features is None:
        return {}
    items: Iterable[Tuple[str, object]]
    if isinstance(features, Mapping):
        items = features.items()
    else:
        items = list(features)
    norm: Dict[str, float] = {}
    for key, value in items:
        try:
            norm[str(key)] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return norm


def _read_cache(path: Path = TELEMETRY_CACHE_PATH) -> List[MutableMapping[str, object]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [dict(entry) for entry in payload if isinstance(entry, Mapping)]
    except Exception as exc:
        LOG.debug("[ml.telemetry] cache_read_failed path=%s err=%s", path, exc)
    return []


def _write_cache(entries: Sequence[Mapping[str, object]], *, path: Path = TELEMETRY_CACHE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as exc:
        LOG.warning("[ml.telemetry] cache_write_failed path=%s err=%s", path, exc)


@dataclass(slots=True)
class TelemetryPoint:
    ts: datetime
    confidence: float
    model: str
    features: Dict[str, float]
    metadata: Dict[str, object]

    def iso(self) -> str:
        return self.ts.isoformat()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TelemetryPoint":
        ts = payload.get("ts")
        if isinstance(ts, str):
            try:
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                dt = datetime.fromisoformat(ts)
            except Exception:
                dt = _utc_now()
        elif isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        elif isinstance(ts, datetime):
            dt = ts.astimezone(timezone.utc)
        else:
            dt = _utc_now()
        return cls(
            ts=dt,
            confidence=_coerce_confidence(payload.get("confidence")),
            model=str(payload.get("model") or "default"),
            features=_normalize_features(payload.get("features")),
            metadata={k: v for k, v in payload.items() if k not in {"ts", "confidence", "model", "features"}},
        )

    def to_mapping(self) -> Dict[str, object]:
        return {
            "ts": self.iso(),
            "confidence": float(self.confidence),
            "model": self.model,
            "features": dict(self.features),
            **self.metadata,
        }


def record_confidence(
    confidence: float,
    *,
    features: Mapping[str, object] | Sequence[Tuple[str, object]] | None = None,
    model: str = "default",
    metadata: Optional[Mapping[str, object]] = None,
    path: Path = TELEMETRY_CACHE_PATH,
) -> TelemetryPoint:
    """Append a telemetry point to the rolling cache."""
    entry = TelemetryPoint(
        ts=_utc_now(),
        confidence=_coerce_confidence(confidence),
        model=str(model),
        features=_normalize_features(features),
        metadata=dict(metadata) if metadata else {},
    )
    history = _read_cache(path)
    history.append(entry.to_mapping())
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    _write_cache(history, path=path)
    LOG.debug("[ml.telemetry] recorded point model=%s conf=%.3f", model, entry.confidence)
    return entry


def load_history(*, limit: int = 200, path: Path = TELEMETRY_CACHE_PATH) -> List[TelemetryPoint]:
    """Return most recent telemetry points (newest last)."""
    history = _read_cache(path)
    if limit > 0 and len(history) > limit:
        history = history[-limit:]
    return [TelemetryPoint.from_mapping(entry) for entry in history]


def latest_point(*, path: Path = TELEMETRY_CACHE_PATH) -> Optional[TelemetryPoint]:
    history = load_history(limit=1, path=path)
    if not history:
        return None
    return history[-1]


def _top_features(history: Iterable[TelemetryPoint], top_n: int = 5) -> List[Tuple[str, float]]:
    counter: Counter[str] = Counter()
    for point in history:
        for feature, value in point.features.items():
            counter[feature] += abs(float(value))
    return counter.most_common(top_n)


def aggregate_history(history: Sequence[TelemetryPoint]) -> Dict[str, object]:
    if not history:
        return {
            "count": 0,
            "avg_confidence": None,
            "latest_confidence": None,
            "top_features": [],
        }
    confidences = [point.confidence for point in history]
    avg_conf = sum(confidences) / len(confidences) if confidences else None
    latest = history[-1].confidence
    return {
        "count": len(history),
        "avg_confidence": avg_conf,
        "latest_confidence": latest,
        "top_features": _top_features(history),
    }

