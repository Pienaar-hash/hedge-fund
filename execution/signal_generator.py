from __future__ import annotations

import importlib
import logging
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from datetime import datetime

from execution.utils import load_json

_LOG = logging.getLogger("signal_generator")
_STRATEGY_MODULES = ("momentum", "relative_value")
_SEEN_KEYS: "OrderedDict[tuple[str, str, str, str], None]" = OrderedDict()
_LRU_CAPACITY = 1000


def _intent_key(intent: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    symbol = str(intent.get("symbol") or intent.get("pair") or "").upper()
    if not symbol:
        return None
    tf = str(
        intent.get("timeframe")
        or intent.get("tf")
        or intent.get("interval")
        or ""
    ).lower()
    side = str(intent.get("signal") or intent.get("side") or "").upper()
    candle_close = intent.get("candle_close") or intent.get("timestamp") or intent.get("t")
    if candle_close is None:
        candle_repr = ""
    else:
        try:
            candle_repr = f"{float(candle_close):.4f}"
        except (TypeError, ValueError):
            candle_repr = str(candle_close)
    return (symbol, tf, side, candle_repr)


def _register_key(key: tuple[str, str, str, str]) -> bool:
    if key in _SEEN_KEYS:
        _SEEN_KEYS.move_to_end(key)
        return False
    _SEEN_KEYS[key] = None
    if len(_SEEN_KEYS) > _LRU_CAPACITY:
        _SEEN_KEYS.popitem(last=False)
    return True


def _module_intents(mod: Any, now: float, universe: Sequence[str], cfg: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    gen_fn = getattr(mod, "generate_signals", None)
    if callable(gen_fn):
        try:
            return gen_fn(now=now, universe=universe, config=cfg)
        except TypeError:
            return gen_fn(now, universe, cfg)
    if hasattr(mod, "StrategyImpl"):
        try:
            strat = mod.StrategyImpl()
        except Exception as exc:
            _LOG.debug("StrategyImpl init failed for %s: %s", mod.__name__, exc)
            return []
        prepare = getattr(strat, "prepare", None)
        if callable(prepare):
            try:
                prepare({"now": now, "universe": universe, "config": cfg})
            except Exception as exc:
                _LOG.debug("prepare failed for %s: %s", mod.__name__, exc)
        signals_fn = getattr(strat, "signals", None)
        if callable(signals_fn):
            try:
                return signals_fn(now) or []
            except Exception as exc:
                _LOG.debug("signals failed for %s: %s", mod.__name__, exc)
                return []
    return []


def generate_intents(now: float, universe: Sequence[str] | None = None, cfg: Mapping[str, Any] | None = None) -> List[Mapping[str, Any]]:
    """Load strategy modules and emit deduplicated intents."""
    universe = universe or []
    if cfg is None:
        cfg = load_json("config/strategy_config.json") or {}

    emitted: List[Mapping[str, Any]] = []
    for name in _STRATEGY_MODULES:
        try:
            mod = importlib.import_module(f"strategies.{name}")
        except ModuleNotFoundError:
            continue
        except Exception as exc:
            _LOG.error("strategy import failed %s: %s", name, exc)
            continue

        try:
            candidates = list(_module_intents(mod, now, universe, cfg))
        except Exception as exc:
            _LOG.error("strategy %s generate failed: %s", name, exc)
            continue

        for intent in candidates:
            if not isinstance(intent, Mapping):
                continue
            try:
                normalized = normalize_intent(intent)
            except Exception as exc:
                _LOG.debug("intent normalization failed for %s: %s", name, exc)
                continue
            key = _intent_key(normalized)
            if key is None:
                continue
            if not _register_key(key):
                continue
            emitted.append(normalized)

    return emitted


def generate_signals(
    now: float,
    universe: Sequence[str] | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> List[Mapping[str, Any]]:
    return generate_intents(now, universe, cfg)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_timestamp(raw: Any) -> str:
    if raw in (None, ""):
        return datetime.utcnow().isoformat()
    if isinstance(raw, (int, float)):
        try:
            return datetime.utcfromtimestamp(float(raw)).isoformat()
        except Exception:
            return datetime.utcnow().isoformat()
    return str(raw)


def normalize_intent(intent: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = dict(intent)
    symbol = normalized.get("symbol") or normalized.get("pair")
    symbol_str = str(symbol or "").upper()
    if not symbol_str:
        raise ValueError("intent missing symbol")
    normalized["symbol"] = symbol_str

    normalized["timestamp"] = _normalize_timestamp(
        normalized.get("timestamp") or normalized.get("ts") or normalized.get("time")
    )

    tf = (
        normalized.get("timeframe")
        or normalized.get("tf")
        or normalized.get("interval")
        or normalized.get("time_frame")
    )
    tf_str = str(tf or "unknown").lower()
    normalized["timeframe"] = tf_str
    normalized["tf"] = tf_str

    signal = normalized.get("signal") or normalized.get("side")
    normalized["signal"] = str(signal or "").upper()

    normalize_reduce = normalized.get("reduceOnly", normalized.get("reduce_only"))
    normalized["reduceOnly"] = bool(normalize_reduce)

    normalized["price"] = _safe_float(normalized.get("price"))
    normalized["capital_per_trade"] = _safe_float(
        normalized.get("capital_per_trade") or normalized.get("capital") or normalized.get("notional")
    )
    lev_value = (
        normalized.get("leverage")
        or normalized.get("lev")
        or normalized.get("leverage_target")
    )
    lev_float = _safe_float(lev_value)
    normalized["leverage"] = lev_float if lev_float > 0 else 1.0

    gross_value = normalized.get("gross_usd")
    if gross_value in (None, ""):
        normalized["gross_usd"] = abs(normalized["capital_per_trade"]) * max(normalized["leverage"], 1.0)
    else:
        normalized["gross_usd"] = abs(_safe_float(gross_value))

    pos_side = normalized.get("positionSide") or normalized.get("posSide") or normalized.get("side")
    normalized["positionSide"] = str(pos_side or "")

    veto_raw = normalized.get("veto")
    if isinstance(veto_raw, str):
        veto_list = [veto_raw]
    elif isinstance(veto_raw, Mapping):
        veto_list = [str(item) for item in veto_raw.values() if item]
    elif isinstance(veto_raw, Iterable) and not isinstance(veto_raw, (str, bytes)):
        veto_list = [str(item) for item in veto_raw if item]
    elif veto_raw:
        veto_list = [str(veto_raw)]
    else:
        veto_list = []
    normalized["veto"] = veto_list

    return normalized


__all__ = ["generate_intents", "generate_signals", "normalize_intent"]
