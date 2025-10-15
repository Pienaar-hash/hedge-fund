from __future__ import annotations

import importlib
import logging
from collections import OrderedDict
from typing import Any, Iterable, List, Mapping, Sequence

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
            key = _intent_key(intent)
            if key is None:
                continue
            if not _register_key(key):
                continue
            emitted.append(dict(intent))

    return emitted


def generate_signals(
    now: float,
    universe: Sequence[str] | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> List[Mapping[str, Any]]:
    return generate_intents(now, universe, cfg)


__all__ = ["generate_intents", "generate_signals"]
