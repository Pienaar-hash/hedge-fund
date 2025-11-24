"""
v5.9 Execution Hardening — Signal gates
- ATR-based volatility filter & inverse-vol size scaling
- Rolling expectancy veto (per symbol)
"""

from __future__ import annotations

import importlib
import math
import logging
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Optional
from datetime import datetime

from execution.utils import load_json
from execution.utils.metrics import is_in_asset_universe
from execution.utils.vol import atr_pct
from execution.utils.execution_health import size_multiplier
from execution.position_sizing import inverse_vol_size, volatility_regime_scale
from execution.intel.symbol_score import symbol_size_factor
from .utils.expectancy import rolling_expectancy

_LOG = logging.getLogger("signal_generator")
_STRATEGY_MODULES = ("momentum", "relative_value")
_SEEN_KEYS: "OrderedDict[tuple[str, str, str, str], None]" = OrderedDict()
_LRU_CAPACITY = 1000
_REGISTRY_PATH = Path("config/strategy_registry.json")
_DEFAULT_REG_ENTRY: Dict[str, Any] = {
    "enabled": True,
    "sandbox": False,
    "max_concurrent": 10,
    "confidence": 1.0,
    "capacity_usd": float("inf"),
}

try:
    from execution.ml.predict import predict_live as _predict_live
except Exception:  # pragma: no cover - optional dependency
    _predict_live = None  # type: ignore[assignment]

_ML_REFRESH_INTERVAL = 180.0
_ML_EXECUTOR: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=1) if _predict_live else None
_ML_FUTURE: Optional[Future] = None
_ML_CACHE_TS: float = 0.0
_ML_SCORES: Dict[str, float] = {}

ATR_LOOKBACK = 50
ATR_MULT = 1.5


def allow_trade(symbol: str) -> bool:
    if not is_in_asset_universe(symbol):
        return False
    atr = atr_pct(symbol, lookback_bars=ATR_LOOKBACK)
    med = atr_pct(symbol, lookback_bars=ATR_LOOKBACK * 10, median_only=True)
    if atr is not None and med is not None and atr > ATR_MULT * med:
        return False
    expectancy = rolling_expectancy(symbol)
    return (expectancy is None) or (expectancy >= 0.0)


def size_for(symbol: str, base_size: float) -> float:
    size = inverse_vol_size(symbol, base_size, lookback=ATR_LOOKBACK)
    size *= volatility_regime_scale(symbol)
    size *= size_multiplier(symbol)
    try:
        intel = symbol_size_factor(symbol)
        factor = float(intel.get("size_factor") or 1.0)
    except Exception:
        factor = 1.0
    size *= factor
    try:
        if base_size > 0:
            scale = size / float(base_size)
            if scale > 1_000.0:
                _LOG.warning(
                    "[size_guard] %s base=%.6f scaled=%.6f factor=%.2f",
                    symbol,
                    base_size,
                    size,
                    scale,
                )
                print(f'[size_dbg] symbol={symbol} base={base_size:.6f} scaled={size:.6f} factor={scale:.2f}')
    except Exception:
        _LOG.debug("size_for debug log failed", exc_info=True)
    return size


def _load_registry() -> Dict[str, Dict[str, Any]]:
    payload = {}
    try:
        data = load_json(str(_REGISTRY_PATH)) or {}
        raw = data.get("strategies") if isinstance(data, Mapping) and "strategies" in data else data
        if isinstance(raw, Mapping):
            payload = {
                str(k): (dict(v) if isinstance(v, Mapping) else {})
                for k, v in raw.items()
            }
    except Exception:
        payload = {}
    # Normalize entries with defaults
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, entry in payload.items():
        normalized_entry = dict(_DEFAULT_REG_ENTRY)
        normalized_entry.update(
            {k: v for k, v in entry.items() if v is not None}
        )
        # Ensure types and sane limits
        normalized_entry["enabled"] = bool(normalized_entry.get("enabled", True))
        normalized_entry["sandbox"] = bool(normalized_entry.get("sandbox", False))
        normalized_entry["max_concurrent"] = max(
            0, int(normalized_entry.get("max_concurrent", 0))
        )
        try:
            normalized_entry["confidence"] = float(normalized_entry.get("confidence", 1.0))
        except Exception:
            normalized_entry["confidence"] = 1.0
        normalized_entry["confidence"] = max(0.0, min(1.5, normalized_entry["confidence"]))
        try:
            cap = float(normalized_entry.get("capacity_usd", float("inf")))
        except Exception:
            cap = float("inf")
        normalized_entry["capacity_usd"] = cap if cap > 0 else float("inf")
        normalized[key] = normalized_entry
    return normalized


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


def _update_ml_predictions(now: float, cfg: Mapping[str, Any]) -> None:
    global _ML_FUTURE, _ML_CACHE_TS, _ML_SCORES

    if _predict_live is None or _ML_EXECUTOR is None:
        return
    ml_cfg = cfg.get("ml") if isinstance(cfg, Mapping) else None
    if not isinstance(ml_cfg, Mapping) or not ml_cfg.get("enabled", False):
        return

    if _ML_FUTURE is not None and _ML_FUTURE.done():
        try:
            result = _ML_FUTURE.result()
        except Exception as exc:
            _LOG.debug("ml future failed: %s", exc)
            _ML_FUTURE = None
        else:
            if isinstance(result, Mapping):
                predictions = result.get("predictions", [])
                scores: Dict[str, float] = {}
                if isinstance(predictions, list):
                    for item in predictions:
                        if not isinstance(item, Mapping):
                            continue
                        sym = str(item.get("symbol") or "").upper()
                        score = item.get("score")
                        try:
                            value = float(score)
                        except Exception:
                            continue
                        if sym:
                            scores[sym] = value
                _ML_SCORES = scores
                _ML_CACHE_TS = now
            _ML_FUTURE = None

    if (_ML_CACHE_TS and (now - _ML_CACHE_TS) < _ML_REFRESH_INTERVAL) or _ML_FUTURE is not None:
        return

    try:
        _ML_FUTURE = _ML_EXECUTOR.submit(_predict_live, dict(cfg))
    except Exception as exc:
        _LOG.debug("ml predict submit failed: %s", exc)
        _ML_FUTURE = None


def generate_intents(now: float, universe: Sequence[str] | None = None, cfg: Mapping[str, Any] | None = None) -> List[Mapping[str, Any]]:
    """Load strategy modules and emit deduplicated intents."""
    universe = universe or []
    if cfg is None:
        cfg = load_json("config/strategy_config.json") or {}

    _update_ml_predictions(now, cfg)

    registry = _load_registry()
    emitted: List[Mapping[str, Any]] = []
    per_strategy_counts: Dict[str, int] = defaultdict(int)
    per_strategy_gross: Dict[str, float] = defaultdict(float)

    for name in _STRATEGY_MODULES:
        reg_entry = registry.get(name, dict(_DEFAULT_REG_ENTRY))
        if not reg_entry.get("enabled", True) or reg_entry.get("sandbox", False):
            _LOG.info("strategy %s disabled%s", name, " (sandbox)" if reg_entry.get("sandbox") else "")
            continue
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

        max_concurrent = reg_entry.get("max_concurrent") or _DEFAULT_REG_ENTRY["max_concurrent"]
        capacity_usd = reg_entry.get("capacity_usd") or _DEFAULT_REG_ENTRY["capacity_usd"]

        for intent in candidates:
            if not isinstance(intent, Mapping):
                continue
            try:
                normalized = normalize_intent(intent)
            except Exception as exc:
                _LOG.debug("intent normalization failed for %s: %s", name, exc)
                continue
            strategy_key = (
                str(
                    normalized.get("strategy")
                    or normalized.get("strategy_name")
                    or normalized.get("strategyId")
                    or name
                )
            )
            entry = registry.get(strategy_key, reg_entry)
            if not entry.get("enabled", True) or entry.get("sandbox", False):
                continue
            max_allowed = entry.get("max_concurrent") or max_concurrent
            if max_allowed and per_strategy_counts[strategy_key] >= max_allowed:
                continue

            # Determine gross sizing for capacity gating
            gross_val = _safe_float(normalized.get("gross_usd"))
            if gross_val <= 0:
                gross_val = 0.0
            cap_usd = entry.get("capacity_usd") or capacity_usd
            if (
                math.isfinite(cap_usd)
                and cap_usd > 0
                and (per_strategy_gross[strategy_key] + gross_val) > cap_usd
            ):
                continue

            confidence = entry.get("confidence", 1.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 1.0
            confidence = max(0.0, min(1.5, confidence))
            normalized["strategy"] = strategy_key
            normalized["confidence"] = confidence
            normalized["signal_strength"] = confidence
            normalized["expected_edge"] = confidence - 0.5
            normalized.setdefault("metadata", {})
            if isinstance(normalized["metadata"], Mapping):
                meta = dict(normalized["metadata"])
                meta.update(
                    {
                        "registry_confidence": confidence,
                        "registry_capacity": cap_usd,
                        "registry_max_concurrent": max_allowed,
                    }
                )
                normalized["metadata"] = meta
                ml_score = _ML_SCORES.get(normalized["symbol"])
                if ml_score is not None:
                    meta["ml_score"] = float(ml_score)
                    normalized["metadata"] = meta
                    normalized["ml_score"] = float(ml_score)
            if not isinstance(normalized.get("params"), Mapping):
                normalized["params"] = dict(normalized.get("params") or {})
            else:
                normalized["params"] = dict(normalized["params"])
            if "symbol" in normalized:
                normalized["symbol"] = str(normalized["symbol"]).upper()
            normalized["generated_at"] = float(
                normalized.get("generated_at") or time.time()
            )
            key = _intent_key(normalized)
            if key is None:
                continue
            if not _register_key(key):
                continue
            per_strategy_counts[strategy_key] += 1
            per_strategy_gross[strategy_key] += gross_val
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
    normalized["capital_per_trade"] = _safe_float(normalized.get("capital_per_trade"))
    pct_raw = normalized.get("per_trade_nav_pct") or normalized.get("nav_pct")
    pct_val = _safe_float(pct_raw)
    if pct_val > 1.0:
        pct_val = min(pct_val / 100.0, 1.0)
    if pct_val < 0.0:
        pct_val = 0.0
    normalized["per_trade_nav_pct"] = pct_val
    lev_value = (
        normalized.get("leverage")
        or normalized.get("lev")
        or normalized.get("leverage_target")
    )
    lev_float = _safe_float(lev_value)
    normalized["leverage"] = lev_float if lev_float > 0 else 1.0

    gross_value = normalized.get("gross_usd")
    if gross_value in (None, ""):
        normalized["gross_usd"] = 0.0
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
