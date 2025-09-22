from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import joblib

from execution.ml.data import load_candles
from execution.ml.features import build_feature_frame


def _load_registry(path: Path) -> Dict:
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}


def score_symbol(cfg: Dict, symbol: str) -> Dict:
    ml_cfg = cfg.get("ml", {})
    if not ml_cfg.get("enabled", False):
        return {"enabled": False}

    registry_path = Path(ml_cfg.get("registry_path", "models/registry.json"))
    registry = _load_registry(registry_path)
    meta = registry.get(symbol)
    if not meta:
        return {"enabled": True, "error": "no_model"}

    model_blob = joblib.load(meta["model_path"])
    model = model_blob["model"]
    feature_names = model_blob["features"]

    candles = load_candles(symbol, ml_cfg.get("timeframe", "1h"), ml_cfg.get("lookback_bars", 2000))
    feats = build_feature_frame(candles, cfg).tail(1)
    if feats.empty:
        return {"enabled": True, "error": "no_features"}
    feats = feats[feature_names]
    prob = float(model.predict_proba(feats.values)[:, 1][0])

    return {
        "enabled": True,
        "symbol": symbol,
        "p": prob,
        "model": meta.get("model"),
        "auc": meta.get("auc"),
        "trained_at": meta.get("trained_at"),
        "ts": int(time.time()),
    }


def score_all(cfg: Dict) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for sym in cfg.get("ml", {}).get("symbols", []):
        try:
            results[sym] = score_symbol(cfg, sym)
        except Exception as exc:
            results[sym] = {"error": str(exc)}
    return results


__all__ = ["score_symbol", "score_all"]
