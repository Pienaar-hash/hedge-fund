from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Mapping

import joblib

from execution.ml.data import load_candles
from execution.ml.features_live import prepare_live_features

ML_CACHE_PATH = Path("logs/cache/ml_predictions.json")


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
    feats = prepare_live_features(candles, cfg)
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


def predict_live(cfg: Mapping[str, Any], cache_path: Path | None = None) -> Dict[str, Any]:
    cache_target = cache_path or ML_CACHE_PATH
    scores = score_all(dict(cfg))
    predictions = []
    for symbol, payload in scores.items():
        if not isinstance(payload, Mapping):
            continue
        if payload.get("error"):
            continue
        score = payload.get("p")
        if score is None:
            continue
        predictions.append(
            {
                "symbol": symbol,
                "score": float(score),
                "model": payload.get("model"),
                "auc": payload.get("auc"),
                "trained_at": payload.get("trained_at"),
                "updated_at": payload.get("ts"),
            }
        )
    predictions.sort(key=lambda item: item.get("score") or 0.0, reverse=True)
    payload = {
        "generated_at": int(time.time()),
        "predictions": predictions,
    }
    try:
        cache_target.parent.mkdir(parents=True, exist_ok=True)
        with cache_target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass
    return payload


def _load_cfg(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Live ML predictor")
    parser.add_argument("--config", default="config/strategy_config.json", help="Strategy config path")
    parser.add_argument("--cache", default=str(ML_CACHE_PATH), help="Output cache path")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path)
    payload = predict_live(cfg, cache_path=Path(args.cache))
    print(json.dumps(payload, indent=2, sort_keys=True))


__all__ = ["score_symbol", "score_all", "predict_live"]


if __name__ == "__main__":
    _main()
