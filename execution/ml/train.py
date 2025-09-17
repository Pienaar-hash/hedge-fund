from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from execution.ml.data import load_candles
from execution.ml.features import build_feature_frame, make_labels


def _ts() -> int:
    return int(time.time())


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _train_logreg(X: np.ndarray, y: np.ndarray, cfg: Dict) -> LogisticRegression:
    params = {
        "max_iter": 200,
        "n_jobs": None,
        "class_weight": cfg.get("ml", {})
        .get("trainer", {})
        .get("class_weight", "balanced"),
    }
    model = LogisticRegression(**{k: v for k, v in params.items() if v is not None})
    model.fit(X, y)
    return model


def train_symbol(cfg: Dict, symbol: str) -> Dict:
    ml_cfg = cfg.get("ml", {})
    interval = ml_cfg.get("timeframe", "1h")
    lookback = int(ml_cfg.get("lookback_bars", 2000))
    horizon = int(ml_cfg.get("horizon_bars", 4))
    target_bps = float(ml_cfg.get("target_ret_bps", 20))

    candles = load_candles(symbol, interval, lookback + horizon + 20)
    feats = build_feature_frame(candles, cfg)
    labels = make_labels(candles["close"], horizon, target_bps).reindex(feats.index).dropna()
    feats = feats.loc[labels.index]
    if feats.empty:
        raise ValueError("insufficient features for training")

    X = feats.values
    y = labels.values.astype(int)
    test_size = float(ml_cfg.get("trainer", {}).get("test_size", 0.2))
    n_test = max(16, int(len(y) * test_size))
    if len(y) <= n_test:
        raise ValueError("not enough samples for holdout")
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    model = _train_logreg(X_train, y_train, cfg)
    prob_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob_test)) if len(set(y_test)) > 1 else float("nan")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{symbol}_logreg.pkl"
    joblib.dump({"model": model, "features": list(feats.columns)}, model_path)

    meta = {
        "symbol": symbol,
        "timeframe": interval,
        "lookback": lookback,
        "horizon": horizon,
        "target_bps": target_bps,
        "model": "logreg",
        "auc": auc,
        "trained_at": _ts(),
        "model_path": str(model_path),
    }

    registry_path = Path(ml_cfg.get("registry_path", "models/registry.json"))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        registry = json.load(open(registry_path, "r", encoding="utf-8"))
    except Exception:
        registry = {}
    registry[symbol] = meta
    with open(registry_path, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)
    return meta


def train_all(cfg: Dict) -> List[Dict]:
    ml_cfg = cfg.get("ml", {})
    registry_path = Path(ml_cfg.get("registry_path", "models/registry.json"))
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists():
        with open(registry_path, "w", encoding="utf-8") as handle:
            json.dump({}, handle)

    metas: List[Dict] = []
    for sym in ml_cfg.get("symbols", []):
        try:
            metas.append(train_symbol(cfg, sym))
        except Exception as exc:
            metas.append({"symbol": sym, "error": str(exc)})
    return metas


__all__ = ["train_symbol", "train_all"]
