#!/usr/bin/env python3
"""
Offline evaluation of signal generation comparing ML probability gate vs rule-based signals.
Outputs models/signal_eval.json summarising per-symbol metrics and aggregate averages.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from execution.ml.data import load_candles
from execution.ml.features import build_feature_frame, make_labels
from execution.utils import load_json


def _rule_signal(close: pd.Series, sig_cfg: dict) -> pd.Series:
    mom = sig_cfg.get("momentum", {})
    ema_fast = close.ewm(span=mom.get("ema_fast", 20), adjust=False).mean()
    ema_slow = close.ewm(span=mom.get("ema_slow", 50), adjust=False).mean()
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / max(1, mom.get("rsi_len", 14)), adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / max(1, mom.get("rsi_len", 14)), adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return ((ema_fast > ema_slow) & (rsi >= mom.get("rsi_buy", 55))).astype(int)


def _metrics(y_true: np.ndarray, signal: np.ndarray, proba: np.ndarray | None = None) -> dict:
    if len(y_true) == 0:
        return {"auc": float("nan"), "precision": 0.0, "recall": 0.0, "f1": 0.0, "hit_rate": 0.0, "coverage": 0.0}
    try:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, signal, average="binary", zero_division=0
        )
    except Exception:
        precision = recall = f1 = 0.0
    hit = float((signal * y_true).sum() / max(1, signal.sum()))
    coverage = float(signal.mean())
    if proba is not None and len(set(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, proba))
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")
    return {
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "hit_rate": hit,
        "coverage": coverage,
    }


def evaluate_symbol(cfg: dict, symbol: str) -> dict:
    ml_cfg = cfg.get("ml", {})
    interval = ml_cfg.get("timeframe", "1h")
    lookback = int(ml_cfg.get("lookback_bars", 2000))
    horizon = int(ml_cfg.get("horizon_bars", 4))
    target_bps = float(ml_cfg.get("target_ret_bps", 20.0))
    prob_threshold = float(ml_cfg.get("prob_threshold", 0.5))

    try:
        candles = load_candles(symbol, interval, lookback + horizon + 20)
    except Exception as exc:
        return {"symbol": symbol, "error": f"load_candles_failed: {exc}"}

    try:
        feats = build_feature_frame(candles, cfg)
    except Exception as exc:
        return {"symbol": symbol, "error": f"feature_build_failed: {exc}"}

    try:
        labels = make_labels(candles["close"], horizon, target_bps).reindex(feats.index).dropna()
    except Exception as exc:
        return {"symbol": symbol, "error": f"label_make_failed: {exc}"}

    feats = feats.loc[labels.index]
    y = labels.values.astype(int)
    if len(y) == 0 or feats.empty:
        return {"symbol": symbol, "error": "insufficient_samples"}

    try:
        rule_sig = _rule_signal(candles["close"], cfg.get("signals", {})).reindex(labels.index).fillna(0).astype(int)
    except Exception as exc:
        return {"symbol": symbol, "error": f"rule_signal_failed: {exc}"}

    proba = np.zeros_like(y, dtype=float)
    try:
        registry = json.load(open(ml_cfg.get("registry_path", "models/registry.json"), "r", encoding="utf-8"))
        meta = registry.get(symbol)
        if meta:
            blob = Path(meta["model_path"])
            if blob.exists():
                import joblib

                model_pack = joblib.load(blob)
                model = model_pack["model"]
                feature_names = model_pack["features"]
                X = feats[feature_names].values
                proba = model.predict_proba(X)[:, 1]
    except Exception:
        proba = np.zeros_like(y, dtype=float)

    ml_signal = (proba >= prob_threshold).astype(int)

    return {
        "symbol": symbol,
        "n": int(len(y)),
        "ml": _metrics(y, ml_signal, proba),
        "rule": _metrics(y, rule_sig.values, None),
    }


def main() -> None:
    cfg = load_json("config/strategy_config.json")

    import os
    if os.environ.get("ML_SIMULATOR", "0") == "1":
        import numpy as np
        import pandas as pd
        from execution.ml import data as data_mod

        def _fake_klines(symbol, interval, limit):
            idx = pd.date_range("2024-01-01", periods=limit, freq="H", tz="UTC")
            drift = np.linspace(0, 0.004 * limit, limit)
            noise = np.random.normal(0, 0.25, size=limit).cumsum()
            base = 100 + drift + noise
            df = pd.DataFrame(
                {
                    "open": base,
                    "high": base * 1.001,
                    "low": base * 0.999,
                    "close": base,
                    "volume": 1.0,
                },
                index=idx,
            )
            rows = []
            for ts, row in df.iterrows():
                rows.append(
                    [
                        int(ts.value // 10**6),
                        float(row["open"]),
                        float(row["high"]),
                        float(row["low"]),
                        float(row["close"]),
                        float(row["volume"]),
                    ]
                )
            data_mod.get_klines = _fake_klines  # type: ignore[attr-defined]

    symbols = cfg.get("ml", {}).get("symbols", [])
    results = [evaluate_symbol(cfg, sym) for sym in symbols]

    valid = [entry for entry in results if "error" not in entry]
    errors = [entry for entry in results if "error" in entry]

    def _agg(key: str) -> float:
        vals = [entry[key]["f1"] for entry in valid if key in entry and not np.isnan(entry[key]["f1"])]
        return float(np.mean(vals)) if vals else float("nan")

    report = {
        "symbols": results,
        "aggregate": {
            "ml_f1": _agg("ml"),
            "rule_f1": _agg("rule"),
            "n_symbols_ok": len(valid),
            "n_symbols_err": len(errors),
        },
        "errors": errors,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }
    Path("models").mkdir(exist_ok=True)
    with open("models/signal_eval.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
