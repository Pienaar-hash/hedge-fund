from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from execution.ml.data import load_candles
from execution.ml.features import build_feature_frame, make_labels


def _ts() -> int:
    return int(time.time())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
) -> LogisticRegression:
    params = {
        "max_iter": 400,
        "n_jobs": None,
        "class_weight": (
            cfg.get("ml", {}).get("trainer", {}).get("class_weight", "balanced")
        ),
    }
    model = LogisticRegression(
        **{k: v for k, v in params.items() if v is not None}
    )
    model.fit(X, y)
    return model


def _rolling_splits(
    n_samples: int,
    *,
    min_train: int,
    test_size: int,
    max_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    train_end = min_train
    while train_end + test_size <= n_samples and len(splits) < max_folds:
        train_idx = np.arange(train_end)
        test_idx = np.arange(train_end, train_end + test_size)
        splits.append((train_idx, test_idx))
        train_end += test_size
    if len(splits) < max_folds and train_end < n_samples:
        remaining = n_samples - train_end
        if remaining >= max(8, test_size // 2) and train_end >= min_train:
            train_idx = np.arange(train_end)
            test_idx = np.arange(train_end, n_samples)
            splits.append((train_idx, test_idx))
    return splits


def _calibrate_model(
    base_model: LogisticRegression,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    method: str,
) -> CalibratedClassifierCV:
    calibrator = CalibratedClassifierCV(
        base_model,
        method=method,
        cv="prefit",
    )
    calibrator.fit(X_calib, y_calib)
    return calibrator


def _calibration_plot_path(symbol: str) -> Path:
    directory = Path("logs") / "ml"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{symbol}_calibration.png"


def _write_calibration_plot(
    symbol: str,
    y_true: Sequence[int],
    y_prob: Sequence[float],
) -> str:
    plot_path = _calibration_plot_path(symbol)
    if not y_prob or len(set(y_true)) < 2:
        if plot_path.exists():
            plot_path.unlink()
        return ""
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )
    plt.figure(figsize=(4.5, 4.0))
    plt.plot(prob_pred, prob_true, marker="o", label="Observed")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"{symbol} Calibration")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return str(plot_path)


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        fval = float(value)
    except Exception:
        return None
    if not math.isfinite(fval):
        return None
    return fval


def _date_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, (np.datetime64, pd.Timestamp)):
            return pd.Timestamp(value).isoformat()
        return str(value)
    except Exception:
        return None


def _build_fold_metrics(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    index: Sequence[Any],
    auc: float,
    brier: float,
) -> Dict[str, Any]:
    train_end = _date_str(index[train_idx[-1]]) if len(train_idx) > 0 else None
    test_start = _date_str(index[test_idx[0]]) if len(test_idx) > 0 else None
    test_end = _date_str(index[test_idx[-1]]) if len(test_idx) > 0 else None
    return {
        "fold": fold_idx,
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "roc_auc": _safe_float(auc),
        "brier": _safe_float(brier),
    }


def train_symbol(cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    ml_cfg = cfg.get("ml", {})
    interval = ml_cfg.get("timeframe", "1h")
    lookback = int(ml_cfg.get("lookback_bars", 2000))
    horizon = int(ml_cfg.get("horizon_bars", 4))
    target_bps = float(ml_cfg.get("target_ret_bps", 20))
    trainer_cfg = ml_cfg.get("trainer", {})
    desired_folds = int(trainer_cfg.get("walk_forward_folds", 5))
    calibration_method = str(trainer_cfg.get("calibration", "sigmoid")).lower()
    if calibration_method not in {"sigmoid", "isotonic"}:
        calibration_method = "sigmoid"

    candles = load_candles(symbol, interval, lookback + horizon + 20)
    feats = build_feature_frame(candles, cfg)
    labels = (
        make_labels(candles["close"], horizon, target_bps)
        .reindex(feats.index)
        .dropna()
    )
    feats = feats.loc[labels.index]
    if feats.empty:
        raise ValueError("insufficient features for training")

    X = feats.values
    y = labels.values.astype(int)
    n_samples = len(y)

    test_ratio = float(trainer_cfg.get("test_size", 0.2))
    test_size = max(32, int(n_samples * max(0.05, min(test_ratio, 0.5))))
    min_train = max(128, test_size * 2)
    if n_samples <= (min_train + test_size):
        raise ValueError("not enough samples for walk-forward evaluation")

    splits = _rolling_splits(
        n_samples,
        min_train=min_train,
        test_size=test_size,
        max_folds=max(1, desired_folds),
    )
    if not splits:
        raise ValueError("unable to produce walk-forward folds with available data")

    fold_metrics: List[Dict[str, Any]] = []
    oos_probs: List[float] = []
    oos_truth: List[int] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        if len(train_idx) <= 64 or len(test_idx) <= 8:
            continue
        calib_size = max(32, int(len(train_idx) * 0.2))
        if len(train_idx) <= calib_size + 16:
            calib_size = max(16, len(train_idx) // 3)
        if calib_size <= 0 or len(train_idx) <= calib_size + 8:
            continue
        fit_idx = train_idx[:-calib_size]
        calib_idx = train_idx[-calib_size:]
        if len(fit_idx) <= 24 or len(calib_idx) <= 8:
            continue

        model = _train_logreg(X[fit_idx], y[fit_idx], cfg)
        calibrator = _calibrate_model(model, X[calib_idx], y[calib_idx], calibration_method)
        prob_test = calibrator.predict_proba(X[test_idx])[:, 1]
        y_test = y[test_idx]

        try:
            fold_auc = roc_auc_score(y_test, prob_test) if len(set(y_test)) > 1 else float("nan")
        except Exception:
            fold_auc = float("nan")
        try:
            fold_brier = brier_score_loss(y_test, prob_test)
        except Exception:
            fold_brier = float("nan")

        fold_metrics.append(
            _build_fold_metrics(
                fold_idx,
                train_idx,
                test_idx,
                feats.index,
                fold_auc,
                fold_brier,
            )
        )
        oos_probs.extend(prob_test.tolist())
        oos_truth.extend(y_test.tolist())

    if not fold_metrics:
        raise ValueError("walk-forward evaluation produced no valid folds")

    try:
        oos_auc = roc_auc_score(oos_truth, oos_probs) if len(set(oos_truth)) > 1 else float("nan")
    except Exception:
        oos_auc = float("nan")
    try:
        oos_brier = brier_score_loss(oos_truth, oos_probs)
    except Exception:
        oos_brier = float("nan")

    calibration_plot = _write_calibration_plot(symbol, oos_truth, oos_probs)

    # Final model fit on full history (with held-out calibration tail)
    final_calib_size = max(32, int(n_samples * 0.1))
    if n_samples <= final_calib_size + 16:
        final_calib_size = max(16, n_samples // 5)
    fit_idx = np.arange(max(0, n_samples - final_calib_size))
    calib_idx = np.arange(max(0, n_samples - final_calib_size), n_samples)
    if len(fit_idx) < 32 or len(calib_idx) < 8:
        raise ValueError("not enough samples to calibrate final model")

    final_model = _train_logreg(X[fit_idx], y[fit_idx], cfg)
    final_calibrator = _calibrate_model(final_model, X[calib_idx], y[calib_idx], calibration_method)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{symbol}_model.joblib"
    joblib.dump(
        {
            "model": final_calibrator,
            "features": list(feats.columns),
            "trained_at": _now_iso(),
            "calibration": calibration_method,
        },
        model_path,
    )

    metadata = {
        "symbol": symbol,
        "timeframe": interval,
        "lookback": lookback,
        "horizon": horizon,
        "target_bps": target_bps,
        "model": "logreg",
        "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": _now_iso(),
        "date_start": _date_str(feats.index.min()),
        "date_end": _date_str(feats.index.max()),
        "folds_evaluated": len(fold_metrics),
        "oos_roc_auc": _safe_float(oos_auc),
        "oos_brier": _safe_float(oos_brier),
        "calibration_method": calibration_method,
        "calibration_plot": calibration_plot or None,
        "model_path": str(model_path),
        "fold_metrics": fold_metrics,
    }

    metadata_path = model_dir / f"{symbol}_model_metadata.json"
    _ensure_dir(metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    registry_path = Path(ml_cfg.get("registry_path", "models/registry.json"))
    _ensure_dir(registry_path)
    try:
        registry = json.load(open(registry_path, "r", encoding="utf-8"))
    except Exception:
        registry = {}

    registry_entry = {
        k: v
        for k, v in metadata.items()
        if k not in {"fold_metrics"}
    }
    registry[str(symbol)] = registry_entry
    with open(registry_path, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)

    return metadata


def _write_report_md(report_path: Path, metas: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append("# ML Training Report")
    lines.append(f"- Generated: {_now_iso()}")
    lines.append("")
    for meta in metas:
        symbol = meta.get("symbol")
        lines.append(f"## {symbol}")
        lines.append(
            f"* Version: `{meta.get('version')}` · Trained: `{meta.get('trained_at')}` · Window: {meta.get('date_start')} → {meta.get('date_end')}"
        )
        lines.append(
            f"* OOS ROC-AUC: {meta.get('oos_roc_auc')} · OOS Brier: {meta.get('oos_brier')}"
        )
        lines.append("")
        lines.append("| Fold | Train Size | Test Size | Train End | Test Start | Test End | ROC-AUC | Brier |")
        lines.append("| --- | ---: | ---: | --- | --- | --- | ---: | ---: |")
        for fold in meta.get("fold_metrics", []):
            lines.append(
                f"| {fold.get('fold')} | {fold.get('train_size')} | {fold.get('test_size')} | "
                f"{fold.get('train_end')} | {fold.get('test_start')} | {fold.get('test_end')} | "
                f"{fold.get('roc_auc')} | {fold.get('brier')} |"
            )
        plot_path = meta.get("calibration_plot")
        if plot_path:
            lines.append("")
            lines.append(f"Calibration plot: `{plot_path}`")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def train_all(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    ml_cfg = cfg.get("ml", {})
    symbols = ml_cfg.get("symbols", [])
    metas: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            metas.append(train_symbol(cfg, sym))
        except Exception as exc:
            metas.append({"symbol": sym, "error": str(exc)})

    # Build aggregated reports (only for successful symbols)
    successful = [m for m in metas if "error" not in m]
    report_dir = Path("logs") / "ml"
    if successful:
        _write_report_md(report_dir / "report.md", successful)
        summary = {
            "trained_at": _now_iso(),
            "symbols": successful,
        }
        with open(Path("models") / "last_train_report.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    return metas


__all__ = ["train_symbol", "train_all"]
