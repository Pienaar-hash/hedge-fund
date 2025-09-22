import numpy as np
import pandas as pd

def talib_safe_rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def zscore(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window)
    return (series - roll.mean()) / (roll.std() + 1e-9)


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def build_feature_frame(df: pd.DataFrame, cfg: dict, ob_imbalance: float | None = None) -> pd.DataFrame:
    ml_cfg = cfg.get("ml", {})
    feat_cfg = ml_cfg.get("features", {})
    close = df["close"]
    feats = pd.DataFrame(index=df.index)
    feats["ret_1"] = close.pct_change().fillna(0.0)
    feats["ret_5"] = close.pct_change(5).fillna(0.0)
    feats["ema_fast"] = ema(close, feat_cfg.get("ema_fast", 20))
    feats["ema_slow"] = ema(close, feat_cfg.get("ema_slow", 50))
    feats["ema_diff"] = (feats["ema_fast"] - feats["ema_slow"]) / (close + 1e-9)
    feats["rsi"] = talib_safe_rsi(close, feat_cfg.get("rsi_len", 14)) / 100.0
    feats["zscore"] = (
        zscore(close.pct_change().fillna(0.0), feat_cfg.get("zscore_lookback", 48))
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )
    if ob_imbalance is not None:
        feats["ob_imb_topn"] = float(ob_imbalance)
    return feats.replace([np.inf, -np.inf], 0.0).dropna().astype(float)


def make_labels(close: pd.Series, horizon: int, target_bps: float) -> pd.Series:
    forward_ret = close.shift(-horizon) / close - 1.0
    threshold = target_bps / 10000.0
    labels = (forward_ret >= threshold).astype(int)
    return labels
