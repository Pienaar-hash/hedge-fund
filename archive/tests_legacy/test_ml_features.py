import numpy as np
import pandas as pd

from execution.ml.features import build_feature_frame, make_labels


def test_features_and_labels():
    idx = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )
    cfg = {
        "ml": {
            "features": {
                "ema_fast": 5,
                "ema_slow": 10,
                "rsi_len": 14,
                "zscore_lookback": 20,
                "ob_imbalance_topn": 20,
            }
        }
    }
    feats = build_feature_frame(df, cfg)
    for col in ["ret_1", "ret_5", "ema_diff", "rsi", "zscore"]:
        assert col in feats.columns
    labels = make_labels(close, horizon=4, target_bps=10).reindex(feats.index).dropna()
    assert set(labels.unique()).issubset({0, 1})
