import numpy as np
import pandas as pd

from execution.ml.features import build_feature_frame, make_labels


def test_smoke_features_labels_shape():
    idx = pd.date_range("2024-01-01", periods=200, freq="H", tz="UTC")
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
            }
        }
    }
    feats = build_feature_frame(df, cfg)
    labels = make_labels(close, horizon=4, target_bps=10).reindex(feats.index).dropna()
    assert feats.loc[labels.index].shape[0] == labels.shape[0]
