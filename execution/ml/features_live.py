from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from execution.ml.features import build_feature_frame


def prepare_live_features(candles: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Return the latest feature row for live inference.

    Applies a lightweight tail window before delegating to the offline feature builder.
    """

    if candles is None or candles.empty:
        return pd.DataFrame()

    ml_cfg = cfg.get("ml", {}) if isinstance(cfg, Mapping) else {}
    lookback = ml_cfg.get("feature_lookback", 240)
    try:
        lookback = int(lookback)
    except Exception:
        lookback = 240
    lookback = max(lookback, 60)
    frame = build_feature_frame(candles.tail(lookback), cfg)
    if frame.empty:
        return frame
    return frame.tail(1)


__all__ = ["prepare_live_features"]
