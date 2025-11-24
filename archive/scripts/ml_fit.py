#!/usr/bin/env python3
import json
from execution.utils import load_json
from execution.ml.train import train_all


def _enable_simulator_if_requested():
    import os

    if os.environ.get("ML_SIMULATOR", "0") != "1":
        return

    import numpy as np
    import pandas as pd
    from execution.ml import data as data_mod

    def _fake_klines(symbol, interval, limit):
        idx = pd.date_range("2024-01-01", periods=limit, freq="H", tz="UTC")
        drift = np.linspace(0, 0.005 * limit, limit)
        noise = np.random.normal(0, 0.2, size=limit).cumsum()
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


def main() -> None:
    _enable_simulator_if_requested()
    cfg = load_json("config/strategy_config.json")
    metas = train_all(cfg)
    print(json.dumps(metas, indent=2))


if __name__ == "__main__":
    main()
