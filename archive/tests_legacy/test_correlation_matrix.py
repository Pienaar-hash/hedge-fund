from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.correlation_matrix import (
    build_snapshot_from_source,
    compute_snapshot,
    persist_snapshot,
)


def _sample_returns() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    data = {
        "momentum": np.sin(np.linspace(0, 6, len(dates))) * 0.02 + np.random.default_rng(42).normal(0, 0.01, len(dates)),
        "mean_reversion": np.cos(np.linspace(0, 4, len(dates))) * 0.015 + np.random.default_rng(7).normal(0, 0.012, len(dates)),
        "carry": np.random.default_rng(21).normal(0.0005, 0.008, len(dates)),
    }
    frame = pd.DataFrame(data, index=dates)
    return frame


def test_compute_snapshot_symmetry(tmp_path: Path) -> None:
    returns = _sample_returns()
    snapshot = compute_snapshot(returns, window=30, method="pearson")
    matrix = snapshot.matrix

    assert np.allclose(matrix.values, matrix.values.T, atol=1e-9)
    assert np.allclose(np.diag(matrix.values), np.ones(len(matrix.columns)))
    assert snapshot.average_abs_correlation >= 0.0
    assert snapshot.max_correlation <= 1.0
    assert snapshot.min_correlation >= -1.0

    payload_path = tmp_path / "correlation.json"
    persist_snapshot(snapshot, path=payload_path)
    persisted = json.loads(payload_path.read_text())
    assert persisted["method"] == "pearson"
    assert persisted["window"] == 30
    assert set(persisted["matrix"].keys()) == set(returns.columns)


def test_build_snapshot_from_directory(tmp_path: Path) -> None:
    returns = _sample_returns().iloc[:40]
    for column in returns.columns:
        df = pd.DataFrame({"timestamp": returns.index, "return": returns[column].values})
        df.to_csv(tmp_path / f"{column}.csv", index=False)

    snapshot = build_snapshot_from_source(tmp_path, window=15, method="spearman")
    assert snapshot.method == "spearman"
    assert snapshot.matrix.shape == (3, 3)
    assert sorted(snapshot.matrix.index.tolist()) == sorted(list(returns.columns))
