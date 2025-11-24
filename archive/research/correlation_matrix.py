from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

CorrelationMethod = Literal["pearson", "spearman"]

DEFAULT_OUTPUT_PATH = Path("logs/cache/strategy_correlation.json")

__all__ = [
    "DEFAULT_OUTPUT_PATH",
    "CorrelationSnapshot",
    "compute_snapshot",
    "persist_snapshot",
    "load_returns",
    "build_snapshot_from_source",
]


@dataclass(slots=True)
class CorrelationSnapshot:
    """Container for a rolling correlation matrix and basic portfolio diagnostics."""

    asof: pd.Timestamp
    window: int
    method: CorrelationMethod
    matrix: pd.DataFrame
    average_abs_correlation: float
    max_correlation: float
    min_correlation: float

    def to_dict(self) -> Mapping[str, object]:
        """Serialize the snapshot into JSON friendly primitives."""
        return {
            "asof": self.asof.isoformat(),
            "window": int(self.window),
            "method": self.method,
            "average_abs_correlation": float(self.average_abs_correlation),
            "max_correlation": float(self.max_correlation),
            "min_correlation": float(self.min_correlation),
            "matrix": {
                str(index): {str(col): float(value) for col, value in row.items()}
                for index, row in self.matrix.round(6).to_dict(orient="index").items()
            },
        }


def _ensure_dataframe(returns: pd.DataFrame | Mapping[str, Sequence[float]] | Sequence[Mapping[str, float]]) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    elif isinstance(returns, Mapping):
        frame = pd.DataFrame(returns)
    else:
        frame = pd.DataFrame(list(returns))
    if frame.empty:
        raise ValueError("cannot compute correlations on an empty dataset")
    if frame.index.name is None:
        frame.index.name = "timestamp"
    return frame.sort_index()


def _rolling_correlation(
    frame: pd.DataFrame,
    *,
    window: int,
    min_periods: int,
    method: CorrelationMethod,
) -> pd.DataFrame:
    if method == "pearson":
        return frame.rolling(window=window, min_periods=min_periods).corr()
    if method == "spearman":
        ranked = frame.rank(axis=0, method="average", na_option="keep")
        return ranked.rolling(window=window, min_periods=min_periods).corr()
    raise ValueError(f"unsupported correlation method: {method}")


def _latest_correlation_matrix(
    returns: pd.DataFrame,
    *,
    window: int,
    method: CorrelationMethod,
    min_periods: Optional[int],
) -> tuple[pd.Timestamp, pd.DataFrame]:
    min_periods = min_periods or min(window, max(5, window // 2))
    rolled = _rolling_correlation(returns, window=window, min_periods=min_periods, method=method)
    if rolled.empty:
        raise ValueError("not enough observations for the requested window")
    timestamps = rolled.index.get_level_values(0)
    latest_ts = timestamps.max()
    matrix = rolled.loc[latest_ts].reindex(index=returns.columns, columns=returns.columns)
    matrix = matrix.fillna(0.0)
    matrix = matrix.clip(lower=-1.0, upper=1.0)
    # enforce symmetry and diagonal of ones
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix.values, 1.0)
    return latest_ts, matrix


def _off_diagonal_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(copy=True)
    mask = ~np.eye(values.shape[0], dtype=bool)
    return values[mask]


def compute_snapshot(
    returns: pd.DataFrame | Mapping[str, Sequence[float]] | Sequence[Mapping[str, float]],
    *,
    window: int = 90,
    method: CorrelationMethod = "pearson",
    min_periods: Optional[int] = None,
) -> CorrelationSnapshot:
    """Return the most recent rolling correlation snapshot for the provided returns."""
    frame = _ensure_dataframe(returns)
    latest_ts, matrix = _latest_correlation_matrix(frame, window=window, method=method, min_periods=min_periods)
    off_diag = _off_diagonal_values(matrix)
    if off_diag.size:
        average_abs = float(np.mean(np.abs(off_diag)))
        max_corr = float(np.max(off_diag))
        min_corr = float(np.min(off_diag))
    else:
        average_abs = 0.0
        max_corr = 0.0
        min_corr = 0.0
    return CorrelationSnapshot(
        asof=pd.Timestamp(latest_ts),
        window=window,
        method=method,
        matrix=matrix,
        average_abs_correlation=average_abs,
        max_correlation=max_corr,
        min_correlation=min_corr,
    )


def persist_snapshot(snapshot: CorrelationSnapshot, *, path: Path = DEFAULT_OUTPUT_PATH) -> None:
    payload = snapshot.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_returns(source: Path) -> pd.DataFrame:
    """Load strategy returns from a CSV/JSON file or a directory of CSV files."""
    if not source.exists():
        raise FileNotFoundError(source)

    frames: list[pd.DataFrame] = []
    if source.is_dir():
        files = sorted(path for path in source.glob("*.csv") if path.is_file())
        if not files:
            raise ValueError(f"no CSV files found in {source}")
        for file in files:
            frames.append(_frame_from_file(file, column_name=file.stem))
    else:
        frames.append(_frame_from_file(source))

    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.loc[:, ~combined.columns.duplicated()]
    if combined.empty:
        raise ValueError("combined returns frame is empty")
    return combined


def _frame_from_file(path: Path, column_name: Optional[str] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        frame = _ensure_dataframe(payload)
    else:
        frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.set_index("timestamp")
    if column_name:
        target_col = _find_value_column(frame)
        frame = frame.rename(columns={target_col: column_name})
        frame = frame[[column_name]]
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if not numeric_cols:
        raise ValueError(f"no numeric columns found in {path}")
    frame = frame[numeric_cols]
    return frame.astype(float)


def _find_value_column(frame: pd.DataFrame) -> str:
    numeric_cols = [col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])]
    if not numeric_cols:
        raise ValueError("no numeric columns available in returns file")
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    if "return" in frame.columns:
        return "return"
    if "pnl" in frame.columns:
        return "pnl"
    return numeric_cols[0]


def build_snapshot_from_source(
    source: Path,
    *,
    window: int,
    method: CorrelationMethod,
    output: Optional[Path] = None,
    min_periods: Optional[int] = None,
) -> CorrelationSnapshot:
    returns = load_returns(source)
    snapshot = compute_snapshot(returns, window=window, method=method, min_periods=min_periods)
    if output is not None:
        persist_snapshot(snapshot, path=output)
    return snapshot


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute rolling correlation matrices for strategy returns.")
    parser.add_argument("source", type=str, help="CSV/JSON file or directory containing strategy returns.")
    parser.add_argument("--window", type=int, default=90, help="Rolling lookback window length.")
    parser.add_argument(
        "--method",
        choices=("pearson", "spearman"),
        default="pearson",
        help="Correlation method to use.",
    )
    parser.add_argument("--min-periods", type=int, default=None, help="Minimum observations required for correlation.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to persist the computed snapshot (JSON).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        output_path = Path(args.output) if args.output else None
        snapshot = build_snapshot_from_source(
            Path(args.source),
            window=max(5, int(args.window)),
            method=args.method,
            output=output_path,
            min_periods=args.min_periods,
        )
    except Exception as exc:  # pragma: no cover - click CLI
        print(f"[correlation_matrix] failed: {exc}")
        return 1
    print(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
