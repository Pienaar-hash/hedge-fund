from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List

__all__ = ["Strategy", "log_strategy_outputs"]

_LOG = logging.getLogger("strategy")


class Strategy:
    """Minimal base class used by strategy modules."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self.trades_df: Any = None
        self.results: Any = None

    def prepare(self, data_ctx: Any) -> None:
        """Hook for loading data or shared state before generating signals."""

    def signals(self, now: Any) -> List[dict[str, Any]]:
        """Return a list of live intents. Default: no signals."""
        return []

    def log_results(self) -> None:
        """Persist trade dataframe if present."""
        df = getattr(self, "trades_df", None)
        if df is None or not hasattr(df, "to_csv"):
            return
        try:
            Path("logs").mkdir(exist_ok=True)
            label = getattr(self, "label", self.__class__.__name__.lower())
            path = Path("logs") / f"{label}_trades.csv"
            df.to_csv(path, index=False)
        except Exception as exc:
            _LOG.debug("log_results failed: %s", exc)


def log_strategy_outputs(df: Any, label: str | None = None) -> Path | None:
    """Helper for strategies that write their trade dataframe to disk."""
    if df is None or not hasattr(df, "to_csv"):
        return None
    try:
        Path("logs").mkdir(exist_ok=True)
        safe_label = (label or "strategy").replace(" ", "_")
        path = Path("logs") / f"{safe_label}_trades.csv"
        df.to_csv(path, index=False)
        return path
    except Exception as exc:
        _LOG.debug("log_strategy_outputs failed: %s", exc)
        return None
