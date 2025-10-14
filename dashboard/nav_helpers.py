"""Helpers for dashboard NAV and screener metrics."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from pandas.io.formats.style import Styler

_UNITS_FMT = "{:.6f}"
_USD_FMT = "{:,.2f}"


def treasury_table_from_summary(summary: Dict[str, Any]) -> pd.DataFrame:
    """Return a DataFrame of treasury assets from compute_nav_summary output.

    Expected summary shape:
      {
        "details": {
            "treasury": {
                "treasury": {
                    "BTC": {"qty": 0.1, "val_usdt": 1000.0, ...},
                    ...
                }
            }
        }
      }
    """

    treasury_detail = (summary.get("details") or {}).get("treasury", {})
    holdings = treasury_detail.get("treasury", {}) if isinstance(treasury_detail, dict) else {}
    rows = []
    for asset, info in holdings.items():
        try:
            qty = float(info.get("qty", 0.0) or 0.0)
            usd = float(info.get("val_usdt", 0.0) or 0.0)
        except Exception:
            continue
        rows.append({"Asset": str(asset), "Units": qty, "USD Value": usd})

    if not rows:
        return pd.DataFrame(columns=["Asset", "Units", "USD Value"])

    df = pd.DataFrame(rows)
    df = df.sort_values(by="USD Value", ascending=False).reset_index(drop=True)
    return df


def format_treasury_table(df: pd.DataFrame) -> Styler:
    """Format treasury DataFrame for display."""

    return df.style.format({"Units": _UNITS_FMT, "USD Value": _USD_FMT})


def signal_attempts_summary(lines: List[str]) -> str:
    """Return latest screener attempted/emitted summary string."""

    for line in reversed(lines):
        if "attempted=" not in line or "emitted=" not in line:
            continue
        attempted = _extract_int(line, "attempted")
        emitted = _extract_int(line, "emitted")
        if attempted is None or emitted is None:
            continue
        pct = (emitted / attempted * 100.0) if attempted else 0.0
        pct_display = f" ({pct:.0f}%)" if attempted else ""
        return f"Signals: {attempted} attempted, {emitted} emitted{pct_display}"
    return "Signals: N/A"


def _extract_int(text: str, key: str) -> int | None:
    needle = f"{key}="
    start = text.find(needle)
    if start == -1:
        return None
    start += len(needle)
    end = start
    while end < len(text) and text[end].isdigit():
        end += 1
    try:
        return int(text[start:end])
    except Exception:
        return None


__all__ = [
    "treasury_table_from_summary",
    "format_treasury_table",
    "signal_attempts_summary",
]
