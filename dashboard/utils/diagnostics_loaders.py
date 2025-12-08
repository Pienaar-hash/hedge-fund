from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def load_equity_state(path: str | Path = "logs/state/equity.json") -> dict:
    payload = _load_json(Path(path))
    return payload if isinstance(payload, dict) else {}


def load_positions_state(path: str | Path = "logs/state/positions.json") -> list:
    payload = _load_json(Path(path))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def load_pnl_attribution_state(path: str | Path = "logs/state/pnl_attribution.json") -> dict:
    payload = _load_json(Path(path))
    return payload if isinstance(payload, dict) else {}
