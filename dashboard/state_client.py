"""
Dashboard state client — single choke point for all state access.

The dashboard is an **observer**: it reads pre-published state files
from ``logs/state/`` and static config from ``config/``.  It NEVER
imports from ``execution/``.

Every getter returns a safe default on missing/corrupt files.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("dash.state_client")

STATE_DIR = Path(os.getenv("STATE_DIR") or "logs/state")
CONFIG_DIR = Path(os.getenv("CONFIG_DIR") or "config")

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    """Read a JSON file; return *default* on any failure."""
    _default = {} if default is None else default
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.debug("[state_client] failed to load %s: %s", path, exc)
    return _default


def _ensure_dict(payload: Any) -> Dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _ensure_list(payload: Any) -> List[Any]:
    return payload if isinstance(payload, list) else []


# ---------------------------------------------------------------------------
# Config readers (static files, not execution logic)
# ---------------------------------------------------------------------------

def load_risk_config() -> Dict[str, Any]:
    """Read ``config/risk_limits.json`` directly — no execution import."""
    return _ensure_dict(_load_json(CONFIG_DIR / "risk_limits.json"))


def get_risk_caps() -> Dict[str, float]:
    """
    Extract key risk caps for dashboard display.

    Previously: instantiated ``execution.risk_limits.RiskGate``.
    Now: reads the same underlying JSON directly.
    """
    cfg = load_risk_config()
    g = _ensure_dict(cfg.get("global"))
    sizing = _ensure_dict(g.get("sizing") or g)

    def _frac(val: Any) -> float:
        """Normalize percentage: >1 treated as percent, else fraction."""
        try:
            v = float(val)
        except (TypeError, ValueError):
            return 0.0
        if v <= 0:
            return 0.0
        return v / 100.0 if v > 1.0 else v

    return {
        "max_trade_nav_pct": _frac(
            sizing.get("max_trade_nav_pct") or g.get("max_trade_nav_pct") or 0.0
        ),
        "max_gross_exposure_pct": _frac(
            sizing.get("max_gross_exposure_pct")
            or sizing.get("max_portfolio_gross_nav_pct")
            or g.get("max_gross_exposure_pct")
            or 0.0
        ),
        "max_symbol_exposure_pct": _frac(
            sizing.get("max_symbol_exposure_pct")
            or g.get("max_symbol_exposure_pct")
            or 0.0
        ),
        "min_notional": float(g.get("min_notional") or sizing.get("min_notional") or 0.0),
    }


# ---------------------------------------------------------------------------
# State-file readers (logs/state/*.json)
# ---------------------------------------------------------------------------

def load_execution_health() -> Dict[str, Any]:
    """Full execution health surface (per-symbol router/risk/vol/sizing)."""
    return _ensure_dict(_load_json(STATE_DIR / "execution_health.json"))


def get_execution_health_for_symbol(symbol: str) -> Dict[str, Any]:
    """
    Return the execution-health entry for *symbol*.

    Previously: called ``execution.utils.execution_health.compute_execution_health(symbol)``.
    Now: reads from the pre-published ``execution_health.json``.
    """
    data = load_execution_health()
    for entry in _ensure_list(data.get("symbols")):
        if isinstance(entry, dict) and str(entry.get("symbol", "")).upper() == symbol.upper():
            return entry
    return {}


def load_kpis_v7() -> Dict[str, Any]:
    """KPI snapshot: ATR regime, DD state, fee/PnL ratio, router quality."""
    return _ensure_dict(_load_json(STATE_DIR / "kpis_v7.json"))


def load_risk_snapshot() -> Dict[str, Any]:
    return _ensure_dict(_load_json(STATE_DIR / "risk_snapshot.json"))


def load_router_health() -> Dict[str, Any]:
    return _ensure_dict(_load_json(STATE_DIR / "router_health.json"))


def load_expectancy_v6() -> Dict[str, Any]:
    return _ensure_dict(_load_json(STATE_DIR / "expectancy_v6.json"))


def get_rolling_expectancy(symbol: str) -> Optional[float]:
    """
    Per-symbol rolling expectancy from ``expectancy_v6.json``.

    Previously: called ``execution.utils.expectancy.rolling_expectancy(symbol)``.
    Now: reads from the pre-published state file.
    """
    data = load_expectancy_v6()
    symbols_map = _ensure_dict(data.get("symbols"))
    entry = _ensure_dict(symbols_map.get(symbol) or symbols_map.get(symbol.upper()))
    val = entry.get("expectancy")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def get_hourly_expectancy(symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Hourly expectancy from ``expectancy_v6.json``.

    Previously: called ``execution.intel.expectancy_map.hourly_expectancy(symbol)``.
    Now: reads from the pre-published state file (``hours`` key).
    """
    data = load_expectancy_v6()
    if symbol:
        sym_data = _ensure_dict(
            _ensure_dict(data.get("symbols")).get(symbol)
            or _ensure_dict(data.get("symbols")).get(symbol.upper())
        )
        return _ensure_dict(sym_data.get("hours"))
    return _ensure_dict(data.get("hours"))


def load_symbol_scores_v6() -> Dict[str, Any]:
    return _ensure_dict(_load_json(STATE_DIR / "symbol_scores_v6.json"))


def get_symbol_score(symbol: str) -> Dict[str, Any]:
    """
    Per-symbol score dict from ``symbol_scores_v6.json``.

    Previously: called ``execution.intel.symbol_score.compute_symbol_score(symbol)``.
    Now: reads from the pre-published state file.
    """
    data = load_symbol_scores_v6()
    for entry in _ensure_list(data.get("symbols")):
        if isinstance(entry, dict) and str(entry.get("symbol", "")).upper() == symbol.upper():
            return entry
    return {}


# ---------------------------------------------------------------------------
# Mirror payloads (local JSONL fallback for exec snapshot panel)
# ---------------------------------------------------------------------------

_MIRROR_TAIL_MAX_BYTES = 128 * 1024
_MIRROR_TAIL_MAX_LINES = 800
_MIRROR_WINDOW_SECONDS = 86400.0
_MIRROR_MAX_ITEMS = 200


def _tail_jsonl(path: Path, *, max_lines: int = _MIRROR_TAIL_MAX_LINES) -> List[Dict[str, Any]]:
    """Read recent records from a JSONL file (last N lines)."""
    if not path.exists():
        return []
    try:
        with path.open("rb") as fh:
            size = path.stat().st_size
            fh.seek(max(0, size - _MIRROR_TAIL_MAX_BYTES))
            chunk = fh.read()
    except Exception:
        return []
    import time
    cutoff = time.time() - _MIRROR_WINDOW_SECONDS
    records: List[Dict[str, Any]] = []
    for line in chunk.decode(errors="ignore").splitlines()[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            ts = rec.get("ts") or rec.get("timestamp") or 0
            try:
                if float(ts) >= cutoff:
                    records.append(rec)
            except (TypeError, ValueError):
                records.append(rec)
    return records


def build_mirror_payloads(log_dir: Path) -> Dict[str, Any]:
    """
    Build router/trades/signals payloads from JSONL log files.

    Previously: imported ``execution.mirror_builders.build_mirror_payloads``.
    Now: reimplemented locally using only stdlib (the original had
    zero execution imports — pure JSONL parsing).
    """
    router = _tail_jsonl(log_dir / "router_health.jsonl")[-_MIRROR_MAX_ITEMS:]
    trades = _tail_jsonl(log_dir / "execution" / "trades.jsonl")[-_MIRROR_MAX_ITEMS:]
    signals = _tail_jsonl(log_dir / "execution" / "signals.jsonl")[-500:]
    return {
        "router": router,
        "trades": trades,
        "signals": signals,
    }
