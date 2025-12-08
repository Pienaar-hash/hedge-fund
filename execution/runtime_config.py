from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


_DEFAULT_PATH = Path(os.getenv("RUNTIME_CONFIG") or "config/runtime.yaml")


@lru_cache(maxsize=1)
def load_runtime_config(path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load runtime.yaml once per process.

    Returns {} on any read/parse error to keep router boot resilient
    when YAML dependencies or files are missing.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_PATH
    if yaml is None or not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# TWAP Config (v7.4 C1)
# ---------------------------------------------------------------------------

@dataclass
class TWAPConfig:
    """Configuration for TWAP execution of large orders."""
    enabled: bool = False
    min_notional_usd: float = 0.0
    slices: int = 1
    interval_seconds: float = 0.0


def get_twap_config(cfg: Dict[str, Any] | None = None) -> TWAPConfig:
    """
    Load TWAP config from runtime configuration.
    
    Args:
        cfg: Optional pre-loaded runtime config dict
    
    Returns:
        TWAPConfig with validated values
    """
    if cfg is None:
        cfg = load_runtime_config()
    
    router_cfg = cfg.get("router", {}) or {}
    twap_cfg = router_cfg.get("twap", {}) or {}
    
    # Parse with defaults
    enabled = bool(twap_cfg.get("enabled", False))
    min_notional_usd = float(twap_cfg.get("min_notional_usd", 0.0))
    slices = int(twap_cfg.get("slices", 1))
    interval_seconds = float(twap_cfg.get("interval_seconds", 0.0))
    
    # Safety: clamp slices to at least 1
    slices = max(1, slices)
    
    # Safety: clamp min_notional to non-negative
    min_notional_usd = max(0.0, min_notional_usd)
    
    # Safety: clamp interval to non-negative
    interval_seconds = max(0.0, interval_seconds)
    
    return TWAPConfig(
        enabled=enabled,
        min_notional_usd=min_notional_usd,
        slices=slices,
        interval_seconds=interval_seconds,
    )


__all__ = ["load_runtime_config", "TWAPConfig", "get_twap_config"]
