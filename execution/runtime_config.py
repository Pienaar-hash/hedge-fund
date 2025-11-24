from __future__ import annotations

import os
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


__all__ = ["load_runtime_config"]
