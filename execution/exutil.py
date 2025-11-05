from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

try:  # dotenv is optional in production images
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


_LOG = logging.getLogger("exutil.env")
if not _LOG.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [exutil] %(message)s")
    )
    _LOG.addHandler(_handler)
_LOG.setLevel(logging.INFO)

_DEFAULT_ENV = "prod"
_DOTENV_PATHS = (
    Path(".env"),
    Path("/root/hedge-fund/.env"),
)


def _load_dotenvs() -> Tuple[str, ...]:
    """Load known dotenv files before resolving ENV."""
    if load_dotenv is None:
        _LOG.info("[exutil] python-dotenv unavailable; skipping dotenv load")
        return ()
    loaded: list[str] = []
    for candidate in _DOTENV_PATHS:
        try:
            path = candidate if candidate.is_absolute() else Path.cwd() / candidate
            if not path.exists():
                continue
            if load_dotenv(path, override=True):
                loaded.append(str(path))
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("[exutil] dotenv load failed path=%s err=%s", candidate, exc)
    if loaded:
        _LOG.info("[exutil] dotenv loaded paths=%s", ",".join(loaded))
    else:
        _LOG.info("[exutil] dotenv not loaded (paths missing)")
    return tuple(loaded)


def _resolve_env(default: str = _DEFAULT_ENV) -> Tuple[str, str]:
    """Resolve ENV from known environment variables with fallback."""
    for key in ("ENV", "HEDGE_ENV", "ENVIRONMENT"):
        raw = os.environ.get(key)
        if raw and raw.strip():
            return raw.strip(), key
    return default, "default"


_load_dotenvs()
ENV, ENV_SOURCE = _resolve_env()
os.environ["ENV"] = ENV  # normalise for downstream imports
_LOG.info("[exutil] ENV resolved as %s source=%s", ENV, ENV_SOURCE)


def refresh_env(default: str = _DEFAULT_ENV) -> str:
    """Re-resolve ENV (after external changes) and broadcast via os.environ."""
    global ENV, ENV_SOURCE
    ENV, ENV_SOURCE = _resolve_env(default)
    os.environ["ENV"] = ENV
    _LOG.info("[exutil] ENV resolved as %s source=%s", ENV, ENV_SOURCE)
    return ENV


def current_env() -> str:
    return ENV


__all__ = ["ENV", "ENV_SOURCE", "refresh_env", "current_env"]
